"""Analytic, invariant, failure/recovery, and end-to-end regression tests."""
from __future__ import annotations
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import json
import sys
import tokenize
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models import ModelSpec,PotentialModel,preset_specs
from simulation import SimConfig,simulate,core_transitions,NUMBA_AVAILABLE
from spectra import (SpectralConfig,make_bands,compute_psd,summarize_psd,ratios,
                     statistic,summarize_coherent)
from analysis import run_experiment,validate_experiment,detail_simulation,scan_evidence
from exporting import experiment_zip
import fokker_planck as fp
import kramers
from plotting import potential_figure,sweep_figure,time_figure,psd_figure,density_figure


@pytest.fixture
def config():
    return SimConfig(f=.1,dt=.02,cycles=8,burn_cycles=2,trials=4,segment_cycles=8)


@pytest.fixture
def bands(config):
    return make_bands(config.actual_dt,config.nperseg,config.f,SpectralConfig())


def test_symmetric_extrema_and_barrier():
    m=PotentialModel(ModelSpec())
    e=m.extrema()
    np.testing.assert_allclose([p.x for p in e],[-1,0,1],atol=1e-12)
    np.testing.assert_allclose([p.curvature for p in e],[2,-1,2])
    np.testing.assert_allclose([b.delta_u for b in m.barriers()],[.25,.25])


def test_asymmetric_directional_heights():
    m=PotentialModel(ModelSpec(kind='asymmetric',c=.1))
    edges=m.barriers()
    assert len(edges)==2 and edges[0].delta_u>edges[1].delta_u


def test_three_well_exact_geometry():
    m=PotentialModel(ModelSpec(kind='multi'))
    minima=[e.x for e in m.extrema() if e.kind=='minimum']
    np.testing.assert_allclose(minima,[-1,0,1],atol=1e-12)
    np.testing.assert_allclose([b.delta_u for b in m.barriers()],4/27,atol=1e-12)


@pytest.mark.parametrize('n',[1,2,3,4,5])
def test_barrier_count_family(n):
    m=PotentialModel(ModelSpec(kind='barriers',wells=n,height=.25,c=0))
    assert sum(e.kind=='minimum' for e in m.extrema())==n
    assert sum(e.kind=='maximum' for e in m.extrema())==n-1
    if n>1:
        np.testing.assert_allclose([b.delta_u for b in m.barriers()],.25,rtol=1e-7)


def test_height_change_preserves_minima():
    a=PotentialModel(ModelSpec(a=.5,b=.5));b=PotentialModel(ModelSpec(a=2,b=2))
    np.testing.assert_allclose([e.x for e in a.extrema()],[e.x for e in b.extrema()])
    assert np.isclose(b.barriers()[0].delta_u/a.barriers()[0].delta_u,4)


def test_nonconfining_rejected_and_recovery():
    with pytest.raises(ValueError): PotentialModel(ModelSpec(kind='multi',a6=-1))
    assert len(PotentialModel(ModelSpec(kind='multi')).barriers())==4


def test_destroyed_well_not_bistable():
    m=PotentialModel(ModelSpec(kind='asymmetric',c=1))
    assert len(m.barriers())==0
    with pytest.raises(ValueError): kramers.generator(m,.1)


def test_flat_minimum_not_kramers():
    m=PotentialModel(ModelSpec(kind='mono',k=0,b=1))
    assert m.extrema()[0].kind=='flat'
    with pytest.raises(ValueError): kramers.generator(m,.1)


def test_unequal_barriers():
    m=PotentialModel(ModelSpec(kind='barriers',wells=3,c=0,skew=.7))
    heights=[b.delta_u for b in m.barriers()]
    assert max(heights)>min(heights)*1.3


def test_force_derivative():
    m=PotentialModel(ModelSpec(kind='multi',a1=.1));x=np.linspace(-1.5,1.5,20);h=1e-5
    numerical=(m.potential(x+h)-m.potential(x-h))/(2*h)
    np.testing.assert_allclose(m.force(x),-numerical,atol=1e-7)


def test_memoryless_state_mapping():
    m=PotentialModel(ModelSpec(kind='multi'))
    np.testing.assert_array_equal(m.observable(np.array([-1.,0.,1.]),'q'),[-1,0,1])
    with pytest.raises(ValueError): PotentialModel(ModelSpec(kind='mono',b=0)).observable([0],'q')


def test_generator_conserves_probability():
    m=PotentialModel(ModelSpec());g=fp.generator(m,.1,fp.make_grid((-3,3),241))
    assert abs(g.apply(np.random.default_rng(1).random(241)).sum())<1e-8
    assert np.all(g.up>=0) and np.all(g.down>=0)
    assert np.linalg.norm(g.apply(g.probability),1)<1e-10


def test_ou_stationary_variance():
    m=PotentialModel(ModelSpec(kind='mono',k=2,b=0))
    g=fp.generator(m,.4,fp.make_grid((-4,4),401))
    mean=g.grid@g.probability
    var=(g.grid-mean)**2@g.probability
    assert np.isclose(var,.2,rtol=1e-5)


def test_bernoulli_extreme_values():
    z=np.array([-1000.,-1e-8,0,1e-8,1000.])
    b=fp.bernoulli(z)
    assert np.all(np.isfinite(b)) and np.all(b>=0)
    np.testing.assert_allclose(b[1:4],1,rtol=1e-7)


def test_fp_rejects_zero_noise_and_bad_grid():
    m=PotentialModel(ModelSpec())
    with pytest.raises(ValueError): fp.generator(m,0,np.linspace(-2,2,100))
    with pytest.raises(ValueError): fp.generator(m,.1,np.array([0,1,2,4,8]))


def test_fp_ou_linear_response(bands):
    m=PotentialModel(ModelSpec(kind='mono',k=1,b=0))
    grid=fp.make_grid((-5,5),801);D=.2;A=.06
    r=fp.response(m,D,A,bands,'x',grid)
    omega=2*np.pi*bands.f0
    expected_amplitude=A/np.sqrt(1+omega**2)
    floor=np.mean(4*D/(1+(2*np.pi*bands.frequency[bands.noise])**2))
    expected_snr=expected_amplitude**2/2/(floor*bands.bandwidth)
    assert np.isclose(r['amplitude'],expected_amplitude,rtol=.002)
    assert np.isclose(r['snr'],expected_snr,rtol=.002)


def test_fp_no_forcing_and_quadratic_scaling(bands):
    m=PotentialModel(ModelSpec());grid=fp.make_grid((-3,3),241)
    zero=fp.response(m,.2,0,bands,'q',grid)
    one=fp.response(m,.2,.02,bands,'q',grid)
    two=fp.response(m,.2,.04,bands,'q',grid)
    assert zero['snr']==0
    assert np.isclose(two['snr']/one['snr'],4)


def test_kramers_symmetric_rate():
    m=PotentialModel(ModelSpec());g=kramers.generator(m,.1)
    expected=np.sqrt(2)/(2*np.pi)*np.exp(-2.5)
    np.testing.assert_allclose(g.up,expected)
    np.testing.assert_allclose(g.down,expected)
    np.testing.assert_allclose(g.probability,[.5,.5])


def test_multistate_exit_time_uses_sum():
    table=kramers.rate_table(PotentialModel(ModelSpec(kind='multi')),.05,.02)
    center=[r for r in table if r['source']==1]
    assert len(center)==2
    assert np.isclose(center[0]['mean_exit_time'],1/sum(r['rate'] for r in center))
    assert center[0]['mean_exit_time']<center[0]['inverse_edge_rate']


def test_topology_loss_under_forcing_not_applied(bands):
    m=PotentialModel(ModelSpec())
    assert not kramers.diagnostics(m,.1,1,.02)['applicable']
    with pytest.raises(ValueError): kramers.response(m,.1,1,bands,'q')


def test_snr_band_geometry(bands):
    assert not np.intersect1d(bands.signal,bands.noise).size
    assert min(bands.signal)>0 and min(bands.noise)>0
    assert np.isclose(bands.bandwidth,len(bands.signal)*bands.df)


def test_bad_spectral_resolution_rejected():
    with pytest.raises(ValueError): make_bands(.01,1000,.01,SpectralConfig())


def test_welch_integrated_power():
    dt=.01;t=np.arange(20000)*dt;y=2*np.cos(2*np.pi*.2*t)
    freq,psd=compute_psd(y,dt,10000)
    assert np.isclose(psd.sum()*(freq[1]-freq[0]),2,rtol=1e-10)


def test_zero_psd_is_undefined(bands):
    cfg=SpectralConfig(bootstrap=100,aggregation='pooled_psd')
    summary=summarize_psd(np.zeros((4,len(bands.frequency))),bands,cfg)
    assert np.isnan(summary['snr']) and summary['undefined_trials']==4


def test_snr_scale_invariance(bands):
    rng=np.random.default_rng(4)
    psd=rng.uniform(.5,1.5,(6,len(bands.frequency)))
    psd[:,bands.signal]+=3
    cfg=SpectralConfig(bootstrap=100,aggregation='pooled_psd')
    a=summarize_psd(psd,bands,cfg);b=summarize_psd(17*psd,bands,cfg)
    assert np.isclose(a['snr'],b['snr'])
    np.testing.assert_allclose([a['ci_low'],a['ci_high']],[b['ci_low'],b['ci_high']])


def test_aggregation_not_silently_changed():
    total=np.array([2.,12.]);noise=np.array([1.,3.])
    assert not np.isclose(statistic(total,noise,'pooled_psd'),statistic(total,noise,'trial_mean'))
    assert np.isnan(statistic([0,12],[0,3],'trial_mean'))
    assert np.isfinite(statistic([0,12],[0,3],'pooled_psd'))


def test_coherent_power_exact_signal():
    t=np.arange(1000)*.01
    y=np.tile(2*np.cos(2*np.pi*t),(8,1))
    r=summarize_coherent(t,y,1,np.full(8,.4))
    assert np.isclose(r['signal_power'],2)
    assert np.isclose(r['snr'],5)
    assert r['standard_error']<1e-12


def test_coherent_constant_signal_not_fake_peak():
    t=np.arange(1000)*.01;y=np.ones((5,len(t)))
    r=summarize_coherent(t,y,1,np.zeros(5))
    assert np.isnan(r['snr'])


def test_reproducible_and_prefix_stable_trials(config):
    m=PotentialModel(ModelSpec())
    t,a=simulate(m,config,.1)
    _,b=simulate(m,config,.1)
    _,c=simulate(m,replace(config,trials=6),.1)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(a,c[:4])
    _,independent=simulate(m,config,.1,stage=101)
    assert not np.array_equal(a,independent)


@pytest.mark.skipif(not NUMBA_AVAILABLE,reason='Optional acceleration is not installed')
def test_numpy_and_jit_agree(config):
    m=PotentialModel(ModelSpec())
    _,a=simulate(m,config,.1,backend='numpy')
    _,b=simulate(m,config,.1)
    np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-11)


def test_ou_euler_deterministic_recurrence(config):
    m=PotentialModel(ModelSpec(kind='mono',k=1,b=0))
    cfg=replace(config,A=0,x0=1,burn_cycles=0,trials=1,scheme='euler')
    t,x=simulate(m,cfg,0)
    np.testing.assert_allclose(x[0],(1-cfg.actual_dt)**np.arange(len(t)),rtol=1e-10,atol=1e-14)


def test_simulation_period_alignment(config):
    cfg=replace(config,dt=.03)
    t,x=simulate(PotentialModel(ModelSpec()),cfg,.1)
    assert cfg.actual_dt<=cfg.dt
    assert len(t)==cfg.cycles*cfg.steps_per_period
    assert np.isclose(len(t)*cfg.actual_dt*cfg.f,cfg.cycles)


def test_unstable_integrator_reports_failure(config):
    with pytest.raises(ValueError): simulate(PotentialModel(ModelSpec()),replace(config,x0=1e10),.1)


def test_resource_guard():
    with pytest.raises(ValueError): SimConfig(dt=1e-5,trials=200,cycles=400).validate()


def test_censored_residence_intervals():
    m=PotentialModel(ModelSpec());t=np.arange(8.)
    r=core_transitions(np.array([-1,-1,1,1,-1,-1,1,1.]),t,m)
    assert len(r['events'])==3 and r['completed']==2
    assert r['mean_residence']==2 and r['left_censored']==1 and r['right_censored']==1


def test_periodic_fp_ou_solution():
    m=PotentialModel(ModelSpec(kind='mono',k=1,b=0))
    r=fp.periodic_density(m,.2,.05,.05,'x',fp.make_grid((-4,4),201),dt_max=.05,max_cycles=10)
    assert r['converged'] and r['mass_error']<1e-10 and r['minimum_probability']>=-1e-12
    assert np.isclose(r['amplitude'],.05/np.sqrt(1+(2*np.pi*.05)**2),rtol=.01)


def test_fp_nonconvergence_flag():
    m=PotentialModel(ModelSpec())
    r=fp.periodic_density(m,.02,.05,.1,'q',fp.make_grid((-3,3),121),dt_max=.1,max_cycles=3)
    assert isinstance(r['converged'],bool)
    assert r['cycles']<=3 and np.isfinite(r['periodic_residual'])


def test_peak_and_boundary_classification():
    rows=[{'method':'MC','D':d,'snr':s} for d,s in zip([1,2,3,4,5],[1,2,4,2,1])]
    assert scan_evidence(rows)['candidate']
    for r in rows:r['snr']=r['D']
    assert scan_evidence(rows)['status']=='境界最大'


def test_missing_data_not_zero():
    rows=[{'method':'MC','D':d,'snr':np.nan} for d in range(5)]
    assert scan_evidence(rows)['status']=='推定不足'


@pytest.fixture(scope='module')
def experiment():
    return run_experiment(ModelSpec(),SimConfig(trials=4,cycles=8,burn_cycles=1,dt=.05),
                          SpectralConfig(bootstrap=100),np.linspace(.06,.3,5),refine=0,nx=121)


def test_three_method_end_to_end(experiment):
    assert len(experiment['rows'])==15 and len(experiment['trials'])==20
    assert {r['method'] for r in experiment['rows']}=={'MC','FP','Kramers'}
    assert all(r.get('status') not in ('計算失敗','適用外') for r in experiment['rows'])


def test_no_input_cannot_claim_resonance(config):
    e=run_experiment(ModelSpec(),replace(config,A=0),SpectralConfig(bootstrap=100),np.linspace(.05,.3,5),refine=0)
    assert not e['evidence']['candidate']
    assert '周期入力なし' in e['evidence']['status']


def test_no_candidate_validation_is_not_run(experiment):
    e=dict(experiment,evidence={'candidate':False})
    r=validate_experiment(e)
    assert r['checks']==[] and r['status']=='ピーク候補なし'


def test_csv_export_and_json_finite(experiment):
    data=experiment_zip(experiment)
    with ZipFile(BytesIO(data)) as z:
        assert 'noise_sweep.csv' in z.namelist() and 'parameters.csv' in z.namelist()
        text=z.read('experiment.json').decode()
        assert 'NaN' not in text and 'Infinity' not in text
        doc=json.loads(text)
        assert doc['simulation']['seed']==2026
        assert 'coherent_real' in z.read('trial_results.csv').decode()


def test_all_reusable_plots(experiment,bands):
    m=PotentialModel(ModelSpec())
    figs=[potential_figure(m,.06),sweep_figure(experiment['rows']),sweep_figure(experiment['rows'],True)]
    t=np.linspace(0,100,10001);x=np.sin(t)
    figs.append(time_figure(t,x,x,.06,.02))
    figs.append(psd_figure(bands.frequency,np.ones(len(bands.frequency)),bands))
    for fig in figs: assert len(fig.to_json())>100


def test_source_comments_are_english():
    for path in ROOT.glob('*.py'):
        with tokenize.open(path) as f:
            tokens=tokenize.generate_tokens(f.readline)
            for token in tokens:
                if token.type==tokenize.COMMENT:
                    assert not any('\u3040'<=c<='\u9fff' for c in token.string),path.name


def test_latex_rendering_has_balanced_braces(config):
    from equations import render_equations
    class Capture:
        def __init__(self):self.formulas=[]
        def latex(self,s):self.formulas.append(s)
        def __getattr__(self,name):
            if name=='expander':return lambda *a,**k:self
            return lambda *a,**k:None
        def __enter__(self):return self
        def __exit__(self,*args):return False
    st=Capture()
    render_equations(st,PotentialModel(ModelSpec()),config,SpectralConfig(),.1)
    assert len(st.formulas)>15
    for formula in st.formulas:
        assert formula.count('{')==formula.count('}')
        assert not any(ord(c)<32 for c in formula)

