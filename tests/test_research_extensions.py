"""Analytic noise laws, coupled refinement, censoring and nonlinear FP regressions."""
from dataclasses import replace
from io import BytesIO
from zipfile import ZipFile
import numpy as np
import pytest
from models import ModelSpec,PotentialModel
from simulation import SimConfig,simulate,coupled_simulations,seed_for
from noise import integrated_noise
from spectra import SpectralConfig,make_bands
from accuracy import equivalence_result,timestep_study,duration_study
from analysis import run_experiment,detail_simulation
from transitions import transition_analysis
from nonlinear_fp import periodic_snr
from fokker_planck import make_grid
from chaos import DuffingConfig,duffing_diagnostic,duffing_refinement
from exporting import experiment_zip


@pytest.mark.parametrize('kind',['white','ou','telegraph'])
def test_noise_integral_variance_and_stationary_initialization(kind):
    D,tau,h=.3,.5,.1
    values=integrated_noise(kind,D,h,120,250,seed_for(88,D),tau)
    expected=2*D*h if kind=='white' else 2*D*(h-tau*(-np.expm1(-h/tau)))
    assert abs(values.mean())<.005
    assert np.isclose(values.var(),expected,rtol=.06)
    if kind!='white':
        assert np.isclose(values[0].var(),expected,rtol=.25)


@pytest.mark.parametrize('kind',['ou','telegraph'])
def test_colored_noise_integral_lag_covariance(kind):
    D,tau,h=.3,.5,.1
    values=integrated_noise(kind,D,h,1000,24,seed_for(99,D),tau)
    expected=D*tau*(-np.expm1(-h/tau))**2
    observed=np.mean(values[:-1]*values[1:])
    assert np.isclose(observed,expected,rtol=.15)


def test_ou_very_small_step_remains_finite():
    x=integrated_noise('ou',.2,1e-7,20,4,seed_for(3,.2),1.)
    assert np.all(np.isfinite(x)) and np.var(x)>0


@pytest.mark.parametrize('kind',['white','ou','telegraph'])
def test_zero_noise_is_exactly_zero(kind):
    assert not integrated_noise(kind,0,.1,10,3,seed_for(4,0)).any()


def test_noise_errors_and_resource_rejection():
    with pytest.raises(ValueError): integrated_noise('unknown',.1,.1,10,3,seed_for(1,.1))
    with pytest.raises(ValueError): integrated_noise('ou',.1,.1,10,3,seed_for(1,.1),tau=0)
    with pytest.raises(ValueError): integrated_noise('white',.1,.1,25_000_000,1,seed_for(1,.1))


def test_heun_reduces_deterministic_discretization_error():
    model=PotentialModel(ModelSpec(kind='mono',b=0))
    cfg=SimConfig(A=0,f=.1,dt=.2,cycles=8,burn_cycles=0,trials=1,x0=1)
    t,e=simulate(model,replace(cfg,scheme='euler'),0)
    _,h=simulate(model,cfg,0)
    target=np.exp(-t)
    assert np.max(abs(h[0]-target))<np.max(abs(e[0]-target))/10


@pytest.mark.parametrize('kind',['white','ou','telegraph'])
def test_coupled_noise_path_has_exact_same_cumulative_forcing(kind):
    # Zero deterministic force makes the trajectory exactly the integrated noise.
    class FreeModel:
        derivative=np.polynomial.Polynomial([0.])
    cfg=SimConfig(f=.1,dt=.1,cycles=8,burn_cycles=0,trials=3,A=0,x0=0,noise_kind=kind)
    output=coupled_simulations(FreeModel(),cfg,.2)
    np.testing.assert_allclose(output[0][2],output[1][2][:,::2],atol=1e-12)
    np.testing.assert_allclose(output[0][2],output[2][2][:,::4],atol=1e-12)


def test_wide_overlapping_intervals_do_not_prove_precision():
    assert equivalence_result(.01,-.5,.5,.2)['result']=='INCONCLUSIVE'
    assert equivalence_result(.1,.08,.12,.2)['result']=='FAIL'
    assert equivalence_result(.001,-.005,.005,.2)['result']=='PASS'
    assert equivalence_result(np.nan,np.nan,np.nan,.2)['result']=='NOT RUN'


def test_three_level_precision_runs_without_selecting_peak():
    cfg=SimConfig(f=.1,dt=.05,cycles=8,burn_cycles=1,trials=6)
    checks=timestep_study(PotentialModel(ModelSpec()),cfg,SpectralConfig(),.2,'q')
    assert len(checks)==2 and all(c['inference']=='paired_jackknife_t' for c in checks)
    assert checks[1]['fine_dt']==pytest.approx(.0125)


def test_nested_observation_time_has_quantitative_difference_interval():
    cfg=SimConfig(f=.1,dt=.05,cycles=8,burn_cycles=1,trials=8)
    r=duration_study(PotentialModel(ModelSpec()),cfg,SpectralConfig(),.2,'q')
    assert r['stage']==701 and r['inference']=='paired_jackknife_t'
    assert r['difference_ci_low']<=r['difference']<=r['difference_ci_high']


def test_transition_counts_censoring_and_exposure_balance():
    model=PotentialModel(ModelSpec(kind='multi'))
    t=np.arange(10.)
    x=np.array([[-1,-1,0,0,1,1,0,0,-1,-1.]])
    r=transition_analysis(x,t,model,.1)
    assert r['counts'].sum()==4
    assert r['counts'][0,1]==r['counts'][1,2]==r['counts'][2,1]==r['counts'][1,0]==1
    assert sum(d['left_censored'] for d in r['dwell'])==1
    assert sum(d['right_censored'] for d in r['dwell'])==1
    assert sum(d['exposure'] for d in r['occupancy'])==9
    assert sum(d['fraction'] for d in r['occupancy'])==pytest.approx(1.)
    assert r['unresolved_skips']==0


def test_unknown_initial_core_and_unsampled_intermediate_state():
    model=PotentialModel(ModelSpec(kind='multi'))
    r=transition_analysis([[-.5,-1,1,1]],np.arange(4.),model,.1)
    assert r['unknown_time']==1
    assert r['unresolved_skips']==1
    assert r['counts'][0,2]==1 and r['counts'][0,1]==0


def test_no_transition_rate_is_zero_only_with_exposure():
    model=PotentialModel(ModelSpec())
    r=transition_analysis([[-1]*10],np.arange(10.),model,.1)
    left=next(e for e in r['edges'] if e['source']==0)
    right=next(e for e in r['edges'] if e['source']==1)
    assert left['empirical_rate']==0 and np.isnan(right['empirical_rate'])
    assert np.isnan(left['phase_locking'])
    assert r['dwell'][0]['left_censored'] and r['dwell'][0]['right_censored']


def test_directional_phase_locking():
    model=PotentialModel(ModelSpec())
    t=np.arange(0,50,.1)
    x=np.where(np.cos(2*np.pi*.1*t)>=0,1.,-1.)
    r=transition_analysis(x,t,model,.1)
    assert all(e['phase_locking']>.99 for e in r['edges'])
    assert all(sum(e['phase_counts'])==e['count'] for e in r['edges'])


@pytest.mark.parametrize('kind',['ou','telegraph'])
def test_colored_theories_are_inapplicable_and_exports_work(kind):
    cfg=SimConfig(f=.1,dt=.05,cycles=8,burn_cycles=1,trials=4,noise_kind=kind)
    exp=run_experiment(ModelSpec(),cfg,SpectralConfig(bootstrap=50),np.linspace(.08,.3,5),nx=121,refine=0)
    assert all(r['status']=='適用外' for r in exp['rows'] if r['method']!='MC')
    detail=detail_simulation(exp,.2)
    assert detail['fp_periodic'] is None and detail['transition_analysis']['available']
    with ZipFile(BytesIO(experiment_zip(exp,detail))) as z:
        assert 'censored_dwell_intervals.csv' in z.namelist()
        assert 'fp_probability.csv' not in z.namelist()


@pytest.fixture(scope='module')
def nonlinear_ou():
    model=PotentialModel(ModelSpec(kind='mono',k=1,b=0))
    c=SimConfig(f=.1,dt=.05,cycles=8)
    bands=make_bands(c.actual_dt,c.nperseg,c.f,SpectralConfig())
    r=periodic_snr(model,.2,.3,bands,'x',make_grid((-4,4),161),steps=256,phases=8,lag_cycles=4)
    return model,bands,r


def test_nonlinear_fp_matches_driven_ou_analytic_signal_and_spectrum(nonlinear_ou):
    model,bands,r=nonlinear_ou
    expected_signal=.3**2/(2*(1+(2*np.pi*bands.f0)**2))
    expected_noise=np.mean(4*.2/(1+(2*np.pi*bands.frequency[bands.noise])**2))*bands.bandwidth
    assert r['valid']
    assert r['signal_power']==pytest.approx(expected_signal,rel=.03)
    assert r['noise_power']==pytest.approx(expected_noise,rel=.03)
    assert r['snr']==pytest.approx(expected_signal/expected_noise,rel=.03)
    assert r['mass_error']<1e-10 and r['minimum_probability']>=0


def test_nonlinear_fp_zero_signal_is_not_a_fake_peak(nonlinear_ou):
    model,bands,_=nonlinear_ou
    r=periodic_snr(model,.2,0,bands,'x',make_grid((-4,4),121),steps=128,phases=4,lag_cycles=4)
    assert r['valid'] and r['snr']<1e-20


def test_unresolved_covariance_tail_suppresses_snr():
    m=PotentialModel(ModelSpec())
    c=SimConfig(f=.1,dt=.05,cycles=8)
    b=make_bands(c.actual_dt,c.nperseg,c.f,SpectralConfig())
    r=periodic_snr(m,.02,.06,b,'q',make_grid((-3,3),121),steps=64,phases=4,lag_cycles=4,max_cycles=3)
    assert not r['valid'] and np.isnan(r['snr'])
    assert not r['tail_converged'] or not r['periodic_converged']


def test_chaos_diagnostic_recovers_stable_linear_exponent():
    m=PotentialModel(ModelSpec(kind='mono',b=0))
    c=DuffingConfig(amplitude=0,frequency=.2,damping=.4,dt=.04,cycles=12,burn_cycles=6)
    r=duffing_diagnostic(m,c)
    assert r['finite_time_exponent']==pytest.approx(-.2,abs=.015)
    assert r['poincare'].shape==(12,2)


def test_chaos_refinement_and_invalid_inputs():
    m=PotentialModel(ModelSpec())
    c=DuffingConfig(frequency=.2,dt=.04,cycles=4,burn_cycles=2)
    r=duffing_refinement(m,c)
    assert r['fine']['actual_dt']<r['coarse']['actual_dt']
    with pytest.raises(ValueError): duffing_diagnostic(m,replace(c,damping=-1))
