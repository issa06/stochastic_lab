"""Reproduce the shipped experiments without launching Streamlit."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from analysis import run_experiment,validate_experiment
from models import preset_specs
from simulation import SimConfig
from spectra import SpectralConfig
from exporting import experiment_zip,json_safe
from models import PotentialModel,ModelSpec
from accuracy import timestep_study
from spectra import make_bands
from fokker_planck import make_grid
from nonlinear_fp import resolution_study
from chaos import DuffingConfig,duffing_refinement


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset',choices=['all','symmetric','asymmetric','multi','mono'],default='all')
    parser.add_argument('--quick',action='store_true',help='Use fewer trials and cycles for smoke checks.')
    parser.add_argument('--validate',action='store_true',help='Run independent confirmation and convergence comparisons.')
    parser.add_argument('--noise',choices=['white','ou','telegraph','all'],default='white')
    parser.add_argument('--tau',type=float,default=.5)
    parser.add_argument('--scheme',choices=['heun','euler'],default='heun')
    parser.add_argument('--trials',type=int)
    parser.add_argument('--cycles',type=int)
    parser.add_argument('--dt',type=float)
    parser.add_argument('--precision',action='store_true',help='Check nested timesteps even when there is no interior peak.')
    parser.add_argument('--nonlinear',action='store_true',help='Compute finite-amplitude FP and resolution checks at the selected D (white noise only).')
    parser.add_argument('--nonlinear-steps',type=int,default=256)
    parser.add_argument('--duffing',action='store_true',help='Run only the deterministic inertial chaos diagnostic.')
    parser.add_argument('--out',type=Path,default=Path('results'))
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit('Output directory must be empty; existing results will not be overwritten.')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.duffing:
        result=duffing_refinement(PotentialModel(ModelSpec()),DuffingConfig())
        (args.out/'duffing.json').write_text(json.dumps(json_safe(result),ensure_ascii=False,allow_nan=False),encoding='utf-8')
        print('duffing',result['fine']['classification'])
        return
    cfg=SimConfig(scheme=args.scheme,noise_tau=args.tau)
    spectral=SpectralConfig()
    ds=np.linspace(.04,.4,13)
    refine=5
    if args.quick:
        cfg=replace(cfg,trials=4,cycles=8,burn_cycles=1,dt=.05)
        spectral=replace(spectral,bootstrap=200)
        ds=np.linspace(.04,.4,5)
        refine=0
    overrides={k:getattr(args,k) for k in ('trials','cycles','dt') if getattr(args,k) is not None}
    cfg=replace(cfg,**overrides)
    cfg.validate()
    for spec in preset_specs():
        if args.preset!='all' and args.preset!=spec.kind:
            continue
        for noise in (['white','ou','telegraph'] if args.noise=='all' else [args.noise]):
            current=replace(cfg,noise_kind=noise)
            exp=run_experiment(spec,current,spectral,ds,refine=refine)
            if args.validate:
                exp['independent_validation']=validate_experiment(exp)
            selected=exp['evidence'].get('D_star',float(ds[len(ds)//2]))
            if not np.isfinite(selected): selected=float(ds[len(ds)//2])
            if args.precision:
                try:
                    previous=(exp.get('independent_validation') or {}).get('convergence',[])
                    coupled=[r for r in previous if '結合経路' in r.get('check','')]
                    exp['precision_study']={'D':selected,'checks':coupled if len(coupled)==2 else timestep_study(PotentialModel(spec),current,spectral,selected,exp['observable'])}
                except (ValueError,ArithmeticError):
                    exp['precision_study']={'D':selected,'checks':[{'result':'NOT RUN','reason':'Resource or numerical limit'}]}
            if args.nonlinear and noise=='white':
                bands=make_bands(current.actual_dt,current.nperseg,current.f,spectral)
                exp['nonlinear_fp']={'selected_D':selected,'result':resolution_study(PotentialModel(spec),selected,current.A,bands,exp['observable'],make_grid(exp['domain'],exp['nx']),steps=args.nonlinear_steps)}
            name=f'{spec.kind}_{noise}'
            (args.out/f'{name}.json').write_text(json.dumps(json_safe(exp),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
            (args.out/f'{name}.zip').write_bytes(experiment_zip(exp))
            print(name,exp['evidence']['status'],exp['evidence'].get('D_star'),flush=True)


if __name__=='__main__':
    main()

