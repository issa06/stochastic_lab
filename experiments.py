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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset',choices=['all','symmetric','asymmetric','multi','mono'],default='all')
    parser.add_argument('--quick',action='store_true',help='Use fewer trials and cycles for smoke checks.')
    parser.add_argument('--validate',action='store_true',help='Run independent confirmation and convergence comparisons.')
    parser.add_argument('--out',type=Path,default=Path('results'))
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit('Output directory must be empty; existing results will not be overwritten.')
    args.out.mkdir(parents=True,exist_ok=True)
    cfg=SimConfig()
    spectral=SpectralConfig()
    ds=np.linspace(.04,.4,13)
    refine=5
    if args.quick:
        cfg=replace(cfg,trials=4,cycles=8,burn_cycles=1,dt=.05)
        spectral=replace(spectral,bootstrap=200)
        ds=np.linspace(.04,.4,5)
        refine=0
    for spec in preset_specs():
        if args.preset!='all' and args.preset!=spec.kind:
            continue
        exp=run_experiment(spec,cfg,spectral,ds,refine=refine)
        if args.validate:
            exp['independent_validation']=validate_experiment(exp)
        (args.out/f'{spec.kind}.json').write_text(json.dumps(json_safe(exp),ensure_ascii=False,indent=2,allow_nan=False))
        (args.out/f'{spec.kind}.zip').write_bytes(experiment_zip(exp))
        print(spec.kind,exp['evidence']['status'],exp['evidence'].get('D_star'))


if __name__=='__main__':
    main()
