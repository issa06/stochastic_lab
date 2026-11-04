"""Collect reproducible numerical evidence into compact reviewable text datasets."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def write_csv(path,rows):
    if not rows:
        return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit('Summary output directory must be empty.')
    args.out.mkdir(parents=True,exist_ok=True)
    summary,sweeps,trials,edges,occupancy,covariance,periodic_means=[],[],[],[],[],[],[]
    for folder in args.inputs:
        for path in sorted(folder.glob('*.json')):
            data=json.loads(path.read_text(encoding='utf-8'))
            if 'simulation' not in data:
                continue
            name=f"{data['model']['kind']}_{data['simulation']['noise_kind']}"
            validation=data.get('independent_validation') or {}
            nonlinear=(data.get('nonlinear_fp') or {}).get('result') or {}
            summary.append({'experiment':name,'model':data['model'],'simulation':data['simulation'],
                            'spectral':data['spectral'],'software':data['software'],'domain':data['domain'],'nx':data['nx'],
                            'scan_settings':data['scan_settings'],'observable':data['observable'],
                            'evidence':data['evidence'],'validation':validation,
                            'precision_study':data.get('precision_study'),
                            'nonlinear_fp':{key:nonlinear[key] for key in nonlinear if key not in
                                            ('lag','covariance','time','mean','probability','grid','noise_frequency','noise_psd')}})
            for source,target in [('rows',sweeps),('trials',trials),('transition_edges',edges),('occupancies',occupancy)]:
                target.extend(dict(row,experiment=name) for row in data.get(source,[]))
            if nonlinear:
                covariance.extend({'experiment':name,'lag':t,'connected_covariance':c}
                                  for t,c in zip(nonlinear['lag'],nonlinear['covariance']))
                periodic_means.extend({'experiment':name,'time':t,'periodic_mean':m}
                                      for t,m in zip(nonlinear['time'],nonlinear['mean']))
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    for name,rows in [('noise_sweep',sweeps),('trial_measurements',trials),('directed_transitions',edges),
                      ('occupancies',occupancy),('nonlinear_covariance',covariance),('periodic_means',periodic_means)]:
        write_csv(args.out/f'{name}.csv',rows)
    print(f'Collected {len(summary)} experiments.')


if __name__=='__main__':
    main()
