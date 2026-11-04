"""Export only numerical experiment fields, never environment configuration."""
from __future__ import annotations
import io
import json
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd


def csv_bytes(rows):
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")


def json_safe(value):
    if isinstance(value,dict):
        return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value,np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value,np.generic):
        return json_safe(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def parameter_rows(experiment):
    rows=[]
    for section in ("model","simulation","spectral","software"):
        for key,value in experiment[section].items():
            rows.append({"section":section,"parameter":key,"value":str(value)})
    for key in ("observable","domain","nx","scan_settings"):
        rows.append({"section":"experiment","parameter":key,"value":str(experiment[key])})
    return rows


def experiment_zip(experiment, detail=None):
    out=io.BytesIO()
    with ZipFile(out,"w",ZIP_DEFLATED) as archive:
        archive.writestr("parameters.csv",csv_bytes(parameter_rows(experiment)))
        archive.writestr("noise_sweep.csv",csv_bytes(experiment["rows"]))
        archive.writestr("trial_results.csv",csv_bytes(experiment["trials"]))
        archive.writestr("experiment.json",json.dumps(json_safe(experiment),ensure_ascii=False,indent=2,allow_nan=False))
        validation=experiment.get("independent_validation")
        if validation:
            archive.writestr("validation_checks.csv",csv_bytes(validation.get("checks",[])))
            archive.writestr("validation_trials.csv",csv_bytes(validation.get("trials",[])))
            archive.writestr("convergence.csv",csv_bytes(validation.get("convergence",[])))
        if detail:
            archive.writestr("time_series.csv",csv_bytes({"t":detail["t"],"x":detail["x"],"observable":detail["y"]}))
            bands=detail["bands"]
            region=np.full(len(detail["frequency"]),"unused",dtype=object)
            region[bands.signal]="signal"
            region[bands.noise]="noise"
            archive.writestr("psd.csv",csv_bytes({"frequency":detail["frequency"],"psd":detail["psd"],"region":region}))
            archive.writestr("kramers_rates.csv",csv_bytes(detail["rates"]))
            archive.writestr("transitions.csv",csv_bytes(detail["transitions"]["events"]))
            density=detail["fp_periodic"]
            rows={"time":np.repeat(density["time"],len(density["grid"])),
                  "x":np.tile(density["grid"],len(density["time"])),
                  "cell_probability":density["probability"].ravel()}
            archive.writestr("fp_probability.csv",csv_bytes(rows))
    return out.getvalue()
