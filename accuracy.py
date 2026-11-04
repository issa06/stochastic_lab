"""Quantitative precision checks using coupled paths and paired jackknife errors."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from scipy.stats import t as student_t
from simulation import coupled_simulations, simulate
from spectra import compute_psd, make_bands, powers_from_psd, statistic, summarize_coherent


def equivalence_result(difference, low, high, reference, relative=.10, absolute=.002):
    tolerance = max(absolute, relative*abs(reference))
    finite = bool(np.all(np.isfinite([difference, low, high, reference])))
    status = "NOT RUN" if not finite else "PASS" if low >= -tolerance and high <= tolerance else "FAIL" if abs(difference)>tolerance else "INCONCLUSIVE"
    return {"difference": float(difference), "difference_ci_low": float(low), "difference_ci_high": float(high),
            "tolerance": tolerance, "relative_change": abs(difference)/max(abs(reference),1e-12),
            "result": status, "criterion": f"95% difference interval entirely within ±max({absolute:g}, {100*relative:g}% of reference)"}


def _estimate(time, paths, config, spectral, model, observable):
    y = model.observable(paths, observable)
    _, psd = compute_psd(y, config.actual_dt, config.nperseg)
    total, noise = powers_from_psd(psd, make_bands(config.actual_dt, config.nperseg, config.f, spectral))
    n = len(noise)
    if n < 3 or noise.mean() <= 0:
        return np.nan, np.full(n, np.nan)
    if spectral.aggregation == "coherent":
        z = 2*np.mean(y*np.exp(-2j*np.pi*config.f*time), axis=-1)
        summed, square = z.sum(), np.sum(abs(z)**2)
        value = (abs(summed)**2-square)/(2*n*(n-1)*noise.mean())
        denom = (noise.sum()-noise)/(n-1)
        leave = (abs(summed-z)**2-(square-abs(z)**2))/(2*(n-1)*(n-2))
        leave = np.divide(leave, denom, out=np.full(n,np.nan), where=denom>0)
    else:
        value = float(statistic(total, noise, spectral.aggregation))
        leave = np.array([float(statistic(np.delete(total,i),np.delete(noise,i),spectral.aggregation)) for i in range(n)])
    return float(value), leave


def timestep_study(model, config, spectral, D, observable, relative=.10, absolute=.002):
    outputs = coupled_simulations(model, config, D)
    estimates = [_estimate(t,x,c,spectral,model,observable) for c,t,x in outputs]
    checks = []
    for i in range(2):
        coarse, fine = estimates[i], estimates[i+1]
        delta = fine[0]-coarse[0]
        leave = fine[1]-coarse[1]
        n = len(leave)
        se = np.sqrt((n-1)/n*np.sum((leave-leave.mean())**2))
        margin = student_t.ppf(.975,n-1)*se
        _,_,a = outputs[i]
        _,_,b = outputs[i+1]
        checks.append({"check": ["dt → dt/2（結合経路）", "dt/2 → dt/4（結合経路）"][i],
                       "coarse_dt": outputs[i][0].actual_dt, "fine_dt": outputs[i+1][0].actual_dt,
                       "coarse_snr": coarse[0], "snr": fine[0], "path_rmse": float(np.sqrt(np.mean((a-b[:,::2])**2))),
                       "inference": "paired_jackknife_t", **equivalence_result(delta,delta-margin,delta+margin,fine[0],relative,absolute)})
    return checks


def independent_precision(a, b):
    if not all(np.isfinite(r.get("standard_error",np.nan)) for r in (a,b)):
        return {"result":"NOT RUN", "criterion":"Precision comparison requires coherent SNR jackknife errors."}
    variance = a["standard_error"]**2+b["standard_error"]**2
    denom = a["standard_error"]**4/(a["n_trials"]-1)+b["standard_error"]**4/(b["n_trials"]-1)
    df = variance**2/denom if denom>0 else np.inf
    delta = b["signed_snr"]-a["signed_snr"]
    margin = student_t.ppf(.975,df)*np.sqrt(variance)
    return equivalence_result(delta,delta-margin,delta+margin,b["signed_snr"])


def duration_study(model,config,spectral,D,observable):
    """Compare T and 2T using the first half of exactly the same trajectory."""
    longer=replace(config,cycles=2*config.cycles)
    time,paths=simulate(model,longer,D,stage=701)
    cutoff=config.cycles*config.steps_per_period
    short=_estimate(time[:cutoff],paths[:,:cutoff],config,spectral,model,observable)
    full=_estimate(time,paths,longer,spectral,model,observable)
    delta=full[0]-short[0]
    leave=full[1]-short[1]
    n=len(leave)
    se=np.sqrt((n-1)/n*np.sum((leave-leave.mean())**2))
    margin=student_t.ppf(.975,n-1)*se
    return {"check":"観測時間 T → 2T（同じ軌道）","snr":full[0],"short_snr":short[0],
            "inference":"paired_jackknife_t","stage":701,
            **equivalence_result(delta,delta-margin,delta+margin,full[0])}
