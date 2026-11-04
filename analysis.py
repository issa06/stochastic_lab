"""Experiment orchestration, adaptive grids, and independent confirmation tests."""
from __future__ import annotations
from dataclasses import asdict, replace
from importlib.metadata import version, PackageNotFoundError
import sys
import numpy as np
from models import ModelSpec, PotentialModel
from simulation import SimConfig, simulate, core_transitions
from spectra import SpectralConfig, make_bands, compute_psd, summarize_psd, coherent_amplitude, bootstrap_distribution, summarize_coherent
import fokker_planck as fp
import kramers
from transitions import transition_analysis
from accuracy import timestep_study, duration_study


METHODS = ("MC", "FP", "Kramers")


def effective_observable(model, observable):
    count = sum(e.kind == "minimum" for e in model.extrema())
    return "x" if count < 2 else observable


def monte_carlo_point(model, config, spectral, D, observable, stage=0, keep_psd=False):
    t, x = simulate(model, config, D, stage)
    y = model.observable(x, observable)
    bands = make_bands(config.actual_dt, config.nperseg, config.f, spectral)
    frequencies, psd = compute_psd(y, config.actual_dt, config.nperseg)
    band_config = replace(spectral, aggregation="pooled_psd") if spectral.aggregation == "coherent" else spectral
    result = summarize_psd(psd, bands, band_config, seed=config.seed + 19 * stage + 73)
    if spectral.aggregation == "coherent":
        result["welch_band_snr"] = result["snr"]
        result.update(summarize_coherent(t, y, config.f, result["noise_trials"]))
    else:
        result["ci_method"] = "trial_percentile_bootstrap"
    result.update(coherent_amplitude(t, y, config.f))
    trans = transition_analysis(x,t,model,config.f)
    result.update(transition_count=len(trans["events"]), unresolved_skips=trans.get("unresolved_skips",0),
                  transition_edges=trans["edges"], occupancy=trans["occupancy"])
    result.update(method="MC", D=float(D), approximation=config.scheme, noise_kind=config.noise_kind, observable=observable,
                  actual_dt=config.actual_dt, analysis_time=config.cycles/config.f,
                  stage=stage, eta=result["coherent_power_debiased"]*2/config.A**2 if config.A > 0 else np.nan)
    if keep_psd:
        result.update(frequency=frequencies, psd=np.mean(psd, axis=0), bands=bands, t=t, x=x[0], y=y[0], transition_analysis=trans)
    return result


def scalar_row(result):
    return {key: value for key, value in result.items() if np.isscalar(value) or value is None}


def scan_evidence(rows):
    mc_all = sorted((r for r in rows if r["method"] == "MC"), key=lambda r: r["D"])
    mc = [r for r in mc_all if np.isfinite(r.get("snr", np.nan))]
    if len(mc) < 5:
        return {"status": "推定不足", "D_star": np.nan, "interior": False,
                "missing_points": len(mc_all)-len(mc), "reason": "有限のSNRが5点未満"}
    best = int(np.argmax([r["snr"] for r in mc]))
    interior = 0 < best < len(mc)-1
    top, low, high = mc[best], mc[0], mc[-1]
    edge = max(low["snr"], high["snr"])
    ratio = top["snr"]/edge if edge > 0 else np.nan
    candidate = interior and top["snr"] > edge and top["snr"] > 0
    return {"status": "ピーク候補" if candidate else "境界最大" if not interior else "ピーク未支持",
            "D_star": top["D"], "snr_star": top["snr"], "interior": interior,
            "peak_to_edge": ratio, "low_D": low["D"], "high_D": high["D"],
            "missing_points": len(mc_all)-len(mc), "candidate": candidate,
            "reason": "探索結果。独立試行による確認前。"}


def run_experiment(spec: ModelSpec, config: SimConfig, spectral: SpectralConfig,
                   D_values, observable="q", nx=241, refine=5, methods=METHODS, progress=None):
    config.validate()
    D_values = np.unique(np.asarray(D_values, float))
    if len(D_values) < 5 or not np.all(np.isfinite(D_values)) or np.any(D_values <= 0):
        raise ValueError("D > 0 の異なる5点以上を指定してください。")
    if len(D_values) + refine > 100:
        raise ValueError("1実験のD候補は100点以下にしてください。")
    model = PotentialModel(spec)
    observable = effective_observable(model, observable)
    domain = model.domain(float(D_values.max()), config.A)
    grid = fp.make_grid(domain, nx)
    bands = make_bands(config.actual_dt, config.nperseg, config.f, spectral)
    rows, trials, transitions, occupancies = [], [], [], []
    total_work = len(D_values) + (refine if refine else 0)
    completed = 0

    def calculate(D):
        nonlocal completed
        for method in methods:
            try:
                if method == "MC":
                    output = monte_carlo_point(model, config, spectral, D, observable)
                    transitions.extend(dict(edge,D=D) for edge in output["transition_edges"])
                    occupancies.extend(dict(row,D=D) for row in output["occupancy"])
                    for i, (total, noise, snr) in enumerate(zip(output["total_trials"], output["noise_trials"], output["snr_trials"])):
                        trials.append({"D": D, "trial": i, "total_power": float(total),
                                       "noise_power": float(noise), "welch_trial_snr": float(snr), "stage": 0,
                                       "coherent_real": float(output["coherent_trials"][i].real) if "coherent_trials" in output else np.nan,
                                       "coherent_imag": float(output["coherent_trials"][i].imag) if "coherent_trials" in output else np.nan})
                elif method == "FP":
                    if config.noise_kind != "white":
                        raise ValueError("1次元FPは白色ガウスノイズのみ対応です。")
                    output = fp.response(model, D, config.A, bands, observable, grid)
                elif method == "Kramers":
                    if config.noise_kind != "white":
                        raise ValueError("このKramers式は白色ガウスノイズのみ対応です。")
                    output = kramers.response(model, D, config.A, bands, observable)
                else:
                    raise ValueError("未知の解析手法です。")
                row = scalar_row(output)
                row.update(observable=observable, status="OK" if np.isfinite(row["snr"]) else "推定不能")
                if method == "FP" and row.get("linear_forcing_scale",0) > .3:
                    row["status"] = "線形応答条件に注意"
                if method == "Kramers" and not row.get("kramers_reliable", False):
                    row["status"] = "近似条件に注意"
            except (ValueError, FloatingPointError, ArithmeticError) as error:
                row = {"method": method, "D": D, "snr": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                       "observable": observable, "status": "適用外" if method == "Kramers" or (method == "FP" and config.noise_kind != "white") else "計算失敗",
                       "error_type": type(error).__name__}
            rows.append(row)
        completed += 1
        if progress:
            progress(completed, max(completed, total_work))

    for D in D_values:
        calculate(float(D))
    if refine and "MC" in methods:
        evidence = scan_evidence(rows)
        if evidence.get("candidate"):
            i = int(np.argmin(abs(D_values - evidence["D_star"])))
            left, right = D_values[max(i-1,0)], D_values[min(i+1,len(D_values)-1)]
            new_values = [float(d) for d in np.linspace(left, right, refine+2)[1:-1]
                          if not np.any(np.isclose(D_values, d, rtol=1e-10, atol=1e-14))]
            for D in new_values:
                calculate(D)
    rows.sort(key=lambda r: (r["D"], r["method"]))
    evidence = scan_evidence(rows)
    if config.A == 0:
        evidence.update(status="周期入力なし（A=0）",candidate=False)
    software = {"python": sys.version.split()[0]}
    for package in ("numpy", "scipy", "streamlit", "plotly", "numba"):
        try:
            software[package] = version(package)
        except PackageNotFoundError:
            software[package] = "not_installed"
    return {"model": asdict(spec), "simulation": asdict(config), "spectral": asdict(spectral),
            "observable": observable, "domain": domain, "nx": nx, "rows": rows, "trials": trials,
            "transition_edges": transitions, "occupancies": occupancies,
            "evidence": evidence, "software": software,
            "methods": list(methods), "independent_validation": None,
            "scan_settings": {"coarse_D": D_values.tolist(), "refine": int(refine)},
            "comparison_note": "共通帯域。MCはWelch推定、FP/Kramersは線形応答のコヒーレント成分と平衡雑音。"}


def _contrast(a, b, spectral, seed, alpha=.05/3):
    if spectral.aggregation == "coherent":
        from scipy.stats import t as student_t
        va, vb = a["standard_error"]**2, b["standard_error"]**2
        variance = va+vb
        denom = va**2/(a["n_trials"]-1) + vb**2/(b["n_trials"]-1)
        df = variance**2/denom if denom > 0 else np.inf
        difference = a["signed_snr"]-b["signed_snr"]
        lower = float(difference-student_t.ppf(1-alpha,df)*np.sqrt(variance))
        return {"difference":float(difference),"lower_bound":lower,
                "one_sided_confidence":1-alpha,"inference":"jackknife_Welch_t",
                "result":"PASS" if np.isfinite(lower) and lower>0 else "FAIL" if np.isfinite(lower) else "NOT RUN"}
    da = bootstrap_distribution(a["total_trials"], a["noise_trials"], spectral, seed)
    db = bootstrap_distribution(b["total_trials"], b["noise_trials"], spectral, seed+100003)
    diff = da-db
    lower = float(np.quantile(diff, alpha)) if np.all(np.isfinite(diff)) else np.nan
    return {"difference": float(a["snr"]-b["snr"]), "lower_bound": lower,
            "one_sided_confidence": 1-alpha,
            "result": "PASS" if np.isfinite(lower) and lower > 0 else "FAIL" if np.isfinite(lower) else "NOT RUN"}


def validate_experiment(experiment, progress=None):
    evidence = experiment["evidence"]
    if not evidence.get("candidate", False):
        return {"status": "ピーク候補なし", "checks": [], "rows": [], "trials": [],
                "conclusion": "探索範囲・観測時間を見直してください。"}
    model = PotentialModel(ModelSpec(**experiment["model"]))
    config = SimConfig(**experiment["simulation"])
    spectral = SpectralConfig(**experiment["spectral"])
    observable = experiment["observable"]
    values = [evidence["low_D"], evidence["D_star"], evidence["high_D"]]
    results, rows, trials, checks = [], [], [], []

    def add_trial_data(result, condition):
        rows.append(dict(scalar_row(result), condition=condition))
        for j, (total, noise) in enumerate(zip(result["total_trials"], result["noise_trials"])):
            trials.append({"condition": condition, "D": result["D"], "trial": j,
                           "total_power": float(total), "noise_power": float(noise), "stage": result["stage"],
                           "coherent_real": float(result["coherent_trials"][j].real) if "coherent_trials" in result else np.nan,
                           "coherent_imag": float(result["coherent_trials"][j].imag) if "coherent_trials" in result else np.nan})

    for i, D in enumerate(values):
        result = monte_carlo_point(model, config, spectral, D, observable, stage=101)
        results.append(result)
        add_trial_data(result, ["low", "peak", "high"][i])
        if progress:
            progress(i+1, 8)
    low, peak, high = results
    null = monte_carlo_point(model, replace(config, A=0), spectral, values[1], observable, stage=102)
    add_trial_data(null, "A_zero")
    for name, other, seed in [("低D端より高い", low, 1001), ("高D端より高い", high, 1002), ("A=0対照より高い", null, 1003)]:
        checks.append({"check": name, **_contrast(peak, other, spectral, seed)})
    if progress:
        progress(4,8)
    convergence = []
    try:
        convergence.extend(timestep_study(model,config,spectral,values[1],observable))
    except (ValueError, ArithmeticError):
        convergence.append({"check":"結合経路 dt/dt2/dt4", "result":"NOT RUN"})
    try:
        convergence.append(duration_study(model,config,spectral,values[1],observable))
    except (ValueError,ArithmeticError):
        convergence.append({"check":"観測時間 T → 2T（同じ軌道）","result":"NOT RUN"})
    if progress:
        progress(6,8)
    bands = make_bands(config.actual_dt, config.nperseg, config.f, spectral)
    domain, nx = tuple(experiment["domain"]), experiment["nx"]
    try:
        if config.noise_kind != "white":
            raise NotImplementedError
        base = fp.response(model, values[1], config.A, bands, observable, fp.make_grid(domain, nx))
        mid, radius = (domain[0]+domain[1])/2, (domain[1]-domain[0])/2
        cases = [("FP 格子細分化", domain, min(2001,2*nx-1)),
                 ("FP 領域×1.25", (mid-1.25*radius,mid+1.25*radius), min(2001,int(np.ceil(1.25*nx))))]
        for name, dom, points in cases:
            r = fp.response(model, values[1], config.A, bands, observable, fp.make_grid(dom,points))
            difference = abs(r["snr"]-base["snr"])/max(abs(base["snr"]),1e-12)
            convergence.append({"check": name, "snr": r["snr"], "relative_change": difference,
                                "threshold": .05, "result": "PASS" if difference <= .05 else "FAIL"})
    except NotImplementedError:
        convergence.append({"check":"FP収束比較（白色ノイズのみ）", "result":"NOT APPLICABLE"})
    except (ValueError, ArithmeticError):
        convergence.append({"check": "FP収束比較", "result": "NOT RUN"})
    no_noise_config = replace(config, trials=1)
    t, x = simulate(model, no_noise_config, 0, stage=301)
    transitions = core_transitions(x[0], t, model)
    control = {"D_zero_transitions": len(transitions["events"]), "A_zero_snr": null["snr"],
               "subthreshold_trajectory": len(transitions["events"]) == 0}
    supported = all(c["result"]=="PASS" for c in checks)
    numerical_pass = all(c["result"]=="PASS" for c in convergence if c["result"]!="NOT APPLICABLE")
    if progress:
        progress(8,8)
    status = "独立試行でピークを支持" if supported else "独立試行では未支持"
    if supported and not numerical_pass:
        status += "（数値精度は未確認）"
    return {"status": status,
            "checks": checks, "rows": rows, "trials": trials, "convergence": convergence,
            "controls": control, "numerical_checks_pass": numerical_pass,
            "conclusion": "指定した観測量・帯域・探索範囲に限定した統計的支持。一般的証明ではありません。",
            "selection_stage": 0, "validation_stage": 101, "alpha_family": .05,
            "multiplicity": "3つの片側比較にBonferroni補正。点ごとのCIとは別。"}


def detail_simulation(experiment, D, dt_fp=.05, max_cycles=80):
    spec = ModelSpec(**experiment["model"])
    config = SimConfig(**experiment["simulation"])
    spectral = SpectralConfig(**experiment["spectral"])
    model = PotentialModel(spec)
    observable = experiment["observable"]
    detail = monte_carlo_point(model, replace(config, trials=min(config.trials,8)), spectral,
                              D, observable, stage=401, keep_psd=True)
    detail["transitions"] = core_transitions(detail["x"], detail["t"], model)
    detail["rates"] = kramers.rate_table(model,D,config.f) if config.noise_kind == "white" else []
    if config.noise_kind != "white":
        detail.update(fp_periodic=None,kramers_periodic=None)
        return detail
    grid = fp.make_grid(tuple(experiment["domain"]),experiment["nx"])
    detail["fp_periodic"] = fp.periodic_density(model,D,config.A,config.f,observable,grid,
                                              dt_max=dt_fp,max_cycles=max_cycles)
    try:
        detail["kramers_periodic"] = kramers.periodic_populations(model,D,config.A,config.f,observable,
                                                               dt_max=dt_fp,max_cycles=max_cycles)
    except (ValueError,ArithmeticError):
        detail["kramers_periodic"] = None
    return detail

