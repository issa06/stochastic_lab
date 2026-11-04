"""Directed core-to-core transitions, censored dwell intervals and phase locking."""
from __future__ import annotations
import numpy as np


def transition_analysis(paths, time, model, frequency, phase_bins=12):
    paths = np.atleast_2d(np.asarray(paths, float))
    t = np.asarray(time, float)
    if len(t) < 2 or paths.shape[1] != len(t) or np.any(np.diff(t) <= 0):
        raise ValueError("遷移解析には昇順時刻と対応する軌道が必要です。")
    if not np.all(np.isfinite(paths)) or not np.all(np.isfinite(t)) or frequency <= 0 or phase_bins < 4:
        raise ValueError("遷移解析の値・周波数・位相bin数を確認してください。")
    minima = [e.x for e in model.extrema() if e.kind == "minimum"]
    tops = [e.x for e in model.extrema() if e.kind == "maximum"]
    n = len(minima)
    if n < 2:
        return {"available": False, "reason": "遷移解析は2井戸以上が対象です。", "events": [], "dwell": [], "edges": [], "occupancy": []}
    counts = np.zeros((n, n), int)
    exposure = np.zeros(n)
    events, dwell, unknown_time = [], [], 0.
    widths = np.diff(t)
    for trial, x in enumerate(paths):
        core = np.full(len(x), -1, int)
        for i, m in enumerate(minima):
            left = max((b for b in tops if b < m), default=-np.inf)
            right = min((b for b in tops if b > m), default=np.inf)
            core[(x >= (left+m)/2) & (x <= (right+m)/2)] = i
        ids = np.maximum.accumulate(np.where(core >= 0, np.arange(len(x)), -1))
        states = np.where(ids >= 0, core[np.maximum(ids,0)], -1)
        valid = np.flatnonzero(states >= 0)
        start = int(valid[0]) if len(valid) else 0
        changes = np.flatnonzero((states[1:] != states[:-1]) & (states[:-1] >= 0)) + 1
        for j,k in enumerate(changes):
            source, target = int(states[k-1]), int(states[k])
            counts[source,target] += 1
            events.append({"trial":trial,"time":float(t[k]),"source":source,"target":target,
                           "phase":float((t[k]*frequency)%1),"unresolved_skip":bool(abs(target-source)>1)})
            dwell.append({"trial":trial,"state":source,"duration":float(t[k]-t[start]),
                          "left_censored":j==0,"right_censored":False})
            start = int(k)
        for i in range(n):
            exposure[i] += widths[states[:-1] == i].sum()
        unknown_time += widths[states[:-1] < 0].sum()
        if len(valid):
            dwell.append({"trial": trial, "state": int(states[-1]), "duration": float(t[-1]-t[start]),
                          "left_censored":len(changes)==0, "right_censored": True})
    edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            phases = np.array([e["phase"] for e in events if e["source"] == i and e["target"] == j])
            vector = np.mean(np.exp(2j*np.pi*phases)) if len(phases) else complex(np.nan, np.nan)
            histogram, _ = np.histogram(phases, bins=np.linspace(0, 1, phase_bins+1))
            edges.append({"source": i, "target": j, "count": int(counts[i, j]),
                          "exposure": float(exposure[i]),
                          "empirical_rate": float(counts[i,j]/exposure[i]) if exposure[i]>0 else np.nan,
                          "phase_locking": float(abs(vector)),
                          "mean_phase": float((np.angle(vector)/(2*np.pi)) % 1) if len(phases) else np.nan,
                          "phase_counts": histogram.tolist()})
    total = float(exposure.sum())
    occupancy = [{"state": i, "x_min": minima[i], "exposure": float(exposure[i]),
                  "fraction": float(exposure[i]/total) if total else np.nan,
                  "exit_count": int(counts[i].sum())} for i in range(n)]
    return {"available": True, "events": events, "dwell": dwell, "edges": edges,
            "occupancy": occupancy, "unknown_time": float(unknown_time), "counts": counts,
            "phase_bins": phase_bins, "unresolved_skips": sum(e["unresolved_skip"] for e in events),
            "definition": "Core states retain their last known value; unknown initial time is excluded. Rates are counts/exposure, not stationary Kramers rates or unbiased MFPTs."}
