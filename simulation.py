"""Euler--Maruyama with reproducible streams and optional local JIT acceleration."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from models import PotentialModel
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


@dataclass(frozen=True)
class SimConfig:
    A: float = 0.06
    f: float = 0.02
    dt: float = 0.02
    cycles: int = 48
    burn_cycles: int = 8
    trials: int = 30
    seed: int = 2026
    x0: float = -1.0
    segment_cycles: int = 8

    def validate(self):
        if not np.all(np.isfinite([self.A, self.f, self.dt, self.x0])):
            raise ValueError("入力値は有限値で指定してください。")
        if self.f <= 0 or self.dt <= 0 or self.A < 0:
            raise ValueError("f, dt > 0、A ≥ 0 が必要です。")
        if self.segment_cycles < 6 or self.cycles < self.segment_cycles or self.burn_cycles < 0:
            raise ValueError("観測周期数 ≥ Welch区間周期数 ≥ 6、過渡周期数 ≥ 0 が必要です。")
        if self.trials < 1 or self.seed < 0:
            raise ValueError("試行数 ≥ 1、乱数 seed ≥ 0 が必要です。")
        if self.steps_per_period < 40:
            raise ValueError("1周期あたり40ステップ以上となる dt にしてください。")
        if self.total_steps * self.trials > 24_000_000:
            raise ValueError("メモリ上限です。試行数・観測時間を減らすか dt を大きくしてください。")

    @property
    def steps_per_period(self):
        return int(math.ceil(1 / (self.f * self.dt)))

    @property
    def actual_dt(self):
        return 1 / (self.f * self.steps_per_period)

    @property
    def total_steps(self):
        return (self.cycles + self.burn_cycles) * self.steps_per_period

    @property
    def nperseg(self):
        return self.segment_cycles * self.steps_per_period


def seed_for(seed: int, D: float, stage: int = 0) -> np.random.SeedSequence:
    bits = np.asarray([float(D)], dtype="<f8").view("<u4")
    return np.random.SeedSequence([int(seed), int(stage), int(bits[0]), int(bits[1])])


def _integrate_numpy(coeff, force, noise, dt, scale, initial):
    n, trials = noise.shape
    result = np.empty((n + 1, trials))
    result[0] = initial
    for i in range(n):
        state = result[i]
        drift = np.zeros(trials)
        for value in coeff[::-1]:
            drift = drift * state + value
        next_state = state + (-drift + force[i]) * dt + scale * noise[i]
        if not np.all(np.isfinite(next_state)) or np.any(np.abs(next_state) > 1e6):
            raise ValueError("Euler–Maruyama が発散しました。dt を小さくしてください。")
        result[i + 1] = next_state
    return result


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _integrate_jit(coeff, force, noise, dt, scale, initial):
        n, trials = noise.shape
        result = np.empty((n + 1, trials))
        result[0, :] = initial
        for i in range(n):
            for j in range(trials):
                state = result[i, j]
                drift = 0.0
                for k in range(len(coeff) - 1, -1, -1):
                    drift = drift * state + coeff[k]
                value = state + (-drift + force[i]) * dt + scale * noise[i, j]
                if not np.isfinite(value) or abs(value) > 1e6:
                    raise ValueError("Euler-Maruyama divergence; decrease dt.")
                result[i + 1, j] = value
        return result


def simulate(model: PotentialModel, config: SimConfig, D: float, stage: int = 0,
             backend: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    config.validate()
    if not np.isfinite(D) or D < 0:
        raise ValueError("D は有限の非負値にしてください。")
    n = config.total_steps
    dt = config.actual_dt
    t = np.arange(n + 1) * dt
    force = config.A * np.cos(2 * np.pi * config.f * t[:-1])
    # Each trial has its own stream, unaffected by the requested trial count.
    children = seed_for(config.seed, D, stage).spawn(config.trials)
    noise = np.column_stack([np.random.default_rng(child).standard_normal(n) for child in children])
    integrator = _integrate_jit if NUMBA_AVAILABLE and backend != "numpy" else _integrate_numpy
    with np.errstate(over="ignore", invalid="ignore"):
        try:
            result = integrator(model.derivative.coef, force, noise, dt, np.sqrt(2 * D * dt), config.x0)
        except ValueError:
            raise ValueError("Euler–Maruyama が発散しました。dt を小さくしてください。") from None
    start = config.burn_cycles * config.steps_per_period
    # Omit the final endpoint to preserve an integer number of sampled cycles.
    return t[start:-1], np.ascontiguousarray(result[start:-1].T)


def core_transitions(x: np.ndarray, t: np.ndarray, model: PotentialModel) -> dict:
    minima = [e for e in model.extrema() if e.kind == "minimum"]
    tops = [e for e in model.extrema() if e.kind == "maximum"]
    if len(minima) < 2:
        return {"events": [], "mean_residence": np.nan, "completed": 0,
                "left_censored": 0, "right_censored": 0, "states": np.zeros(len(x), int)}
    bounds = []
    for m in minima:
        left = [b.x for b in tops if b.x < m.x]
        right = [b.x for b in tops if b.x > m.x]
        bounds.append(((left[-1] + m.x) / 2 if left else -np.inf,
                       (right[0] + m.x) / 2 if right else np.inf))
    core = np.full(len(x), -1, dtype=int)
    for i, (lower, upper) in enumerate(bounds):
        core[(x >= lower) & (x <= upper)] = i
    valid = np.flatnonzero(core >= 0)
    if not len(valid):
        return {"events": [], "mean_residence": np.nan, "completed": 0,
                "left_censored": 1, "right_censored": 1, "states": core}
    core[:valid[0]] = core[valid[0]]
    last = np.maximum.accumulate(np.where(core >= 0, np.arange(len(x)), 0))
    states = core[last]
    changes = np.flatnonzero(np.diff(states)) + 1
    events = []
    for j, index in enumerate(changes):
        duration = float(t[index] - t[changes[j - 1]]) if j else np.nan
        events.append({"time": float(t[index]), "source": int(states[index - 1]),
                       "target": int(states[index]), "duration": duration})
    durations = [e["duration"] for e in events if np.isfinite(e["duration"])]
    return {"events": events, "mean_residence": float(np.mean(durations)) if durations else np.nan,
            "completed": len(durations), "left_censored": 1, "right_censored": 1, "states": states}
