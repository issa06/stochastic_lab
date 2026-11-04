"""Column-generator utilities, linear response, and periodic probability evolution."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_banded
from spectra import Bands


@dataclass
class BirthDeath:
    grid: np.ndarray
    up: np.ndarray
    down: np.ndarray
    probability: np.ndarray

    @property
    def diagonal(self):
        d = np.zeros(len(self.grid))
        d[:-1] -= self.up
        d[1:] -= self.down
        return d

    def apply(self, v):
        out = self.diagonal * v
        out[1:] += self.up * v[:-1]
        out[:-1] += self.down * v[1:]
        return out

    def resolvent(self, omega, source):
        n = len(self.grid)
        ab = np.zeros((3, n), dtype=complex)
        ab[0, 1:] = -self.down
        ab[1] = 1j * omega - self.diagonal
        ab[2, :-1] = -self.up
        return solve_banded((1, 1), ab, source.astype(complex), check_finite=False)

    def implicit_matrix(self, dt):
        ab = np.zeros((3, len(self.grid)))
        ab[0, 1:] = -dt * self.down
        ab[1] = 1 - dt * self.diagonal
        ab[2, :-1] = -dt * self.up
        return ab


def linear_response(base: BirthDeath, plus: BirthDeath, minus: BirthDeath, epsilon: float,
                    observable: np.ndarray, A: float, bands: Bands) -> dict:
    p = base.probability
    g = observable - observable @ p
    source = (plus.apply(p) - minus.apply(p)) / (2 * epsilon)
    omega = 2 * np.pi * bands.f0
    p1 = base.resolvent(omega, source)
    susceptibility = complex(observable @ p1)
    needed = np.unique(np.r_[bands.signal, bands.noise])
    densities = np.array([max(0.0, float(4 * np.real(g @ base.resolvent(
        2 * np.pi * bands.frequency[k], p * g)))) for k in needed])
    background = dict(zip(needed, densities))
    floor = float(np.mean([background[k] for k in bands.noise]))
    power = float(0.5 * A**2 * abs(susceptibility)**2)
    noise_power = floor * bands.bandwidth
    snr = power / noise_power if noise_power > 0 else np.nan
    background_in_signal = float(np.mean([background[k] for k in bands.signal]))
    return {"snr": float(snr), "ci_low": np.nan, "ci_high": np.nan,
            "snr_db": float(10 * np.log10(snr)) if snr > 0 else -np.inf if snr == 0 else np.nan,
            "signal_power": power, "noise_power": noise_power, "bandwidth": bands.bandwidth,
            "amplitude": A * abs(susceptibility), "phase": float(-np.angle(susceptibility)),
            "susceptibility_real": susceptibility.real, "susceptibility_imag": susceptibility.imag,
            "eta": float(abs(susceptibility)**2),
            "floor_curvature_ratio": background_in_signal / floor if floor > 0 else np.nan,
            "stationarity_residual": float(np.linalg.norm(base.apply(p), 1)),
            "response_mass_residual": float(abs(np.sum(p1))),
            "n_trials": 0, "undefined_trials": 0}


def periodic_evolution(builder, observable, A: float, f: float, dt_max: float = .05,
                       tolerance: float = 1e-7, max_cycles: int = 100,
                       output_frames: int = 128) -> dict:
    period = 1 / f
    steps = max(64, int(np.ceil(period / dt_max)))
    if steps > 16000:
        raise ValueError("FP時間ステップ数が多すぎます。dt_FP または f を調整してください。")
    dt = period / steps
    base = builder(0.0)
    if steps * len(base.grid) > 4_000_000:
        raise ValueError("確率密度の作業配列が大きすぎます。FP時間刻みまたは格子数を調整してください。")
    # Right-endpoint forcing matches the implicit Euler time discretization.
    matrices = [builder(A * np.cos(2 * np.pi * (j + 1) / steps)).implicit_matrix(dt)
                for j in range(steps)]
    p = base.probability.copy()
    sample_index = set(np.unique(np.linspace(0, steps - 1, min(output_frames, steps), dtype=int)))
    mass_error = 0.0
    minimum_probability = 1.0
    residual = np.inf
    for cycle in range(1, max_cycles + 1):
        previous = p.copy()
        frames, means, phases = [], [], []
        for j, matrix in enumerate(matrices):
            if j in sample_index:
                frames.append(p.copy())
                means.append(float(observable @ p))
                phases.append(j * dt)
            p = solve_banded((1, 1), matrix, p, check_finite=False)
            mass_error = max(mass_error, abs(float(p.sum()) - 1))
            minimum_probability = min(minimum_probability, float(p.min()))
            if minimum_probability < -1e-10 or mass_error > 1e-7:
                raise ArithmeticError("確率の保存・非負性に問題が発生しました。計算条件を確認してください。")
        residual = float(np.linalg.norm(p - previous, 1))
        if cycle >= 3 and residual < tolerance:
            break
    frames = np.asarray(frames)
    phases = np.asarray(phases)
    means = np.asarray(means)
    # Fit the sampled periodic mean; frames need not be exactly equally spaced.
    design = np.column_stack([np.ones(len(phases)), np.cos(2*np.pi*f*phases), np.sin(2*np.pi*f*phases)])
    fit = np.linalg.lstsq(design, means, rcond=None)[0]
    return {"time": phases, "probability": frames, "mean": means, "grid": base.grid,
            "amplitude": float(np.hypot(fit[1], fit[2])), "phase": float(np.arctan2(fit[2], fit[1])),
            "mass_error": mass_error, "minimum_probability": minimum_probability,
            "periodic_residual": residual, "converged": bool(residual < tolerance),
            "cycles": cycle, "dt": dt,
            "boundary_mass": float(np.max(frames[:, :3].sum(axis=1) + frames[:, -3:].sum(axis=1)))}
