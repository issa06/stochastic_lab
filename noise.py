"""Integrated stationary noise with a common long-time diffusion intensity D."""
from __future__ import annotations
import numpy as np
from scipy.signal import lfilter

NOISE_LABELS = {"white": "白色ガウス", "ou": "色付きガウス（OU）", "telegraph": "二値ノイズ"}


def integrated_noise(kind, D, dt, steps, trials, seed_sequence, tau=0.5):
    """Return integrals over each step; colored covariance is D/tau exp(-|t|/tau).

    Both colored models satisfy integral_0^infinity C(t) dt = D. OU integrals
    and endpoints are sampled jointly exactly. Telegraph switches are integrated
    at continuous exponential event times, rather than rounded to the time grid.
    """
    if kind not in NOISE_LABELS or not np.isfinite(D) or D < 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("ノイズ種類・D・時間刻みを確認してください。")
    if kind != "white" and (not np.isfinite(tau) or tau <= 0):
        raise ValueError("相関時間 τ > 0 が必要です。")
    if steps < 1 or trials < 1 or steps * trials > 24_000_000:
        raise ValueError("ノイズ配列のサイズを確認してください。")
    out = np.zeros((steps, trials))
    if D == 0:
        return out
    for j, child in enumerate(seed_sequence.spawn(trials)):
        rng = np.random.default_rng(child)
        if kind == "white":
            out[:, j] = np.sqrt(2 * D * dt) * rng.standard_normal(steps)
        elif kind == "ou":
            u = dt / tau
            q = -np.expm1(-u)
            rho = np.exp(-u)
            var_end = D / tau * (-np.expm1(-2 * u))
            v = (2*u**3/3-u**4/2+7*u**5/30-u**6/12) if u < .001 else 2*u-4*q-np.expm1(-2*u)
            var_integral = D * tau * v
            cov = D * q**2
            z1, z2 = rng.standard_normal((2, steps))
            initial = np.sqrt(D / tau) * rng.standard_normal()
            before = np.empty(steps)
            before[0] = initial
            if steps > 1:
                before[1:] = lfilter([np.sqrt(var_end)], [1, -rho], z1[:-1], zi=[rho*initial])[0]
            weight = cov / np.sqrt(var_end)
            out[:, j] = tau*q*before + weight*z1 + np.sqrt(max(0., var_integral-weight**2))*z2
        else:
            horizon = steps * dt
            if horizon / (2*tau) > 2_000_000:
                raise ValueError("二値ノイズの切替回数が多すぎます。相関時間を増やしてください。")
            events, clock = [], 0.
            while True:
                clock += rng.exponential(2*tau)
                if clock >= horizon:
                    break
                events.append(clock)
            edges = np.r_[0., events, horizon]
            signs = rng.choice([-1., 1.]) * (-1.)**np.arange(len(edges)-1)
            cumulative = np.r_[0., np.cumsum(np.diff(edges)*signs)]
            times = np.arange(steps+1)*dt
            ids = np.minimum(np.searchsorted(edges, times, side="right")-1, len(signs)-1)
            integral = cumulative[ids] + signs[ids]*(times-edges[ids])
            out[:, j] = np.sqrt(D/tau)*np.diff(integral)
    return out
