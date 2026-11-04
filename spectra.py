"""Explicit Welch-band SNR estimators and trial-resampling uncertainty."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import welch, get_window


@dataclass(frozen=True)
class SpectralConfig:
    half_width: int = 1
    guard: int = 1
    outer: int = 5
    bootstrap: int = 1200
    aggregation: str = "coherent"


@dataclass
class Bands:
    frequency: np.ndarray
    signal: np.ndarray
    noise: np.ndarray
    df: float
    bandwidth: float
    f0: float


def make_bands(dt: float, nperseg: int, f0: float, config: SpectralConfig) -> Bands:
    freq = np.fft.rfftfreq(nperseg, dt)
    k = int(np.argmin(abs(freq - f0)))
    h = int(config.half_width)
    if h < 1 or config.guard < 0 or config.outer <= h + config.guard:
        raise ValueError("信号帯域は片側1 bin以上、雑音帯域はガードの外側に指定してください。")
    if k - config.outer < 1 or k + config.outer >= len(freq) - 1:
        raise ValueError("駆動周波数の両側に雑音帯域を取れません。Welch区間を長くしてください。")
    if abs(freq[k] - f0) > 1e-8 * max(f0, 1):
        raise ValueError("駆動周波数をFFT binに合わせてください。周期数指定を使用します。")
    signal = np.arange(k - h, k + h + 1)
    offsets = np.arange(h + config.guard + 1, config.outer + 1)
    noise = np.r_[k - offsets, k + offsets]
    # Exclude harmonics and their signal-width neighborhoods from the background.
    harmonic = np.rint(freq[noise] / f0).astype(int)
    noise = noise[(harmonic < 2) | (abs(freq[noise] - harmonic * f0) > (h + 0.5) * (freq[1] - freq[0]))]
    if len(noise) < 4 or not np.any(noise < k) or not np.any(noise > k):
        raise ValueError("高調波を除いた左右の雑音 bin が不足しています。")
    df = float(freq[1] - freq[0])
    return Bands(freq, signal, np.sort(noise), df, len(signal) * df, f0)


def compute_psd(x: np.ndarray, dt: float, nperseg: int):
    data = np.asarray(x, dtype=float)
    if data.shape[-1] < nperseg or not np.all(np.isfinite(data)):
        raise ValueError("Welch区間より長い有限値の時系列が必要です。")
    return welch(data, fs=1 / dt, window=get_window("hann", nperseg, fftbins=True),
                 nperseg=nperseg, noverlap=nperseg // 2, nfft=nperseg,
                 detrend="constant", return_onesided=True, scaling="density", axis=-1,
                 average="mean")


def powers_from_psd(psd: np.ndarray, bands: Bands):
    total = np.sum(psd[..., bands.signal], axis=-1) * bands.df
    noise = np.mean(psd[..., bands.noise], axis=-1) * bands.bandwidth
    return np.atleast_1d(total), np.atleast_1d(noise)


def ratios(total, noise):
    total, noise = np.broadcast_arrays(np.asarray(total, float), np.asarray(noise, float))
    ratio = np.full_like(total, np.nan)
    np.divide(np.maximum(total - noise, 0), noise, out=ratio, where=noise > 0)
    return ratio


def statistic(total, noise, aggregation: str, axis=-1):
    if aggregation == "pooled_psd":
        return ratios(np.mean(total, axis=axis), np.mean(noise, axis=axis))
    if aggregation == "trial_mean":
        # Undefined trial ratios remain undefined; no survivor-only averaging.
        return np.mean(ratios(total, noise), axis=axis)
    raise ValueError("未知のSNR集約法です。")


def bootstrap_distribution(total, noise, config: SpectralConfig, seed=1234):
    total, noise = np.asarray(total), np.asarray(noise)
    if len(total) < 2:
        return np.full(config.bootstrap, np.nan)
    index = np.random.default_rng(seed).integers(0, len(total), (config.bootstrap, len(total)))
    return statistic(total[index], noise[index], config.aggregation)


def summarize_psd(psd: np.ndarray, bands: Bands, config: SpectralConfig, seed=1234) -> dict:
    total, noise = powers_from_psd(np.atleast_2d(psd), bands)
    value = float(statistic(total, noise, config.aggregation))
    trial = ratios(total, noise)
    distribution = bootstrap_distribution(total, noise, config, seed)
    finite_boot = np.all(np.isfinite(distribution))
    low, high = np.quantile(distribution, [.025, .975]) if finite_boot else (np.nan, np.nan)
    all_valid = np.all(np.isfinite(trial))
    return {"snr": value, "ci_low": float(low), "ci_high": float(high),
            "snr_db": float(10 * np.log10(value)) if value > 0 else -np.inf if value == 0 else np.nan,
            "trial_snr_mean": float(np.mean(trial)) if all_valid else np.nan,
            "trial_snr_sd": float(np.std(trial, ddof=1)) if all_valid and len(trial) > 1 else np.nan,
            "bootstrap_sd": float(np.std(distribution, ddof=1)) if finite_boot else np.nan,
            "n_trials": len(trial), "undefined_trials": int(np.sum(~np.isfinite(trial))),
            "signal_power": float(max(np.mean(total - noise), 0)),
            "noise_power": float(np.mean(noise)), "bandwidth": bands.bandwidth,
            "total_trials": total, "noise_trials": noise, "snr_trials": trial}


def coherent_amplitude(t: np.ndarray, y: np.ndarray, f0: float) -> dict:
    z = 2 * np.mean(y * np.exp(-2j * np.pi * f0 * t), axis=-1)
    mean = complex(np.mean(z))
    correction = float(np.var(z, ddof=1) / len(z)) if len(z) > 1 else np.nan
    corrected_power = max(abs(mean)**2 - correction, 0) / 2 if np.isfinite(correction) else np.nan
    return {"amplitude": abs(mean), "phase": float(-np.angle(mean)),
            "coherent_power_debiased": corrected_power}


def summarize_coherent(t, y, f0, noise_trials):
    """Cross-trial coherent power with jackknife-t uncertainty for the ratio.

    The untruncated U-statistic removes the finite-ensemble power bias. Confidence
    limits use its signed value; only the displayed physical power is truncated.
    """
    from scipy.stats import t as student_t
    z = 2*np.mean(y*np.exp(-2j*np.pi*f0*t),axis=-1)
    n=len(z)
    background=float(np.mean(noise_trials))
    if n < 3 or background <= 0:
        return {"snr":np.nan,"signed_snr":np.nan,"ci_low":np.nan,"ci_high":np.nan,
                "standard_error":np.nan,"signal_power":np.nan,"coherent_trials":z,
                "ci_method":"jackknife_t"}
    total=z.sum()
    squared=np.sum(abs(z)**2)
    signal=float((abs(total)**2-squared)/(2*n*(n-1)))
    theta=signal/background
    leave_signal=(abs(total-z)**2-(squared-abs(z)**2))/(2*(n-1)*(n-2))
    leave_noise=(np.sum(noise_trials)-noise_trials)/(n-1)
    if np.any(leave_noise<=0):
        se=np.nan
    else:
        leave_ratio=leave_signal/leave_noise
        se=float(np.sqrt((n-1)/n*np.sum((leave_ratio-leave_ratio.mean())**2)))
    delta=float(student_t.ppf(.975,n-1)*se)
    value=max(theta,0)
    return {"snr":value,"signed_snr":theta,"standard_error":se,
            "ci_low":max(theta-delta,0) if np.isfinite(delta) else np.nan,
            "ci_high":max(theta+delta,0) if np.isfinite(delta) else np.nan,
            "signal_power":max(signal,0),"coherent_trials":z,"ci_method":"jackknife_t",
            "snr_db":float(10*np.log10(value)) if value>0 else -np.inf}
