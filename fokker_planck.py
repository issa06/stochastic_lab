"""Scharfetter--Gummel finite volumes with reflecting boundaries."""
from __future__ import annotations
import numpy as np
from scipy.special import softmax
from markov import BirthDeath, linear_response, periodic_evolution
from models import PotentialModel
from spectra import Bands


def bernoulli(z):
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z)
    small = abs(z) < 1e-5
    positive = z > 50
    negative = z < -50
    regular = ~(small | positive | negative)
    result[small] = 1 - z[small] / 2 + z[small]**2 / 12 - z[small]**4 / 720
    result[positive] = z[positive] * np.exp(-z[positive])
    result[negative] = -z[negative]
    result[regular] = z[regular] / np.expm1(z[regular])
    return result


def generator(model: PotentialModel, D: float, grid: np.ndarray, force: float = 0) -> BirthDeath:
    if D <= 0 or not np.isfinite(D):
        raise ValueError("FP法は D > 0 が必要です。D=0 は決定論的軌道で検証します。")
    grid = np.asarray(grid, dtype=float)
    if len(grid) < 5 or not np.all(np.diff(grid) > 0) or not np.allclose(np.diff(grid), grid[1]-grid[0]):
        raise ValueError("FP格子は5点以上の等間隔昇順にしてください。")
    dx = grid[1] - grid[0]
    energy = model.potential(grid) - force * grid
    jumps = np.diff(energy) / D
    up = D / dx**2 * bernoulli(jumps)
    down = D / dx**2 * bernoulli(-jumps)
    p = softmax(-(energy - np.min(energy)) / D)
    return BirthDeath(grid, up, down, p)


def make_grid(domain: tuple[float, float], nx: int = 241):
    if domain[0] >= domain[1] or nx < 41 or nx > 2001:
        raise ValueError("FP領域と格子点数（41〜2001）を確認してください。")
    # Cell centers, with zero flux through the two domain edges.
    dx = (domain[1] - domain[0]) / nx
    return domain[0] + (np.arange(nx) + .5) * dx


def response(model: PotentialModel, D: float, A: float, bands: Bands,
             observable: str, grid: np.ndarray) -> dict:
    epsilon = 1e-5 * max(1.0, D)
    base = generator(model, D, grid)
    row = linear_response(base, generator(model, D, grid, epsilon), generator(model, D, grid, -epsilon),
                          epsilon, model.observable(grid, observable), A, bands)
    variance_x = float(np.sum(base.probability * (grid - grid @ base.probability)**2))
    row["linear_forcing_scale"] = A * np.sqrt(variance_x) / D
    row.update(method="FP", D=D, boundary_mass=float(base.probability[:3].sum() + base.probability[-3:].sum()),
               dx=float(grid[1] - grid[0]), approximation="linear_response")
    return row


def periodic_density(model: PotentialModel, D: float, A: float, f: float, observable: str,
                     grid: np.ndarray, dt_max=.05, max_cycles=100):
    return periodic_evolution(lambda force: generator(model, D, grid, force),
                              model.observable(grid, observable), A, f, dt_max=dt_max,
                              max_cycles=max_cycles)
