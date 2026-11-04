"""Adjacent-well Kramers network, not a forced two-state model."""
from __future__ import annotations
import numpy as np
from scipy.special import softmax
from models import PotentialModel
from markov import BirthDeath, linear_response, periodic_evolution
from spectra import Bands


def generator(model: PotentialModel, D: float, force: float = 0) -> BirthDeath:
    if D <= 0 or not np.isfinite(D):
        raise ValueError("Kramers法は D > 0 が必要です。")
    minima = [e for e in model.extrema(force) if e.kind == "minimum"]
    edges = model.barriers(force)
    if len(minima) < 2 or len(edges) != 2 * (len(minima) - 1):
        raise ValueError("非退化の隣接井戸・障壁がそろわないためKramers法は適用外です。")
    up, down = np.zeros(len(minima)-1), np.zeros(len(minima)-1)
    for edge in edges:
        rate = edge.prefactor * np.exp(-edge.delta_u / D)
        if edge.target > edge.source:
            up[edge.source] = rate
        else:
            down[edge.target] = rate
    p = softmax(np.array([-e.u / D - .5*np.log(e.curvature) for e in minima]))
    return BirthDeath(np.array([e.x for e in minima]), up, down, p)


def diagnostics(model: PotentialModel, D: float, A: float, f: float) -> dict:
    minima = [e for e in model.extrema() if e.kind == "minimum"]
    edges = model.barriers()
    topology = all(len([e for e in model.extrema(force) if e.kind == "minimum"]) == len(minima)
                   and len(model.barriers(force)) == len(edges) for force in np.linspace(-A, A, 17))
    critical_x = [float(z.real) for z in model.second_derivative.roots()
                  if abs(z.imag) <= 1e-7 * (1 + abs(z.real))]
    critical_force = [float(model.dU(x)) for x in critical_x]
    if any(abs(force) <= abs(A) + 1e-10 for force in critical_force):
        topology = False
    if not edges:
        return {"applicable": False, "topology_preserved": False, "barrier_over_D": np.nan,
                "tilt_over_D": np.nan, "adiabatic_ratio": np.nan, "kramers_reliable": False}
    barrier_ratio = min(e.delta_u / D for e in edges)
    tilt_ratio = max(A * abs(e.xb-e.xm) / D for e in edges)
    adiabatic = 2*np.pi*f / min(e.curvature for e in minima)
    return {"applicable": bool(topology), "topology_preserved": bool(topology),
            "barrier_over_D": barrier_ratio, "tilt_over_D": tilt_ratio, "adiabatic_ratio": adiabatic,
            # These are explicit diagnostic thresholds, not a validity theorem.
            "kramers_reliable": bool(topology and barrier_ratio >= 5 and adiabatic <= .1 and tilt_ratio <= .3)}


def response(model: PotentialModel, D: float, A: float, bands: Bands, observable: str) -> dict:
    diag = diagnostics(model, D, A, bands.f0)
    if not diag["applicable"]:
        raise ValueError("周期外力で井戸数が変わるか障壁がないため、固定状態Kramersモデルは適用外です。")
    epsilon = 1e-5 * max(1.0, D)
    base = generator(model, D)
    plus, minus = generator(model, D, epsilon), generator(model, D, -epsilon)
    if len(plus.grid) != len(base.grid) or len(minus.grid) != len(base.grid):
        raise ValueError("分岐点に近いためKramers線形応答を計算できません。")
    g = base.grid if observable == "x" else np.linspace(-1, 1, len(base.grid))
    row = linear_response(base, plus, minus, epsilon, g, A, bands)
    row.update(diag)
    row.update(method="Kramers", D=D, approximation="linear_response_well_network")
    return row


def rate_table(model: PotentialModel, D: float, f: float):
    if D <= 0:
        return []
    edges = model.barriers()
    outgoing = {}
    for e in edges:
        outgoing[e.source] = outgoing.get(e.source, 0.0) + e.prefactor*np.exp(-e.delta_u/D)
    rows = []
    for edge in edges:
        rate = float(edge.prefactor*np.exp(-edge.delta_u/D))
        match = edge.delta_u / np.log(edge.prefactor / (2*f)) if edge.prefactor > 2*f else np.nan
        rows.append({"source": edge.source, "target": edge.target, "x_min": edge.xm, "x_barrier": edge.xb,
                     "delta_u": edge.delta_u, "curvature_min": edge.curvature_min,
                     "curvature_barrier": edge.curvature_barrier, "prefactor": edge.prefactor,
                     "rate": rate, "inverse_edge_rate": 1/rate if rate > 0 else np.inf,
                     "mean_exit_time": 1/outgoing[edge.source] if outgoing[edge.source] > 0 else np.inf,
                     "half_period": 1/(2*f), "edge_time_match_D": match,
                     "barrier_over_D": edge.delta_u/D})
    return rows


def periodic_populations(model, D, A, f, observable, dt_max=.05, max_cycles=100):
    if not diagnostics(model, D, A, f)["applicable"]:
        raise ValueError("周期中の井戸構造が変わるため、Kramers人口動態は適用外です。")
    base = generator(model, D)
    g = base.grid if observable == "x" else np.linspace(-1, 1, len(base.grid))
    result = periodic_evolution(lambda force: generator(model, D, force), g, A, f,
                                dt_max=dt_max, max_cycles=max_cycles)
    result.pop("boundary_mass", None)
    return result
