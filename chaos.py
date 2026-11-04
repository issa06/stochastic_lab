"""Deterministic inertial Duffing dynamics and finite-time variational exponents."""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import numpy as np


@dataclass(frozen=True)
class DuffingConfig:
    amplitude: float = .30
    frequency: float = 1.2/(2*np.pi)
    damping: float = .20
    dt: float = .02
    cycles: int = 100
    burn_cycles: int = 40
    x0: float = .1
    v0: float = 0.


def duffing_diagnostic(model, config=DuffingConfig()):
    """Integrate an added velocity state; no noisy trajectory is called chaos.

    A tangent in (x,v) is advanced by RK4 and renormalized every step. The
    reported exponent is conditional on the forcing phase and finite duration.
    """
    c = config
    values = list(asdict(c).values())
    if not np.all(np.isfinite(values)) or c.frequency <= 0 or c.dt <= 0 or c.damping <= 0 or c.amplitude < 0:
        raise ValueError("Duffingモデルの有限値・周波数・減衰・刻みを確認してください。")
    if not isinstance(c.cycles,int) or not isinstance(c.burn_cycles,int) or c.cycles < 4 or c.burn_cycles < 0:
        raise ValueError("観測は4周期以上、過渡除外は0周期以上の整数です。")
    steps = int(np.ceil(1/(c.frequency*c.dt)))
    total = steps*(c.cycles+c.burn_cycles)
    if steps < 40 or total > 500_000:
        raise ValueError("Duffing計算は40ステップ/周期以上・合計50万ステップ以下です。")
    dt = 1/(c.frequency*steps)
    state = np.array([c.x0,c.v0,1.,0.])
    coeff = model.derivative.coef
    curvature = model.second_derivative.coef

    def derivative(t,y):
        x,v,a,b = y
        force = np.polynomial.polynomial.polyval(x,coeff)
        second = np.polynomial.polynomial.polyval(x,curvature)
        return np.array([v,-c.damping*v-force+c.amplitude*np.cos(2*np.pi*c.frequency*t),
                         b,-second*a-c.damping*b])

    start = c.burn_cycles*steps
    traces, section, growth, running, blocks = [], [], 0., [], []
    block_growth = 0.
    stride = max(1,int(np.ceil(c.cycles*steps/5000)))
    for k in range(total):
        t = k*dt
        k1 = derivative(t,state)
        k2 = derivative(t+dt/2,state+dt*k1/2)
        k3 = derivative(t+dt/2,state+dt*k2/2)
        k4 = derivative(t+dt,state+dt*k3)
        state += dt*(k1+2*k2+2*k3+k4)/6
        norm = np.linalg.norm(state[2:])
        if not np.all(np.isfinite(state)) or norm <= 0 or max(abs(state[:2])) > 1e6:
            raise ValueError("Duffing積分が発散しました。刻み幅を小さくしてください。")
        state[2:] /= norm
        if k >= start:
            growth += np.log(norm)
            block_growth += np.log(norm)
            if (k-start) % stride == 0:
                traces.append(((k+1)*dt,float(state[0]),float(state[1])))
            if (k+1) % steps == 0:
                section.append((float(state[0]),float(state[1])))
                running.append(growth/((k+1-start)*dt))
                blocks.append(block_growth/(steps*dt))
                block_growth = 0.
    exponent = float(running[-1])
    classification = "0付近の有限時間指数（判定保留）" if abs(exponent)<=.01 else "正の有限時間指数（カオス候補）" if exponent>0 else "正の有限時間指数を確認せず"
    return {"model":asdict(model.spec),"config":asdict(c),"actual_dt":dt,"trace":np.asarray(traces),"poincare":np.asarray(section),
            "running_exponent":np.asarray(running),"period_exponents":np.asarray(blocks),
            "finite_time_exponent":exponent,"classification":classification,
            "scope":"Deterministic inertial extension; finite-time evidence is not an asymptotic chaos proof."}


def duffing_refinement(model,config):
    coarse = duffing_diagnostic(model,config)
    fine = duffing_diagnostic(model,replace(config,dt=coarse["actual_dt"]/2))
    difference = abs(fine["finite_time_exponent"]-coarse["finite_time_exponent"])
    return {"coarse":coarse,"fine":fine,"exponent_difference":difference,
            "threshold":max(.01,.1*abs(fine["finite_time_exponent"])),
            "precision_supported":bool(difference<=max(.01,.1*abs(fine["finite_time_exponent"]))) }
