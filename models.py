"""Confining polynomial potentials and adjacent-well geometry."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
from numpy.polynomial import Polynomial, Chebyshev


@dataclass(frozen=True)
class ModelSpec:
    kind: str = "symmetric"
    a: float = 1.0
    b: float = 1.0
    c: float = 0.04
    k: float = 1.0
    a6: float = 1.0
    a4: float = -2.0
    a2: float = 1.0
    a1: float = 0.0
    height: float = 0.25
    wells: int = 3
    skew: float = 0.0


@dataclass(frozen=True)
class Extremum:
    x: float
    u: float
    curvature: float
    kind: str


@dataclass(frozen=True)
class Barrier:
    source: int
    target: int
    xm: float
    xb: float
    delta_u: float
    curvature_min: float
    curvature_barrier: float

    @property
    def prefactor(self) -> float:
        return float(np.sqrt(self.curvature_min * abs(self.curvature_barrier)) / (2 * np.pi))


LABELS = {"symmetric": "対称双安定", "asymmetric": "非対称双安定",
          "multi": "多井戸（6次多項式）", "mono": "単安定", "barriers": "障壁数・高さ指定"}


class PotentialModel:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        if spec.kind not in LABELS:
            raise ValueError("ポテンシャルの種類が不正です。")
        for key, value in asdict(spec).items():
            if key != "kind" and not np.isfinite(value):
                raise ValueError("係数は有限値で指定してください。")
        if spec.kind in ("symmetric", "asymmetric"):
            if spec.a <= 0 or spec.b <= 0:
                raise ValueError("双安定モデルは a > 0, b > 0 が必要です。")
            p = Polynomial([0, spec.c if spec.kind == "asymmetric" else 0, -spec.a / 2, 0, spec.b / 4])
        elif spec.kind == "multi":
            p = Polynomial([0, spec.a1, spec.a2, 0, spec.a4, 0, spec.a6])
        elif spec.kind == "mono":
            if spec.k < 0 or spec.b < 0 or spec.k + spec.b <= 0:
                raise ValueError("単安定では k, b ≥ 0 かつ少なくとも一方を正にしてください。")
            p = Polynomial([0, 0, spec.k / 2, 0, spec.b / 4])
        else:
            if spec.height <= 0 or not 1 <= spec.wells <= 5 or int(spec.wells) != spec.wells:
                raise ValueError("高さ h > 0、井戸数 1〜5 を指定してください。")
            # Equal-height Chebyshev barriers at zero skew; mean well spacing is two.
            n = int(spec.wells)
            scale = 1.0 if n == 1 else (n - 1) / np.cos(np.pi / (2 * n))
            z = Polynomial([0, 1 / scale])
            cheb = Chebyshev.basis(n).convert(kind=Polynomial)(z)
            p = spec.height * cheb**2 * (1 + spec.skew * z + 0.5 * (spec.skew * z)**2) + Polynomial([0, spec.c])
        self.poly = p.trim(tol=0)
        if self.poly.degree() < 2 or self.poly.degree() % 2 or self.poly.coef[-1] <= 0:
            raise ValueError("両端で U(x) → +∞ となる正の偶数次最高係数が必要です。")
        self.coefficients = np.asarray(self.poly.coef, dtype=float)
        self.derivative = self.poly.deriv()
        self.second_derivative = self.poly.deriv(2)
        self.name = LABELS[spec.kind]

    def potential(self, x):
        return self.poly(x)

    def dU(self, x):
        return self.derivative(x)

    def ddU(self, x):
        return self.second_derivative(x)

    def force(self, x):
        return -self.dU(x)

    def extrema(self, force: float = 0.0) -> list[Extremum]:
        derivative = self.derivative - Polynomial([force])
        candidates = sorted(float(r.real) for r in derivative.roots()
                            if abs(r.imag) <= 1e-7 * (1 + abs(r.real)))
        unique: list[float] = []
        for x in candidates:
            if not unique or abs(x - unique[-1]) > 1e-5 * (1 + abs(x)):
                unique.append(x)
        out = []
        for x in unique:
            residual_scale = max(1.0, float(np.sum(np.abs(derivative.coef) * abs(x)**np.arange(len(derivative.coef)))))
            if abs(derivative(x)) > 1e-7 * residual_scale:
                continue
            curvature = float(self.ddU(x))
            kind = "minimum" if curvature > 1e-7 else "maximum" if curvature < -1e-7 else "flat"
            # Flat stationary points are deliberately not assigned a Kramers rate.
            out.append(Extremum(x, float(self.potential(x) - force * x), curvature, kind))
        return out

    def barriers(self, force: float = 0.0) -> list[Barrier]:
        extrema = self.extrema(force)
        minima = [p for p in extrema if p.kind == "minimum"]
        barriers = []
        for i in range(len(minima) - 1):
            between = [p for p in extrema if p.kind == "maximum" and minima[i].x < p.x < minima[i + 1].x]
            if len(between) != 1:
                continue
            top = between[0]
            for source, target in ((i, i + 1), (i + 1, i)):
                bottom = minima[source]
                if top.u > bottom.u:
                    barriers.append(Barrier(source, target, bottom.x, top.x, top.u - bottom.u,
                                            bottom.curvature, top.curvature))
        return barriers

    def observable(self, x, kind: str = "x"):
        values = np.asarray(x, dtype=float)
        if kind == "x":
            return values
        minima = [p for p in self.extrema() if p.kind == "minimum"]
        if len(minima) < 2:
            raise ValueError("井戸状態 q(x) は2井戸以上で使用できます。")
        tops = [p.x for p in self.extrema() if p.kind == "maximum" and minima[0].x < p.x < minima[-1].x]
        if len(tops) != len(minima) - 1:
            raise ValueError("井戸境界が一意でないため q(x) を定義できません。")
        # A memoryless mapping is shared by Monte Carlo and Fokker--Planck.
        levels = np.linspace(-1.0, 1.0, len(minima))
        return levels[np.searchsorted(tops, values)]

    def domain(self, D_max: float, A: float = 0.0) -> tuple[float, float]:
        roots = [e.x for e in self.extrema()]
        radius = max(1.0, max(map(abs, roots), default=1.0) + 0.3)
        floor = min((e.u for e in self.extrema()), default=0.0)
        target = 14 * max(D_max, 0.01)
        for _ in range(100):
            edge = min(float(self.potential(-radius)), float(self.potential(radius))) - abs(A) * radius
            if edge - floor > target:
                return -radius, radius
            radius *= 1.12
        raise ValueError("FP計算領域の自動設定に失敗しました。係数を確認してください。")

    def latex(self, derivative: bool = False) -> str:
        coefficients = -self.derivative.coef if derivative else self.poly.coef
        terms = []
        for power in reversed(range(len(coefficients))):
            coefficient = float(coefficients[power])
            if abs(coefficient) < 1e-14:
                continue
            variable = "" if power == 0 else "x" if power == 1 else f"x^{{{power}}}"
            magnitude = "" if abs(coefficient) == 1 and power else f"{abs(coefficient):.6g}"
            sign = "-" if coefficient < 0 else "+" if terms else ""
            terms.append(sign + magnitude + variable)
        return "".join(terms) or "0"


def preset_specs() -> list[ModelSpec]:
    return [ModelSpec(), ModelSpec(kind="asymmetric", c=0.04),
            ModelSpec(kind="multi"), ModelSpec(kind="mono", k=1, b=0)]
