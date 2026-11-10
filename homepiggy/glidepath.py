"""Glidepath strategies: target sleeve weights as a function of time to purchase.

Method (docs/METHODS.md, section 4)
-----------------------------------
A glidepath maps months-to-purchase tau (H at the start, 0 at the purchase
date) to weights over SLEEVES = (equity, protected_equity, short_bonds, cash).

Static kinds (same weights on every path):
  step         piecewise-constant table, e.g. the agreed default:
               10-7y 80/20/0/0, 7-4y 30/60/10/0, 4-2y 0/20/30/50, 2-0y 0/0/20/80
  linear       linear interpolation between the midpoints of the table's
               segments, flat beyond the first and last midpoint
  time_warp    step table evaluated at warped time tau' = H * (tau/H)**gamma.
               gamma < 1: de-risk later (stay in risk longer); gamma > 1: de-risk earlier

Path-dependent kinds (weights differ per path; evaluated inside the portfolio loop):
  target_vol   lambda * w_base + (1 - lambda) * cash, with lambda in [0, 1]
               chosen so ex-ante volatility (regime-conditional sleeve covariance)
               equals a target that falls linearly from vol_start to vol_end.
               It only ever de-risks relative to the base table (no leverage).
  lock_in      follows the base table until pot >= multiple * required deposit,
               then switches permanently to `lock_weights`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .sleeves import SLEEVES

DEFAULT_TABLE: tuple[dict[str, Any], ...] = (
    {"from_years": 10, "to_years": 7, "weights": [0.80, 0.20, 0.00, 0.00]},
    {"from_years": 7, "to_years": 4, "weights": [0.30, 0.60, 0.10, 0.00]},
    {"from_years": 4, "to_years": 2, "weights": [0.00, 0.20, 0.30, 0.50]},
    {"from_years": 2, "to_years": 0, "weights": [0.00, 0.00, 0.20, 0.80]},
)


@dataclass(frozen=True)
class Segment:
    from_months: float
    to_months: float
    weights: np.ndarray


@dataclass(frozen=True)
class GlidepathSpec:
    kind: str = "step"
    table: Sequence[Mapping[str, Any]] = DEFAULT_TABLE
    gamma: float = 1.0                    # time_warp
    vol_start: float = 0.12               # target_vol (annualised)
    vol_end: float = 0.01
    lock_multiple: float = 1.0            # lock_in
    lock_weights: Sequence[float] | None = None
    name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.name or self.kind


KINDS = ("step", "linear", "time_warp", "target_vol", "lock_in")


def parse_table(table: Sequence[Mapping[str, Any]], horizon_months: int) -> list[Segment]:
    """Validate a glidepath table and convert it to months. Fail closed on gaps/overlaps."""
    segs = []
    for i, row in enumerate(table):
        w = np.asarray(row["weights"], dtype=float)
        if w.shape != (len(SLEEVES),):
            raise ValueError(f"Segment {i}: weights need {len(SLEEVES)} entries {SLEEVES}.")
        if np.any(w < -1e-12) or not np.isclose(w.sum(), 1.0, atol=1e-9):
            raise ValueError(f"Segment {i}: weights must be non-negative and sum to 1 (got {w.sum():.6f}).")
        f, to = float(row["from_years"]) * 12, float(row["to_years"]) * 12
        if not f > to >= 0:
            raise ValueError(f"Segment {i}: need from_years > to_years >= 0.")
        segs.append(Segment(f, to, np.clip(w, 0, None)))
    segs.sort(key=lambda s: -s.from_months)
    if not np.isclose(segs[-1].to_months, 0.0):
        raise ValueError("Glidepath table must end at 0 years to purchase.")
    for a, b in zip(segs, segs[1:]):
        if not np.isclose(a.to_months, b.from_months):
            raise ValueError("Glidepath table has a gap or overlap between segments.")
    if segs[0].from_months + 1e-9 < horizon_months:
        raise ValueError(
            f"Glidepath table starts at {segs[0].from_months/12:g}y but the horizon is {horizon_months/12:g}y."
        )
    return segs


def _step(segs: list[Segment], tau: np.ndarray) -> np.ndarray:
    out = np.empty(tau.shape + (len(SLEEVES),))
    for i, x in np.ndenumerate(tau):
        for s in segs:
            # segment covers (to, from]; tau = 0 belongs to the last segment
            if s.to_months < x <= s.from_months or (x <= 0 and s.to_months == 0):
                out[i] = s.weights
                break
        else:
            out[i] = segs[0].weights        # tau beyond the table start (validated against)
    return out


def _linear(segs: list[Segment], tau: np.ndarray) -> np.ndarray:
    mids = np.array([(s.from_months + s.to_months) / 2 for s in segs])[::-1]   # ascending
    W = np.stack([s.weights for s in segs])[::-1]
    return np.stack([np.interp(tau, mids, W[:, k]) for k in range(len(SLEEVES))], axis=-1)


def static_weights(spec: GlidepathSpec, horizon_months: int) -> np.ndarray:
    """(T+1, 4) target weights for month t = 0..T (tau = T - t). Base path for all kinds."""
    if spec.kind not in KINDS:
        raise ValueError(f"Unknown glidepath kind '{spec.kind}'; choose from {KINDS}.")
    segs = parse_table(spec.table, horizon_months)
    tau = horizon_months - np.arange(horizon_months + 1, dtype=float)
    if spec.kind == "linear":
        W = _linear(segs, tau)
    elif spec.kind == "time_warp":
        if not spec.gamma > 0:
            raise ValueError("gamma must be > 0.")
        W = _step(segs, horizon_months * (tau / horizon_months) ** spec.gamma)
    else:
        W = _step(segs, tau)
    W = W / W.sum(axis=1, keepdims=True)
    return W


def target_vol_lambda(
    w_base: np.ndarray, cov: np.ndarray, target: float, iters: int = 50
) -> np.ndarray:
    """lambda in [0, 1] such that vol(lambda * w_base + (1 - lambda) * e_cash) = target.

    w_base: (P, 4); cov: (P, 4, 4) annualised sleeve covariance for each path's
    current regime. vol(lambda) is increasing on [0, 1] whenever the base mix is
    riskier than cash, so bisection is valid. If even lambda = 1 is below the
    target, lambda = 1 (never lever up).
    """
    cash = np.zeros(len(SLEEVES))
    cash[SLEEVES.index("cash")] = 1.0

    def vol(lam: np.ndarray) -> np.ndarray:
        w = lam[:, None] * w_base + (1 - lam[:, None]) * cash
        return np.sqrt(np.maximum(np.einsum("pi,pij,pj->p", w, cov, w), 0.0))

    lo = np.zeros(w_base.shape[0])
    hi = np.ones(w_base.shape[0])
    full = vol(hi) <= target
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        above = vol(mid) > target
        hi = np.where(above, mid, hi)
        lo = np.where(above, lo, mid)
    return np.where(full, 1.0, 0.5 * (lo + hi))
