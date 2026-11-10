"""Outcome metrics with explicit Monte Carlo uncertainty.

Method (docs/METHODS.md, section 7)
-----------------------------------
* Probabilities are reported with a Wilson score 95% interval (Wilson 1927).
  It stays accurate near 0 and 1, where the normal-approximation interval
  fails.
* Quantiles are reported with a distribution-free 95% interval from order
  statistics: the number of draws below the true q-quantile is
  Binomial(n, q), so the interval [X_(l), X_(u)] uses binomial quantiles
  for l and u (e.g. Hahn & Meeker 1991).
* Means are reported with standard error s / sqrt(n).
* Money-weighted return (IRR) per path: the monthly rate i with
  NPV(i) = 0 for the cash flows (-initial, -contributions, +final value).
  These flows change sign exactly once, so by Descartes' rule of signs the
  root is unique, and bisection on [-0.2, 0.2] per month converges.
* Drawdown is measured on the time-weighted NAV index (returns net of
  flows), so contributions cannot hide losses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom, norm


@dataclass(frozen=True)
class Estimate:
    value: float
    lo: float
    hi: float
    method: str

    def as_dict(self) -> dict[str, float | str]:
        return {"value": self.value, "lo": self.lo, "hi": self.hi, "method": self.method}


def wilson(successes: int, n: int, conf: float = 0.95) -> Estimate:
    if n <= 0:
        raise ValueError("n must be positive.")
    z = norm.ppf(0.5 + conf / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Estimate(p, max(0.0, centre - half), min(1.0, centre + half), "wilson")


def prob(mask: np.ndarray, conf: float = 0.95) -> Estimate:
    mask = np.asarray(mask, bool)
    return wilson(int(mask.sum()), mask.size, conf)


def quantile(x: np.ndarray, q: float, conf: float = 0.95) -> Estimate:
    x = np.sort(np.asarray(x, float).ravel())
    n = x.size
    a = (1 - conf) / 2
    lo_i = int(binom.ppf(a, n, q))           # number of draws below the lower bound
    hi_i = int(binom.ppf(1 - a, n, q))
    lo_i = min(max(lo_i - 1, 0), n - 1)
    hi_i = min(max(hi_i, 0), n - 1)
    return Estimate(float(np.quantile(x, q)), float(x[lo_i]), float(x[hi_i]), "order-statistic")


def mean(x: np.ndarray, conf: float = 0.95) -> Estimate:
    x = np.asarray(x, float).ravel()
    se = x.std(ddof=1) / np.sqrt(x.size) if x.size > 1 else np.nan
    z = norm.ppf(0.5 + conf / 2)
    m = float(x.mean())
    return Estimate(m, m - z * se, m + z * se, "normal (CLT), s/sqrt(n)")


def irr_monthly(flows: np.ndarray, lo: float = -0.2, hi: float = 0.2, iters: int = 80) -> np.ndarray:
    """Vectorised IRR. flows: (P, T+1), investor's view (negative = paid in).

    Returns the monthly rate; NaN where no sign change exists (e.g. zero final value).
    """
    flows = np.asarray(flows, float)
    T = flows.shape[1] - 1
    t = np.arange(T + 1)

    def npv(i: np.ndarray) -> np.ndarray:
        return (flows / (1.0 + i[:, None]) ** t).sum(axis=1)

    a = np.full(flows.shape[0], lo)
    b = np.full(flows.shape[0], hi)
    fa = npv(a)
    fb = npv(b)
    ok = np.sign(fa) != np.sign(fb)
    for _ in range(iters):
        mid = 0.5 * (a + b)
        fm = npv(mid)
        same = np.sign(fm) == np.sign(fa)
        a = np.where(same, mid, a)
        fa = np.where(same, fm, fa)
        b = np.where(same, b, mid)
    out = 0.5 * (a + b)
    return np.where(ok, out, np.nan)


def max_drawdown(nav: np.ndarray) -> np.ndarray:
    """Per-path maximum drawdown (<= 0) of a NAV index (P, n)."""
    peak = np.maximum.accumulate(nav, axis=1)
    return (nav / peak - 1.0).min(axis=1)
