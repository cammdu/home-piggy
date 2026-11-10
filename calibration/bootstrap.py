"""Stationary block bootstrap for parameter uncertainty (option C).

Politis & Romano (1994) stationary bootstrap: blocks start at uniformly random
months and have geometric lengths with mean L, wrapping circularly. Whole
months are resampled jointly (all variables and the regime label together),
which preserves cross-sectional dependence and, within blocks, serial
dependence and regime persistence.

Mean block length L is chosen by the automatic rule of Politis & White (2004),
with the correction of Patton, Politis & White (2009). It is computed for each
key series and the maximum is used, the most conservative choice.

Pairs that straddle a block join are flagged invalid (``valid_prev = False``),
so the re-estimation never counts an artificial transition or AR(1) step.
"""
from __future__ import annotations

import numpy as np


def politis_white_block_length(x: np.ndarray) -> float:
    """Optimal mean block length for the stationary bootstrap (PW 2004; PPW 2009 correction)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    x = x - x.mean()
    kn = max(5, int(np.ceil(np.sqrt(np.log10(n)))))
    mmax = int(np.ceil(np.sqrt(n))) + kn
    c = 2.0 * np.sqrt(np.log10(n) / n)
    acov = np.array([x[: n - k] @ x[k:] / n for k in range(mmax + 1)])
    rho = acov / acov[0]
    # smallest m such that |rho(m+1..m+kn)| < c
    mhat = None
    for m in range(1, mmax - kn + 1):
        if np.all(np.abs(rho[m + 1: m + 1 + kn]) < c):
            mhat = m
            break
    if mhat is None:
        mhat = mmax - kn
    M = min(2 * mhat, mmax)
    k = np.arange(-M, M + 1)
    lam = np.where(np.abs(k / M) <= 0.5, 1.0, 2.0 * (1.0 - np.abs(k / M)))   # flat-top lag window
    lam = np.where(np.abs(k / M) > 1.0, 0.0, lam)
    R = acov[np.abs(k)]
    G = np.sum(lam * np.abs(k) * R)
    g0 = np.sum(lam * R)
    D = 2.0 * g0 ** 2
    b = (2.0 * G ** 2 / D) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(min(max(b, 1.0), np.ceil(min(3 * np.sqrt(n), n / 3))))


def stationary_indices(n: int, mean_block: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Resampled row indices (n,) and valid_prev mask (True where row j-1 precedes row j in time)."""
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    valid = np.zeros(n, dtype=bool)
    idx[0] = rng.integers(n)
    for j in range(1, n):
        if rng.random() < p:
            idx[j] = rng.integers(n)          # new block
        else:
            idx[j] = (idx[j - 1] + 1) % n     # continue block (circular)
            valid[j] = idx[j] != 0            # wrap-around is not a true succession
    return idx, valid
