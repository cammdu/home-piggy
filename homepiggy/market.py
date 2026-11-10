"""Regime-switching market simulator (monthly).

Method (docs/METHODS.md, section 2)
-----------------------------------
Given the regime path s_t, the six state variables evolve as follows. Here
z_t ~ N(0, R[s_t]) is a jointly Gaussian innovation vector, built as
z = L[s] e with e ~ iid N(0, I) and L the Cholesky factor of R[s].

Growth variables (log returns):
  equity:  x_t = mu_e[s_t]/12 + sigma_e[s_t]/sqrt(12) * z_eq
  house:   g_t = mu_h[s_t]/12 + phi * (g_{t-1} - mu_h[s_{t-1}]/12)
                 + sigma_h[s_t]/sqrt(12) * z_h
           (AR(1) in monthly growth: house prices show momentum, i.e.
            positively autocorrelated returns; phi = 0 gives iid growth)

Level variables (short rate, 2y-minus-short term spread, mortgage rate,
inflation) follow a regime-switching discretised Ornstein-Uhlenbeck / AR(1):
  y_t = y_{t-1} + kappa * (theta[s_t] - y_{t-1}) + sigma[s_t]/sqrt(12) * z_y
with kappa = 1 - 0.5**(1/half_life_months), so a deviation halves in
`half_life_months`. Rates are NOT floored. Euro rates were negative from 2014
to 2022, so a zero floor would be wrong.

Narrative shocks are deterministic additions, spread evenly over
`duration_months`:
  * equity / house shocks add to the log *level* (not to the AR state), so a
    -20% house shock moves the index by exactly exp(-0.2231)-1 = -20%;
  * rate / inflation shocks add to the level and then decay through the
    AR(1) at the regime's mean-reversion speed.

Correlation matrices are repaired to the nearest positive-definite matrix by
eigenvalue clipping and rescaled to a unit diagonal (the PSD repair also used
by the agentic repo's ``enforce_psd``) before the Cholesky factorisation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import LEVEL_VARS, STATE_VARS, MarketParams

IDX = {v: i for i, v in enumerate(STATE_VARS)}


@dataclass(frozen=True)
class ShockEvent:
    """Deterministic shock, spread evenly over months [month, month + duration).

    Units: equity_log / house_log are log changes (ln(1 + simple change)); rate
    fields are decimal levels (0.01 = 100 bp); inflation is decimal p.a.
    """

    month: int
    duration_months: int = 1
    equity_log: float = 0.0
    house_log: float = 0.0
    short_rate: float = 0.0
    term_spread: float = 0.0
    mortgage_rate: float = 0.0
    inflation: float = 0.0


@dataclass
class MarketPaths:
    regimes: np.ndarray        # (P, T+1) int
    equity_logret: np.ndarray  # (P, T) monthly log total return
    equity_index: np.ndarray   # (P, T+1) total-return index, starts at 1
    house_index: np.ndarray    # (P, T+1) house price index, starts at 1
    short_rate: np.ndarray     # (P, T+1)
    term_spread: np.ndarray    # (P, T+1)
    mortgage_rate: np.ndarray  # (P, T+1)
    inflation: np.ndarray      # (P, T+1)
    cpi: np.ndarray            # (P, T+1) consumer price index, starts at 1

    @property
    def y2(self) -> np.ndarray:
        """2-year government yield = short rate + term spread."""
        return self.short_rate + self.term_spread

    @property
    def n_paths(self) -> int:
        return self.regimes.shape[0]

    @property
    def n_months(self) -> int:
        return self.regimes.shape[1] - 1


def nearest_correlation(C: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Eigenvalue-clipping PSD repair, rescaled to unit diagonal."""
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, eps, None)
    A = (V * w) @ V.T
    d = np.sqrt(np.diag(A))
    A = A / np.outer(d, d)
    np.fill_diagonal(A, 1.0)
    return A


def cholesky_factors(corr: np.ndarray) -> np.ndarray:
    return np.stack([np.linalg.cholesky(nearest_correlation(C)) for C in corr])


def shock_matrix(shocks: Sequence[ShockEvent], n_months: int) -> np.ndarray:
    """(T, 6) array of per-month deterministic shock increments."""
    S = np.zeros((n_months, len(STATE_VARS)))
    for ev in shocks:
        if ev.duration_months < 1:
            raise ValueError("Shock duration must be >= 1 month.")
        if not (0 <= ev.month and ev.month + ev.duration_months <= n_months):
            raise ValueError(f"Shock {ev} falls outside the horizon of {n_months} months.")
        sl = slice(ev.month, ev.month + ev.duration_months)
        n = ev.duration_months
        S[sl, IDX["equity"]] += ev.equity_log / n
        S[sl, IDX["house"]] += ev.house_log / n
        for v in LEVEL_VARS:
            S[sl, IDX[v]] += getattr(ev, v) / n
    return S


def simulate_market(
    params: MarketParams,
    regimes: np.ndarray,
    normals: np.ndarray,
    shocks: Sequence[ShockEvent] = (),
) -> MarketPaths:
    """Simulate all state variables along given regime paths.

    regimes : (P, T+1) regime indices; normals : (P, T, 6) iid N(0, 1).
    """
    n_paths, T1 = regimes.shape
    T = T1 - 1
    if normals.shape != (n_paths, T, len(STATE_VARS)):
        raise ValueError(f"normals must have shape {(n_paths, T, len(STATE_VARS))}.")
    L = cholesky_factors(params.corr)                 # (K, 6, 6)
    S = shock_matrix(shocks, T)
    sq12 = np.sqrt(12.0)

    eq = np.empty((n_paths, T))
    house_log = np.zeros((n_paths, T1))
    lv = {v: np.empty((n_paths, T1)) for v in LEVEL_VARS}
    for v in LEVEL_VARS:
        lv[v][:, 0] = params.levels[v].initial
    kappa = {v: 1.0 - 0.5 ** (1.0 / params.levels[v].half_life_months) for v in LEVEL_VARS}

    mu_h = params.house.mu / 12.0
    g_prev = mu_h[regimes[:, 0]]   # start with no growth deviation from the regime mean
    phi = params.house_ar1

    for t in range(T):
        s = regimes[:, t + 1]
        z = np.einsum("pij,pj->pi", L[s], normals[:, t, :])
        eq[:, t] = params.equity.mu[s] / 12.0 + params.equity.sigma[s] / sq12 * z[:, IDX["equity"]] \
            + S[t, IDX["equity"]]
        g = mu_h[s] + phi * (g_prev - mu_h[regimes[:, t]]) + params.house.sigma[s] / sq12 * z[:, IDX["house"]]
        house_log[:, t + 1] = house_log[:, t] + g + S[t, IDX["house"]]
        g_prev = g
        for v in LEVEL_VARS:
            p = params.levels[v]
            prev = lv[v][:, t]
            lv[v][:, t + 1] = prev + kappa[v] * (p.theta[s] - prev) + p.sigma[s] / sq12 * z[:, IDX[v]] + S[t, IDX[v]]

    infl = lv["inflation"]
    cpi = np.ones((n_paths, T1))
    cpi[:, 1:] = np.cumprod(1.0 + infl[:, 1:] / 12.0, axis=1)
    eq_index = np.ones((n_paths, T1))
    eq_index[:, 1:] = np.exp(np.cumsum(eq, axis=1))
    return MarketPaths(
        regimes=regimes, equity_logret=eq, equity_index=eq_index, house_index=np.exp(house_log),
        short_rate=lv["short_rate"], term_spread=lv["term_spread"], mortgage_rate=lv["mortgage_rate"],
        inflation=infl, cpi=cpi,
    )
