"""Macro regimes: a discrete-time Markov chain plus forced narrative paths.

Method (declared in docs/METHODS.md, section 1)
------------------------------------------------
The macro state s_t takes one of K regimes (default the four used by the
agentic-architecture repo: Expansion, Late-cycle, Recession, Recovery).
s_t follows a first-order, time-homogeneous Markov chain with a monthly
transition matrix P, where P[i, j] = Pr(s_{t+1} = j | s_t = i).

Sampling uses the inverse-CDF method on each row of P: draw u ~ U(0, 1), and
the next state is the first j with cumsum(P[i])[j] > u. That is exact for a
categorical distribution.

A narrative scenario can *force* the regime over chosen month ranges; outside
forced ranges the chain evolves freely from the last state. Forcing is
deterministic, so the same uniforms are still consumed in every month. Paths
under different scenarios then stay aligned draw-for-draw (common random
numbers), which keeps comparisons between scenarios paired.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

DEFAULT_REGIMES: tuple[str, ...] = ("Expansion", "Late-cycle", "Recession", "Recovery")


@dataclass(frozen=True)
class RegimeSegment:
    """Force regime `regime` for model months [start, end) (0-based)."""

    start: int
    end: int
    regime: str


def validate_transition_matrix(P: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    """Fail closed unless P is a proper row-stochastic square matrix."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"Transition matrix must be square, got shape {P.shape}.")
    if not np.all(np.isfinite(P)):
        raise ValueError("Transition matrix contains non-finite values.")
    if np.any(P < -atol):
        raise ValueError("Transition matrix has negative probabilities.")
    rows = P.sum(axis=1)
    if not np.allclose(rows, 1.0, atol=1e-6):
        raise ValueError(f"Transition matrix rows must sum to 1, got {rows}.")
    P = np.clip(P, 0.0, None)
    return P / P.sum(axis=1, keepdims=True)


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Stationary distribution pi solving pi P = pi, sum(pi) = 1.

    Solved as a linear system with the normalisation row replacing one balance
    equation. That is exact and avoids eigenvector sign/scale ambiguity. Raises
    if the chain is reducible (singular system).
    """
    P = validate_transition_matrix(P)
    k = P.shape[0]
    A = np.vstack([(P.T - np.eye(k))[:-1], np.ones(k)])
    b = np.zeros(k)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - reducible chain
        raise ValueError("Markov chain has no unique stationary distribution.") from exc
    if np.any(pi < -1e-10):
        raise ValueError("Stationary distribution has negative mass; check P.")
    return np.clip(pi, 0.0, None) / np.clip(pi, 0.0, None).sum()


def expected_durations_months(P: np.ndarray) -> np.ndarray:
    """Expected sojourn in each regime: 1 / (1 - P_ii) (geometric durations)."""
    P = validate_transition_matrix(P)
    stay = np.diag(P)
    with np.errstate(divide="ignore"):
        return np.where(stay < 1.0, 1.0 / (1.0 - stay), np.inf)


def simulate_regimes(
    P: np.ndarray,
    n_paths: int,
    n_months: int,
    initial: np.ndarray,
    uniforms: np.ndarray,
    forced: Sequence[RegimeSegment] = (),
    regime_names: Sequence[str] = DEFAULT_REGIMES,
) -> np.ndarray:
    """Simulate regime indices of shape (n_paths, n_months + 1).

    Column 0 is the regime in force at the start date; column t (t >= 1) is the
    regime governing month t (the interval (t-1, t]).

    Parameters
    ----------
    initial : (n_paths,) int array of starting regime indices.
    uniforms : (n_paths, n_months) U(0,1) draws, supplied by the caller so the
        random stream is controlled centrally (common random numbers).
    forced : month ranges with a fixed regime. Month indices refer to columns
        1..n_months; a segment [a, b) forces columns a+1..b.
    """
    P = validate_transition_matrix(P)
    k = P.shape[0]
    if len(regime_names) != k:
        raise ValueError("regime_names length does not match the transition matrix.")
    if uniforms.shape != (n_paths, n_months):
        raise ValueError("uniforms has the wrong shape.")
    initial = np.asarray(initial, dtype=int)
    if initial.shape != (n_paths,) or initial.min() < 0 or initial.max() >= k:
        raise ValueError("initial regimes out of range.")

    force_map = np.full(n_months, -1, dtype=int)
    name_to_idx = {n: i for i, n in enumerate(regime_names)}
    for seg in forced:
        if seg.regime not in name_to_idx:
            raise ValueError(f"Unknown regime '{seg.regime}' in scenario.")
        if not (0 <= seg.start < seg.end <= n_months):
            raise ValueError(f"Segment {seg} lies outside the horizon 0..{n_months}.")
        force_map[seg.start:seg.end] = name_to_idx[seg.regime]

    cum = np.cumsum(P, axis=1)
    cum[:, -1] = 1.0 + 1e-12  # guard against rounding at the top of the CDF
    out = np.empty((n_paths, n_months + 1), dtype=np.int8)
    out[:, 0] = initial
    for t in range(n_months):
        nxt = (uniforms[:, t][:, None] < cum[out[:, t]]).argmax(axis=1)
        if force_map[t] >= 0:
            nxt = np.full(n_paths, force_map[t])
        out[:, t + 1] = nxt
    return out


def empirical_transition_matrix(paths: np.ndarray, k: int) -> np.ndarray:
    """Maximum-likelihood estimate of P from observed paths (row-normalised counts)."""
    counts = np.zeros((k, k))
    a = paths[:, :-1].ravel()
    b = paths[:, 1:].ravel()
    np.add.at(counts, (a, b), 1.0)
    rows = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(rows > 0, counts / rows, np.nan)
