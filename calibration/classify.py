"""Regime labelling.

Primary: the agentic repo's deterministic threshold classifier, ported exactly
(agent_tools.score_*_dimension and classify_regime; config.yaml
METHODOLOGY_PARAMS):
  * z-score of each indicator against its history up to and including month t
    (expanding window, point-in-time), sample std (ddof=1), minimum 24 obs;
  * dimension score = weighted sum of z-scores, clipped to [-3, 3];
  * membership(regime) = number of threshold conditions met (bounds inclusive);
  * softmax over memberships; argmax, ties -> first regime in order.

Cross-check: a 4-state Gaussian hidden Markov model on the four dimension
scores, fitted by EM (Baum-Welch; Hamilton 1989). Implemented here directly in
numpy (forward-backward with scaling), with multiple deterministic random starts.
States are mapped to the named regimes by the assignment that maximises
agreement with the primary labels (Hungarian algorithm).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal

REGIMES = ("Expansion", "Late-cycle", "Recession", "Recovery")

# config.yaml (agentic repo) METHODOLOGY_PARAMS.MACRO_SCORING_WEIGHTS, mapped to euro-area indicators
WEIGHTS = {
    "s_g": {"gdp_yoy": 0.6, "neg_unemp_change": 0.4},
    "s_pi": {"hicp_yoy": 0.7, "hicp_mom_sa": 0.3},
    "s_pol": {"dfr": 0.5, "dfr_3m_change": 0.5},
    "s_fc": {"ciss": 0.6, "ciss_change": 0.4},
}
DIMS = ("s_g", "s_pi", "s_pol", "s_fc")
# config.yaml MACRO_REGIME_THRESHOLDS (the repo marks these "UNSPECIFIED IN MANUSCRIPT"; frozen placeholders)
THRESHOLDS = {
    "Expansion": {"s_g": (0.0, None), "s_pi": (None, 1.0), "s_pol": (None, 0.5), "s_fc": (None, 0.5)},
    "Late-cycle": {"s_g": (0.0, None), "s_pi": (1.0, None), "s_pol": (0.5, None), "s_fc": (None, None)},
    "Recession": {"s_g": (None, 0.0), "s_pi": (None, None), "s_pol": (None, None), "s_fc": (0.5, None)},
    "Recovery": {"s_g": (None, 0.0), "s_pi": (None, 0.0), "s_pol": (None, 0.0), "s_fc": (None, 0.5)},
}
MIN_OBS = 24
CLIP = 3.0


def expanding_z(x: np.ndarray, min_obs: int = MIN_OBS) -> np.ndarray:
    out = np.full_like(x, np.nan)
    for t in range(len(x)):
        h = x[: t + 1]
        h = h[~np.isnan(h)]
        if np.isnan(x[t]) or len(h) < min_obs:
            continue
        sd = h.std(ddof=1)
        if sd > 1e-8:
            out[t] = (x[t] - h.mean()) / sd
    return out


def dimension_scores(ind: dict[str, np.ndarray]) -> np.ndarray:
    """(T, 4) dimension scores; NaN where any input lacks history."""
    cols = []
    for d in DIMS:
        s = sum(w * expanding_z(ind[k]) for k, w in WEIGHTS[d].items())
        cols.append(np.clip(s, -CLIP, CLIP))
    return np.stack(cols, axis=1)


def classify(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Labels (T,) in 0..3 (-1 if undefined) and softmax probabilities (T, 4)."""
    T = scores.shape[0]
    labels = np.full(T, -1)
    probs = np.full((T, 4), np.nan)
    for t in range(T):
        s = scores[t]
        if np.any(np.isnan(s)):
            continue
        m = np.zeros(4)
        for i, r in enumerate(REGIMES):
            for j, d in enumerate(DIMS):
                lo, hi = THRESHOLDS[r][d]
                if (lo is None or s[j] >= lo) and (hi is None or s[j] <= hi):
                    m[i] += 1
        e = np.exp(m - m.max())
        p = e / e.sum()
        probs[t] = p
        labels[t] = int(np.argmax(p))          # argmax returns the first maximum: repo tie-break
    return labels, probs


# ------------------------------------------------------------------ HMM cross-check
@dataclass
class HMMFit:
    loglik: float
    pi0: np.ndarray
    A: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    states: np.ndarray        # Viterbi-free: argmax of smoothed posteriors
    post: np.ndarray


def _emission(X: np.ndarray, means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    return np.stack([multivariate_normal(means[k], covs[k], allow_singular=False).pdf(X)
                     for k in range(len(means))], axis=1)


def _forward_backward(B: np.ndarray, A: np.ndarray, pi0: np.ndarray):
    T, K = B.shape
    alpha = np.zeros((T, K)); c = np.zeros(T)
    alpha[0] = pi0 * B[0]; c[0] = alpha[0].sum(); alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[t]; c[t] = alpha[t].sum(); alpha[t] /= c[t]
    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    gamma = alpha * beta
    xi = np.zeros((K, K))
    for t in range(T - 1):
        xi += (alpha[t][:, None] * A * (B[t + 1] * beta[t + 1])[None, :]) / c[t + 1]
    return gamma, xi, np.log(c).sum()


def fit_hmm(X: np.ndarray, k: int = 4, n_starts: int = 20, iters: int = 300, seed: int = 0,
            ridge: float = 1e-3) -> HMMFit:
    """Gaussian HMM by EM with `n_starts` deterministic random initialisations; best log-likelihood wins."""
    rng = np.random.default_rng(seed)
    T, d = X.shape
    best = None
    for _ in range(n_starts):
        means = X[rng.choice(T, k, replace=False)]
        covs = np.repeat(np.cov(X.T)[None] + ridge * np.eye(d), k, axis=0)
        A = np.full((k, k), 0.1 / (k - 1)); np.fill_diagonal(A, 0.9)
        pi0 = np.full(k, 1.0 / k)
        prev = -np.inf
        try:
            for _ in range(iters):
                B = _emission(X, means, covs) + 1e-300
                g, xi, ll = _forward_backward(B, A, pi0)
                pi0 = g[0] / g[0].sum()
                A = xi / xi.sum(axis=1, keepdims=True)
                w = g.sum(axis=0)
                means = (g.T @ X) / w[:, None]
                for j in range(k):
                    D = X - means[j]
                    covs[j] = (g[:, j, None] * D).T @ D / w[j] + ridge * np.eye(d)
                if ll - prev < 1e-7:
                    break
                prev = ll
        except (np.linalg.LinAlgError, ValueError):
            continue
        if best is None or ll > best.loglik:
            best = HMMFit(ll, pi0, A, means.copy(), covs.copy(), g.argmax(axis=1), g)
    if best is None:
        raise RuntimeError("HMM estimation failed for all starts.")
    return best


def align_states(hmm_states: np.ndarray, labels: np.ndarray, k: int = 4) -> tuple[np.ndarray, np.ndarray, dict]:
    """Map HMM states to regime indices maximising agreement; returns mapped states, confusion matrix, mapping."""
    C = np.zeros((k, k))
    for a, b in zip(hmm_states, labels):
        C[a, b] += 1
    rows, cols = linear_sum_assignment(-C)
    mapping = dict(zip(rows, cols))
    mapped = np.array([mapping[s] for s in hmm_states])
    conf = np.zeros((k, k))
    for a, b in zip(labels, mapped):
        conf[a, b] += 1
    return mapped, conf, {int(a): int(b) for a, b in mapping.items()}


def cohens_kappa(conf: np.ndarray) -> float:
    n = conf.sum()
    po = np.trace(conf) / n
    pe = (conf.sum(axis=0) * conf.sum(axis=1)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0
