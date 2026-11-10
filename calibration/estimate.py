"""Estimate model parameters from a (possibly resampled) monthly panel with regime labels.

Data layout: rows in time order (or bootstrap order), each with the model
variables and a regime label. ``valid_prev[j]`` is True when row j-1 is the true
calendar predecessor of row j. Only those pairs enter transition counts and
AR(1) regressions, so bootstrap block joins never create spurious transitions.

Estimators (docs/CALIBRATION.md, section 3):
  P               MLE: row-normalised transition counts over valid pairs
  equity mu,sigma sample mean / std (ddof=1) of monthly log returns per regime, x12 / x sqrt(12)
  house           state-space MLE (Kalman filter): observed CBS growth = latent true growth +
                  differenced index measurement noise; latent growth follows the engine's AR(1)
                  around regime means. The engine simulates the latent (true) growth.
  levels          y_t = a y_{t-1} + sum_k b_k 1[s_t = k] + e_t by OLS; kappa = 1 - a,
                  theta_k = b_k / kappa, sigma[k] = residual std in regime k x sqrt(12).
                  Constrained LS: theta_k must lie in the observed range of y (near-unit-root
                  rates otherwise give theta = b/kappa far outside history); half-life searched
                  on [1, HALF_LIFE_CAP] months. Binding constraints are reported.
  correlation     pooled residual correlation shrunk to I, then each regime's correlation
                  shrunk towards the pooled matrix (Ledoit-Wolf 2004 optimal intensity)
Fallback (bootstrap only): a regime with fewer than MIN_REGIME_OBS rows (or a
transition row with no observed exits) takes the full-sample estimate for that
regime. Every use is counted and reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

LEVELS = ("short_rate", "term_spread", "mortgage_rate", "inflation")
ORDER = ("equity", "house", "short_rate", "term_spread", "mortgage_rate", "inflation")
HALF_LIFE_CAP = 240.0          # months (upper end of the half-life search)
MIN_REGIME_OBS = 12
PHI_MAX = 0.99                 # house-growth persistence bound (= engine validation bound)


@dataclass
class Estimates:
    params: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    fallbacks: list[str] = field(default_factory=list)


def ledoit_wolf_corr(Z: np.ndarray, target: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) optimal shrinkage of the correlation of columns of Z.

    Columns are standardised; target F = I by default (or a supplied correlation
    matrix, e.g. the pooled estimate). Intensity delta = min(1, b^2 / d^2) with
    d^2 = ||S - F||^2 and b^2 = n^-2 sum_k ||x_k x_k' - S||^2 (the estimated sampling
    variance of S). Returns (shrunk correlation, delta).
    """
    X = (Z - Z.mean(axis=0)) / Z.std(axis=0, ddof=0)
    n, p = X.shape
    S = X.T @ X / n
    F = np.eye(p) if target is None else np.asarray(target, float)
    d2 = np.sum((S - F) ** 2)
    b2 = sum(np.sum((np.outer(x, x) - S) ** 2) for x in X) / n ** 2
    b2 = min(b2, d2)
    delta = b2 / d2 if d2 > 0 else 1.0
    C = delta * F + (1 - delta) * S
    dg = np.sqrt(np.diag(C))
    C = C / np.outer(dg, dg)
    return C, float(delta)


def _transition(labels: np.ndarray, valid_prev: np.ndarray, k: int):
    counts = np.zeros((k, k))
    j = np.where(valid_prev[1:] & (labels[1:] >= 0) & (labels[:-1] >= 0))[0] + 1
    np.add.at(counts, (labels[j - 1], labels[j]), 1)
    return counts


def _house_ar(g: np.ndarray, lab: np.ndarray, vp: np.ndarray, k: int, iters: int = 200):
    ok = ~np.isnan(g) & (lab >= 0)
    pair = np.where(vp[1:] & ok[1:] & ok[:-1])[0] + 1
    mu = np.array([g[ok & (lab == r)].mean() if np.any(ok & (lab == r)) else np.nan for r in range(k)])
    phi = 0.0
    for _ in range(iters):
        d = g - mu[np.clip(lab, 0, k - 1)]
        x, y = d[pair - 1], d[pair]
        phi_new = float(x @ y / (x @ x))
        # mu given phi: g_t - phi g_{t-1} = mu[s_t] - phi mu[s_{t-1}] + e  (linear in mu)
        Xd = np.zeros((len(pair), k))
        Xd[np.arange(len(pair)), lab[pair]] += 1.0
        Xd[np.arange(len(pair)), lab[pair - 1]] -= phi_new
        yv = g[pair] - phi_new * g[pair - 1]
        present = Xd.any(axis=0)
        sol = np.linalg.lstsq(Xd[:, present], yv, rcond=None)[0]
        mu_new = mu.copy(); mu_new[present] = sol
        conv = abs(phi_new - phi) < 1e-12 and np.allclose(mu_new[present], mu[present], atol=1e-14)
        phi, mu = phi_new, mu_new
        if conv:
            break
    resid = np.full_like(g, np.nan)
    d = g - mu[np.clip(lab, 0, k - 1)]
    resid[pair] = d[pair] - phi * d[pair - 1]
    return mu, phi, resid


def _segments(vp: np.ndarray, ok: np.ndarray) -> list[np.ndarray]:
    """Index arrays of maximal runs of consecutive valid rows (a new run starts at each invalid link)."""
    segs, cur = [], []
    for j in range(len(vp)):
        if not ok[j]:
            if cur:
                segs.append(np.array(cur)); cur = []
            continue
        if cur and not vp[j]:
            segs.append(np.array(cur)); cur = []
        cur.append(j)
    if cur:
        segs.append(np.array(cur))
    return segs


def kalman_house(g: np.ndarray, lab: np.ndarray, segs: list, mu, phi: float, sig, se: float,
                 want: bool = False):
    """Kalman filter for the house state-space model, scalar arithmetic (3x3 algebra unrolled).

    State a = (d_t, e_t, e_{t-1}); transition d' = phi d, e' = 0, e_prev' = e; Q = diag(sig[s]^2, se^2, 0);
    observation g_t = mu[s_t] + d_t + e_t - e_{t-1}. Returns log-likelihood and, if `want`, the
    standardised innovations v/sqrt(F) and filtered eps_t = d_t|t - phi d_{t-1}|t-1.
    """
    LOG2PI = 1.8378770664093453
    ll = 0.0
    se2 = se * se
    z = np.full(len(g), np.nan) if want else None
    eps = np.full(len(g), np.nan) if want else None
    for sg in segs:
        s0 = lab[sg[0]]
        a0 = a1 = a2 = 0.0
        p00 = sig[s0] ** 2 / (1.0 - phi * phi); p11 = se2; p22 = se2; p01 = p02 = p12 = 0.0
        d_prev = 0.0
        first = True
        for t in sg:
            s = lab[t]
            if not first:
                # predict: a = T a ; P = T P T' + Q
                a0, a1, a2 = phi * a0, 0.0, a1
                n00 = phi * phi * p00 + sig[s] ** 2
                n02 = phi * p01
                n22 = p11
                p00, p01, p02, p11, p12, p22 = n00, 0.0, n02, se2, 0.0, n22
            v = g[t] - mu[s] - (a0 + a1 - a2)
            pz0 = p00 + p01 - p02
            pz1 = p01 + p11 - p12
            pz2 = p02 + p12 - p22
            F = pz0 + pz1 - pz2
            if F <= 0.0:
                return -1e300, z, eps
            k0, k1, k2 = pz0 / F, pz1 / F, pz2 / F
            a0 += k0 * v; a1 += k1 * v; a2 += k2 * v
            p00 -= k0 * pz0; p01 -= k0 * pz1; p02 -= k0 * pz2
            p11 -= k1 * pz1; p12 -= k1 * pz2; p22 -= k2 * pz2
            ll += -0.5 * (LOG2PI + np.log(F) + v * v / F)
            if want:
                z[t] = v / np.sqrt(F)
                if not first:
                    eps[t] = a0 - phi * d_prev
            d_prev = a0
            first = False
    return ll, z, eps


def _house_kalman(g: np.ndarray, lab: np.ndarray, vp: np.ndarray, k: int, start: dict | None = None):
    """Exact Gaussian MLE of the house-growth state-space model (Kalman filter).

    Observed monthly log growth of the CBS index:   g_t = mu[s_t] + d_t + e_t - e_{t-1}
    Latent true growth deviation:                   d_t = phi d_{t-1} + eps_t,  eps_t ~ N(0, sig[s_t]^2)
    Index (level) measurement noise:                e_t ~ iid N(0, sig_e^2)
    Each run of consecutive months starts from the stationary state distribution.
    The frequency-weighted mean of mu is constrained to the sample mean of g (method of moments).
    L-BFGS-B on unconstrained transforms (phi = tanh, sigmas = exp). Full-sample fits use
    three starting values for phi (0.5, 0.8, 0.95); bootstrap fits warm-start from the
    full-sample optimum (`start`).
    """
    from scipy.optimize import minimize
    ok = ~np.isnan(g) & (lab >= 0)
    segs = [list(sg) for sg in _segments(vp, ok) if len(sg) >= 2]
    present = np.array([np.any(lab[ok] == r) for r in range(k)])
    gl = g.tolist(); ll_lab = lab.tolist()

    # Mean constraint (method of moments): sum_k freq_k * mu_k = sample mean of g. With phi near 1 the MLE
    # of the level is weakly identified (it trades off against the unobserved initial deviation) and was
    # biased by about +2 pp p.a.; the sample mean is the robust estimator of a stationary process's mean.
    # Free parameters: delta_1..delta_{K-1} (delta_0 = 0), atanh(phi), log sig (K), log sig_e.
    gbar = float(np.mean(g[ok]))
    freq = np.array([np.mean(lab[ok] == r) for r in range(k)])

    def mu_of(delta_free):
        d = np.r_[0.0, delta_free]
        return gbar + d - freq @ d

    def unpack(th):
        return (mu_of(th[: k - 1]), float(np.tanh(th[k - 1])), np.exp(th[k: 2 * k]), float(np.exp(th[2 * k])))

    def obj(th):
        mu, phi, sig, se = unpack(th)
        return -kalman_house(gl, ll_lab, segs, mu.tolist(), phi, sig.tolist(), se)[0]

    def pack(mu_start, phi_s, sig_s, se_s):
        mu_start = np.nan_to_num(np.asarray(mu_start, float), nan=gbar)
        d = mu_start - mu_start[0]
        return np.r_[d[1:], np.arctanh(phi_s), np.log(sig_s), np.log(se_s)]

    if start is not None:
        starts = [pack(np.asarray(start["mu"]) / 12, start["phi"],
                       np.asarray(start["sigma"]) / np.sqrt(12), start["sig_e"])]
    else:
        mu0, phi0, res0 = _house_ar(g, lab, vp, k)
        sd0 = float(np.nanstd(res0))
        starts = [pack(mu0, ph, np.full(k, sd0 * 0.5), sd0 * 0.5) for ph in (0.5, 0.8, 0.95)]
    pidx = k - 1
    bnds = [(None, None)] * (k - 1) + [(-np.arctanh(PHI_MAX), np.arctanh(PHI_MAX))] + [(None, None)] * (k + 1)
    best = None
    for th in starts:
        th = np.where(np.isfinite(th), th, 0.0)
        th[pidx] = np.clip(th[pidx], -np.arctanh(PHI_MAX), np.arctanh(PHI_MAX))
        r = minimize(obj, th, method="L-BFGS-B", bounds=bnds, options={"maxiter": 3000})
        if best is None or r.fun < best.fun:
            best = r
    mu, phi, sig, se = unpack(best.x)
    ll, zz, eps = kalman_house(g, lab, segs, mu, phi, sig, se, want=True)
    mu = np.where(present, mu, np.nan); sig = np.where(present, sig, np.nan)
    return mu, phi, sig, se, eps, float(ll), bool(best.success), zz


def _level_ar(y: np.ndarray, lab: np.ndarray, vp: np.ndarray, k: int):
    """Constrained least squares for y_t = (1-kappa) y_{t-1} + kappa theta[s_t] + e_t.

    Constraint: every theta_k lies within the observed range of y (the long-run level a
    regime pulls towards cannot be extrapolated beyond the sample). For a fixed kappa
    the SSR is separable and quadratic in each theta_k, so the constrained optimum is
    the regime mean of z_t = (y_t - (1-kappa) y_{t-1}) / kappa, clipped to the range.
    kappa is found by profiling the SSR over half-lives 1..HALF_LIFE_CAP months
    (log grid, then bounded Brent refinement). With no binding bound this equals OLS.
    """
    from scipy.optimize import minimize_scalar
    ok = ~np.isnan(y) & (lab >= 0)
    pair = np.where(vp[1:] & ok[1:] & ok[:-1])[0] + 1
    lo, hi = float(np.nanmin(y[ok])), float(np.nanmax(y[ok]))
    yp, yc, lc = y[pair - 1], y[pair], lab[pair]
    present = np.array([np.any(lc == r) for r in range(k)])

    def theta_for(kappa: float) -> np.ndarray:
        z = (yc - (1 - kappa) * yp) / kappa
        th = np.array([z[lc == r].mean() if present[r] else np.nan for r in range(k)])
        return np.clip(th, lo, hi)

    def ssr_hl(hl: float) -> float:
        kappa = 1 - 0.5 ** (1 / hl)
        th = theta_for(kappa)
        e = yc - (1 - kappa) * yp - kappa * th[lc]
        return float(e @ e)

    grid = np.geomspace(1.0, HALF_LIFE_CAP, 300)
    vals = np.array([ssr_hl(h) for h in grid])
    i = int(np.argmin(vals))
    a_, b_ = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    hl = float(minimize_scalar(ssr_hl, bounds=(a_, b_), method="bounded", options={"xatol": 1e-6}).x) if b_ > a_ else float(grid[i])
    kappa = 1 - 0.5 ** (1 / hl)
    z = (yc - (1 - kappa) * yp) / kappa
    raw = np.array([z[lc == r].mean() if present[r] else np.nan for r in range(k)])
    theta = np.clip(raw, lo, hi)
    bound = [bool(present[r] and (raw[r] < lo or raw[r] > hi)) for r in range(k)]
    resid = np.full_like(y, np.nan)
    resid[pair] = yc - (1 - kappa) * yp - kappa * theta[lc]
    at_cap = hl >= HALF_LIFE_CAP - 1e-6
    return kappa, theta, resid, hl, {"half_life_at_cap": bool(at_cap), "theta_bound_binding": bound,
                                     "range": [lo, hi]}


def estimate(rows: dict[str, np.ndarray], labels: np.ndarray, valid_prev: np.ndarray,
             regimes: tuple[str, ...], fallback: dict[str, Any] | None = None) -> Estimates:
    k = len(regimes)
    lab = labels.astype(int)
    est = Estimates(params={})
    diag = est.diagnostics
    counts_reg = np.array([(lab == r).sum() for r in range(k)])
    diag["months_per_regime"] = dict(zip(regimes, counts_reg.tolist()))
    thin = counts_reg < MIN_REGIME_OBS

    # ---- transitions
    C = _transition(lab, valid_prev, k)
    diag["transition_counts"] = C.tolist()
    P = np.zeros((k, k))
    for r in range(k):
        if C[r].sum() > 0 and not thin[r]:
            P[r] = C[r] / C[r].sum()
        elif fallback is not None:
            P[r] = fallback["transition_matrix"][r]
            est.fallbacks.append(f"transition row {regimes[r]}")
        else:
            raise ValueError(f"No observed transitions out of {regimes[r]}.")

    def fb(block: str, field_: str, r: int, val: float) -> float:
        if thin[r] or not np.isfinite(val):
            if fallback is None:
                raise ValueError(f"Regime {regimes[r]} has too few months to estimate {block}.{field_}.")
            est.fallbacks.append(f"{block}.{field_}[{regimes[r]}]")
            return float(fallback[block][field_][r])
        return float(val)

    # ---- equity
    x = rows["equity"]
    eq_mu, eq_sd = [], []
    for r in range(k):
        v = x[(lab == r) & ~np.isnan(x)]
        eq_mu.append(fb("equity", "mu", r, v.mean() * 12 if len(v) > 1 else np.nan))
        eq_sd.append(fb("equity", "sigma", r, v.std(ddof=1) * np.sqrt(12) if len(v) > 1 else np.nan))
    eq_res = x - np.array(eq_mu)[np.clip(lab, 0, k - 1)] / 12

    # ---- house: state-space MLE (latent persistent growth + index measurement noise)
    hs = None
    if fallback is not None and "_house_state_space" in fallback:
        hs = fallback["_house_state_space"]
    h_mu, phi, h_sig, sig_e, h_res, h_ll, h_conv, h_z = _house_kalman(rows["house"], lab, valid_prev, k, start=hs)
    h_sd = [fb("house", "sigma", r, h_sig[r] * np.sqrt(12)) for r in range(k)]
    h_mu_a = [fb("house", "mu", r, h_mu[r] * 12) for r in range(k)]
    diag["house_state_space"] = {"phi": phi, "index_noise_sd_monthly": sig_e, "loglik": h_ll, "converged": h_conv,
                                 "phi_at_bound": bool(abs(phi) >= PHI_MAX - 1e-6)}
    if abs(phi) >= PHI_MAX - 1e-6:
        est.fallbacks.append("house.ar1 at bound 0.99")
    diag["house_std_innovations"] = h_z
    est.params["_house_state_space"] = {"mu": h_mu_a, "phi": phi, "sigma": h_sd, "sig_e": sig_e}
    est.params["equity"] = {"mu": eq_mu, "sigma": eq_sd}
    est.params["house"] = {"mu": h_mu_a, "sigma": h_sd, "ar1": float(phi)}

    # ---- levels
    resid = {"equity": eq_res, "house": h_res}
    for v in LEVELS:
        kappa, theta, res, hl, capped = _level_ar(rows[v], lab, valid_prev, k)
        sd = []
        for r in range(k):
            e = res[(lab == r) & ~np.isnan(res)]
            sd.append(fb(v, "sigma", r, e.std(ddof=1) * np.sqrt(12) if len(e) > 1 else np.nan))
        th = [fb(v, "theta", r, theta[r]) for r in range(k)]
        est.params[v] = {"theta": th, "sigma": sd, "half_life_months": hl}
        diag.setdefault("level_constraints", {})[v] = capped
        resid[v] = res

    # ---- correlations: pooled (all regimes) shrunk to I; each regime shrunk to the pooled matrix
    R = np.stack([resid[v] for v in ORDER], axis=1)
    cc = ~np.isnan(R).any(axis=1) & (lab >= 0)
    pooled, d_pool = ledoit_wolf_corr(R[cc])
    corr, shrink, n_cc = {}, {"_pooled_to_identity": d_pool}, {"_pooled": int(cc.sum())}
    for r, name in enumerate(regimes):
        Z = R[cc & (lab == r)]
        n_cc[name] = int(len(Z))
        if len(Z) >= MIN_REGIME_OBS and np.all(Z.std(axis=0) > 0):
            Cm, dlt = ledoit_wolf_corr(Z, target=pooled)
            corr[name] = Cm.round(6).tolist()
            shrink[name] = dlt
        elif fallback is not None:
            corr[name] = fallback["correlation"][name]
            est.fallbacks.append(f"correlation[{name}]")
        else:
            raise ValueError(f"Too few complete-case months to estimate correlations in {name}.")
    for name in regimes:   # symmetrise exactly after rounding
        Cm = np.asarray(corr[name]); Cm = (Cm + Cm.T) / 2; np.fill_diagonal(Cm, 1.0); corr[name] = Cm.tolist()
    est.params["correlation"] = corr
    diag["lw_shrinkage"] = shrink
    diag["complete_case_months"] = n_cc
    est.params["transition_matrix"] = P.tolist()
    est.params["regimes"] = list(regimes)
    return est
