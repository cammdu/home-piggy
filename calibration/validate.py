"""Model validation against the calibration data (in-sample; see LIMITATIONS).

1. One-step-ahead predictive checks (probability integral transform, PIT).
   For every month t with a known regime label, the model implies a Gaussian
   predictive distribution for each variable given its state at t-1. Under a
   correct model, u_t = Phi((x_t - mean_t) / sd_t) is Uniform(0, 1). We report:
     * Kolmogorov-Smirnov test of uniformity (p-value), and
     * the share of observations outside the central 90% / 98% predictive intervals
       (nominal 10% / 2%). Excess tail exceedances quantify fat tails.
   PITs are serially dependent (overlapping information), so KS p-values are
   indicative rather than exact.
2. Long-horizon distribution check: the share of historical 10-year windows whose
   realised outcome falls inside the model's simulated 5-95% range, for NL house-price
   growth and the change in the mortgage rate, starting each simulation from the
   observed state at the start of the window. With heavily overlapping windows this is
   a descriptive check, not a formal test.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from homepiggy.config import STATE_VARS, market_params_from_dict
from homepiggy.market import simulate_market
from homepiggy.regimes import simulate_regimes

LEVELS = ("short_rate", "term_spread", "mortgage_rate", "inflation")


def pit_checks(rows: dict[str, np.ndarray], lab: np.ndarray, params: dict, house_z: np.ndarray) -> dict:
    out = {}
    k = lab
    eq_mu = np.asarray(params["equity"]["mu"]) / 12
    eq_sd = np.asarray(params["equity"]["sigma"]) / np.sqrt(12)
    x = rows["equity"][1:]
    ok = ~np.isnan(x) & (k[1:] >= 0)
    out["equity"] = _pit(x[ok], eq_mu[k[1:][ok]], eq_sd[k[1:][ok]])

    # house: standardised Kalman innovations (predictive distribution includes index noise)
    zz = house_z[~np.isnan(house_z)]
    out["house"] = _pit(zz, np.zeros_like(zz), np.ones_like(zz))

    for v in LEVELS:
        y = rows[v]
        p = params[v]
        kap = 1 - 0.5 ** (1 / p["half_life_months"])
        th = np.asarray(p["theta"]); s = np.asarray(p["sigma"]) / np.sqrt(12)
        ok = ~np.isnan(y[1:]) & ~np.isnan(y[:-1])
        mean = y[:-1] + kap * (th[k[1:]] - y[:-1])
        out[v] = _pit(y[1:][ok], mean[ok], s[k[1:]][ok])
    return out


def _pit(x, m, s) -> dict:
    s = np.maximum(s, 1e-12)
    u = stats.norm.cdf((x - m) / s)
    ks = stats.kstest(u, "uniform")
    return {"n": int(len(u)), "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
            "outside_90pct": float(np.mean((u < 0.05) | (u > 0.95))),
            "outside_98pct": float(np.mean((u < 0.01) | (u > 0.99)))}


def ten_year_coverage(rows_raw: dict[str, np.ndarray], lab: np.ndarray, params: dict, n_paths: int = 2000,
                      horizon: int = 120, seed: int = 11) -> dict:
    mp = market_params_from_dict(params)
    hidx = np.exp(np.nancumsum(np.nan_to_num(rows_raw["house_raw"])))
    T = len(lab)
    starts = list(range(0, T - horizon, 6))       # every 6 months to limit cost
    res = {"house_10y_log_growth": [], "mortgage_10y_change": []}
    rng = np.random.default_rng(seed)
    for a in starts:
        b = a + horizon
        mp_a = market_params_from_dict({**params, "initial_regime": params["regimes"][lab[a]],
                                        **{v: {**params[v], "initial": float(rows_raw[v][a])} for v in LEVELS
                                           if not np.isnan(rows_raw[v][a])}})
        init = np.full(n_paths, lab[a])
        reg = simulate_regimes(mp_a.P, n_paths, horizon, init, rng.random((n_paths, horizon)), (), mp_a.regimes)
        m = simulate_market(mp_a, reg, rng.standard_normal((n_paths, horizon, len(STATE_VARS))))
        sim_h = np.log(m.house_index[:, -1])
        obs_h = np.log(hidx[b] / hidx[a])
        res["house_10y_log_growth"].append(_cover(sim_h, obs_h))
        if not np.isnan(rows_raw["mortgage_rate"][a]):
            sim_m = m.mortgage_rate[:, -1] - m.mortgage_rate[:, 0]
            obs_m = rows_raw["mortgage_rate"][b] - rows_raw["mortgage_rate"][a]
            res["mortgage_10y_change"].append(_cover(sim_m, obs_m))
    return {k: {"windows": len(v), "share_inside_5_95": float(np.mean([x["inside"] for x in v])) if v else None,
                "mean_percentile_of_outcome": float(np.mean([x["pct"] for x in v])) if v else None}
            for k, v in res.items()}


def _cover(sim: np.ndarray, obs: float) -> dict:
    lo, hi = np.percentile(sim, [5, 95])
    return {"inside": bool(lo <= obs <= hi), "pct": float((sim < obs).mean())}
