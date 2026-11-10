"""Black-Scholes pricing with a regime-dependent implied-volatility smile.

Method (docs/METHODS.md, section 3.3)
-------------------------------------
European options on the equity total-return index are priced with
Black-Scholes (Black & Scholes 1973; Merton 1973). The dividend yield is 0
because the underlying is a *total-return* index, so dividends are already in
the index level. The risk-free rate is the simulated short rate r (simple,
annual), converted to continuous compounding as ln(1 + r).

Implied volatility is a deterministic function of the current regime's
realised equity volatility:

  sigma_ATM = vrp_ratio * sigma_e[s]
  sigma(K)  = max(min_vol, sigma_ATM + skew * (-ln(K / S)))

``vrp_ratio`` > 1 captures the volatility risk premium: implied volatility
usually exceeds realised volatility. ``skew`` captures the equity put skew:
out-of-the-money puts trade at higher implied volatility. Both parameters
come from calibration data, never from judgement.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def implied_vol(sigma_atm: np.ndarray, moneyness: np.ndarray | float, skew: float, min_vol: float) -> np.ndarray:
    """Smile: sigma_ATM + skew * (-ln(K/S)), floored at min_vol. moneyness = K / S."""
    m = np.asarray(moneyness, float)
    safe = np.where(m > 0, m, 1.0)   # K = 0 (expired / no option): vol is irrelevant, use ATM
    return np.maximum(min_vol, sigma_atm + skew * (-np.log(safe)))


def bs_price(
    S: np.ndarray, K: np.ndarray, tau: np.ndarray, r_simple: np.ndarray, sigma: np.ndarray, kind: str
) -> np.ndarray:
    """Vectorised Black-Scholes price (q = 0). tau in years; tau <= 0 -> intrinsic value."""
    S, K, tau, r_simple, sigma = np.broadcast_arrays(
        np.asarray(S, float), np.asarray(K, float), np.asarray(tau, float),
        np.asarray(r_simple, float), np.asarray(sigma, float),
    )
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'.")
    r = np.log1p(np.maximum(r_simple, -0.99))
    intrinsic = np.maximum(S - K, 0.0) if kind == "call" else np.maximum(K - S, 0.0)
    live = (tau > 1e-12) & (K > 0.0)   # K <= 0: worthless put / call worth S (= intrinsic)
    out = np.array(intrinsic, dtype=float)   # np.array (not astype) also handles 0-d scalars
    if np.any(live):
        s, k, t, rr, v = S[live], K[live], tau[live], r[live], np.maximum(sigma[live], 1e-8)
        sqt = v * np.sqrt(t)
        d1 = (np.log(s / k) + (rr + 0.5 * v * v) * t) / sqt
        d2 = d1 - sqt
        disc = np.exp(-rr * t)
        if kind == "call":
            out[live] = s * ndtr(d1) - k * disc * ndtr(d2)
        else:
            out[live] = k * disc * ndtr(-d2) - s * ndtr(-d1)
    return out


def solve_call_strike(
    S: np.ndarray, target_premium: np.ndarray, tau: float, r_simple: np.ndarray, sigma_atm: np.ndarray,
    skew: float, min_vol: float, k_lo: float = 1.0, k_hi: float = 3.0, iters: int = 60,
) -> np.ndarray:
    """Call strike (as K/S ratio) whose premium equals ``target_premium``.

    The call price is strictly decreasing in K (also with the smile, since
    d sigma/dK < 0), so bisection on [k_lo, k_hi] converges; 60 iterations
    gives a precision of 2e-18 * (k_hi - k_lo). If the target exceeds the
    premium at k_lo, k_lo is returned: the overwrite cannot fully fund the
    puts. If it is below the premium at k_hi, k_hi is returned.
    """
    lo = np.full_like(np.asarray(S, float), k_lo)
    hi = np.full_like(lo, k_hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        vol = implied_vol(sigma_atm, mid, skew, min_vol)
        c = bs_price(S, S * mid, tau, r_simple, vol, "call")
        too_cheap = c < target_premium      # strike too high -> move down
        hi = np.where(too_cheap, mid, hi)
        lo = np.where(too_cheap, lo, mid)
    return 0.5 * (lo + hi)
