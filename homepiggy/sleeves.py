"""The four sleeves: equity, protected equity (put-spread buffer), short bonds, cash.

Method (docs/METHODS.md, section 3)
-----------------------------------
equity:      R_t = exp(x_t) - 1 - TER/12
cash:        R_t = (r_{t-1} - spread)/12            (rate known at start of month)
short bonds: R_t = (y_{t-1} + D * slope_{t-1})/12 - D * (y_t - y_{t-1}) - TER/12
             carry + roll-down - duration x yield change. This is the fixed-
             income building-block formula of the agentic repo
             (``run_fi_cma_builder``), applied month by month. y = 2y yield,
             D = fund modified duration, slope = (y2 - y1) per year with y1
             linearly interpolated between the cash tenor and 2y (the repo's
             linear-interpolation roll-down). Convexity is omitted (as in the
             repo); for D ~ 1.9 and |dy| <= 50 bp/month the error is < 0.25 bp.
protected equity:
             index exposure + long put (K1 = 100%) - short put (K2 = 85-90%)
             [- optional short call to fund the spread]. Held as a ladder of
             L tranches with 12-month outcome periods that reset in staggered
             fashion. Each tranche is marked to market monthly with
             Black-Scholes (options.py). At its reset date a tranche's value
             is reinvested in a fresh structure struck at the then-current
             index level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MarketParams
from .market import MarketPaths
from .options import bs_price, implied_vol, solve_call_strike

SLEEVES: tuple[str, ...] = ("equity", "protected_equity", "short_bonds", "cash")


@dataclass(frozen=True)
class ProductParams:
    """Product design (fund terms). These are design choices, not market data."""

    equity_ter: float = 0.0020
    protected_equity_ter: float = 0.0050
    short_bonds_ter: float = 0.0015
    cash_spread: float = 0.0010          # money-market sleeve earns short rate minus this
    bond_duration: float = 1.9           # modified duration of a 1-3y government bond fund
    bond_tenor_years: float = 2.0        # tenor whose yield is the fund yield
    cash_tenor_years: float = 0.25       # tenor of the short rate (3m)
    buffer_upper_strike: float = 1.00    # K1 / S
    buffer_lower_strike: float = 0.90    # K2 / S  -> buffer absorbs first 10% of losses
    outcome_months: int = 12
    n_tranches: int = 4
    call_funding: float = 0.0            # 0 = pure put spread; 1 = zero-cost (call fully funds)
    max_cap: float = 3.0                 # upper search bound for call strike (K/S)
    option_cost: float = 0.0020          # trading cost per reset, fraction of notional

    def validate(self) -> "ProductParams":
        if not (0.0 < self.buffer_lower_strike < self.buffer_upper_strike <= 1.2):
            raise ValueError("Need 0 < lower strike < upper strike <= 1.2.")
        if self.outcome_months < 1 or self.n_tranches < 1:
            raise ValueError("outcome_months and n_tranches must be >= 1.")
        if self.outcome_months % self.n_tranches != 0:
            raise ValueError("outcome_months must be divisible by n_tranches for an even ladder.")
        if not (0.0 <= self.call_funding <= 1.0):
            raise ValueError("call_funding must be in [0, 1].")
        if not (0.0 < self.bond_duration < 10.0):
            raise ValueError("bond_duration out of range.")
        if not (0.0 <= self.cash_tenor_years < self.bond_tenor_years):
            raise ValueError("cash tenor must be shorter than bond tenor.")
        for f in ("equity_ter", "protected_equity_ter", "short_bonds_ter", "option_cost"):
            if not (0.0 <= getattr(self, f) < 0.05):
                raise ValueError(f"{f} must be in [0, 5%).")
        return self


def equity_returns(m: MarketPaths, prod: ProductParams) -> np.ndarray:
    return np.expm1(m.equity_logret) - prod.equity_ter / 12.0


def cash_returns(m: MarketPaths, spread: float) -> np.ndarray:
    return (m.short_rate[:, :-1] - spread) / 12.0


def bond_returns(m: MarketPaths, prod: ProductParams) -> np.ndarray:
    r, y2 = m.short_rate, m.y2
    w = (1.0 - prod.cash_tenor_years) / (prod.bond_tenor_years - prod.cash_tenor_years)
    y1 = r + (y2 - r) * w
    slope = (y2 - y1) / (prod.bond_tenor_years - 1.0)
    carry_roll = (y2[:, :-1] + prod.bond_duration * slope[:, :-1]) / 12.0
    price = -prod.bond_duration * np.diff(y2, axis=1)
    return carry_roll + price - prod.short_bonds_ter / 12.0


class ProtectedEquitySleeve:
    """Laddered put-spread buffer on the equity total-return index.

    State per path p and tranche j: units n[p, j] of (index + options), strikes
    K1, K2, Kc (index points) and a common expiry month per tranche. Value per
    unit: u = S + P(K1) - P(K2) - C(Kc). Since P(K1) >= P(K2) and C < S, u > 0
    always, so the sleeve can never go negative.
    """

    def __init__(self, m: MarketPaths, params: MarketParams, prod: ProductParams):
        prod.validate()
        self.m, self.params, self.prod = m, params, prod
        P, L = m.n_paths, prod.n_tranches
        self.L = L
        self.n = np.zeros((P, L))
        self.K1 = np.zeros((P, L))
        self.K2 = np.zeros((P, L))
        self.Kc = np.full((P, L), np.inf)
        step = prod.outcome_months // L
        # Staggered first expiries: tranche j first expires at month (j+1)*step,
        # so the initial periods are shorter; afterwards every period is outcome_months.
        self.expiry = np.array([(j + 1) * step for j in range(L)], dtype=int)
        self.t = 0
        self.premium_paid = np.zeros(P)        # cumulative net option premium per unit value (diagnostic)
        for j in range(L):
            self._strike(j, 0, tenor_months=int(self.expiry[j]))

    # -- pricing helpers -------------------------------------------------------
    def _sigma_atm(self, t: int) -> np.ndarray:
        s = self.m.regimes[:, t]
        return self.params.implied_vol.vrp_ratio * self.params.equity.sigma[s]

    def _unit_values(self, t: int) -> np.ndarray:
        S = self.m.equity_index[:, t][:, None]
        r = self.m.short_rate[:, t][:, None]
        tau = np.maximum(self.expiry - t, 0)[None, :] / 12.0
        iv = self.params.implied_vol
        atm = self._sigma_atm(t)[:, None]
        p1 = bs_price(S, self.K1, tau, r, implied_vol(atm, self.K1 / S, iv.skew, iv.min_vol), "put")
        p2 = bs_price(S, self.K2, tau, r, implied_vol(atm, self.K2 / S, iv.skew, iv.min_vol), "put")
        finite = np.isfinite(self.Kc)
        Kc = np.where(finite, self.Kc, S)
        c = np.where(finite, bs_price(S, Kc, tau, r, implied_vol(atm, Kc / S, iv.skew, iv.min_vol), "call"), 0.0)
        return S + p1 - p2 - c

    def _strike(self, j: int, t: int, tenor_months: int) -> np.ndarray:
        """Set new strikes for tranche j at month t; return the per-unit purchase cost."""
        prod, iv = self.prod, self.params.implied_vol
        S = self.m.equity_index[:, t]
        r = self.m.short_rate[:, t]
        tau = tenor_months / 12.0
        atm = self._sigma_atm(t)
        k1, k2 = prod.buffer_upper_strike, prod.buffer_lower_strike
        p1 = bs_price(S, k1 * S, tau, r, implied_vol(atm, k1, iv.skew, iv.min_vol), "put")
        p2 = bs_price(S, k2 * S, tau, r, implied_vol(atm, k2, iv.skew, iv.min_vol), "put")
        spread = p1 - p2
        self.K1[:, j], self.K2[:, j] = k1 * S, k2 * S
        if prod.call_funding > 0:
            kc = solve_call_strike(S, prod.call_funding * spread, tau, r, atm, iv.skew, iv.min_vol,
                                   k_lo=1.0, k_hi=prod.max_cap)
            c = bs_price(S, kc * S, tau, r, implied_vol(atm, kc, iv.skew, iv.min_vol), "call")
            self.Kc[:, j] = kc * S
        else:
            c = np.zeros_like(S)
            self.Kc[:, j] = np.inf
        self.expiry[j] = t + tenor_months
        return S + spread - c + prod.option_cost * S

    # -- public API --------------------------------------------------------------
    def value(self, t: int | None = None) -> np.ndarray:
        t = self.t if t is None else t
        return (self.n * self._unit_values(t)).sum(axis=1)

    def advance(self, t: int) -> np.ndarray:
        """Move to month t (mark to market, roll expiring tranches). Returns sleeve value."""
        if t != self.t + 1:
            raise RuntimeError("ProtectedEquitySleeve must be advanced one month at a time.")
        self.t = t
        # monthly TER, charged by reducing units
        self.n *= 1.0 - self.prod.protected_equity_ter / 12.0
        for j in range(self.L):
            if self.expiry[j] == t:
                u_exp = self._unit_values(t)[:, j]          # tau = 0 -> intrinsic payoff
                V = self.n[:, j] * u_exp
                if t < self.m.n_months:                     # roll unless the horizon ends here
                    cost = self._strike(j, t, self.prod.outcome_months)
                    self.n[:, j] = V / cost
                else:
                    S = self.m.equity_index[:, t]
                    self.K1[:, j] = self.K2[:, j] = 0.0
                    self.Kc[:, j] = np.inf
                    self.n[:, j] = V / S                    # plain index units, no options
        return self.value(t)

    def set_value(self, target: np.ndarray) -> None:
        """Buy/sell pro rata across tranches at mark-to-market so the sleeve equals `target`."""
        target = np.maximum(target, 0.0)
        u = self._unit_values(self.t)
        cur = (self.n * u).sum(axis=1)
        has = cur > 1e-12
        scale = np.where(has, target / np.where(has, cur, 1.0), 0.0)
        self.n = self.n * scale[:, None]
        empty = ~has & (target > 0)
        if np.any(empty):
            self.n[empty] = (target[empty, None] / self.L) / u[empty]
