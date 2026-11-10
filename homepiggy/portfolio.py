"""Monthly portfolio accounting: returns, tax, contributions, rebalancing.

Order of events within month t = 1..T (docs/METHODS.md, section 4.2):
  1. sleeve returns for month t are applied (protected equity marked to market;
     tranches roll at their reset dates);
  2. at each 12-month anniversary, box 3 tax is withdrawn pro rata;
  3. the month-end contribution is invested at the current target weights;
  4. rebalance to target if it is a calendar rebalance month or the L1 drift
     sum_i |w_i - w*_i| exceeds the threshold (the agentic repo's quarterly
     plus drift-trigger policy). Transaction cost = tc * traded value.
The time-weighted NAV index is updated with the pre-flow return in step 1, so
it measures investment performance independently of contributions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .sleeves import SLEEVES, ProtectedEquitySleeve
from .tax_nl import Box3Params, box3_tax

PE = SLEEVES.index("protected_equity")
CASH = SLEEVES.index("cash")


@dataclass(frozen=True)
class RebalancePolicy:
    every_months: int = 3
    drift_threshold_l1: float = 0.05
    transaction_cost: float = 0.0005      # fraction of traded value

    def validate(self) -> "RebalancePolicy":
        if self.every_months < 1 or not (0 <= self.drift_threshold_l1 <= 2) or not (0 <= self.transaction_cost < 0.02):
            raise ValueError("Invalid rebalancing policy.")
        return self


@dataclass
class PortfolioResult:
    value: np.ndarray          # (P, T+1) pot value at month end, after flows
    sleeves: np.ndarray        # (P, T+1, 4) sleeve values after flows
    target: np.ndarray         # (P, T+1, 4) target weights used
    nav: np.ndarray            # (P, T+1) time-weighted NAV index (starts at 1)
    contributions: np.ndarray  # (P, T+1) money paid in at each month (t = 0: initial)
    tax_paid: np.ndarray       # (P, T+1) box 3 tax withdrawn
    costs_paid: np.ndarray     # (P, T+1) transaction costs
    rebalanced: np.ndarray     # (P, T+1) bool
    locked: np.ndarray         # (P,) month at which lock-in fired (-1 = never)


TargetFn = Callable[[int, np.ndarray, np.ndarray], np.ndarray]
"""target(t, pot_value (P,), regime (P,)) -> (P, 4) weights."""


def run_portfolio(
    returns: dict[str, np.ndarray],
    pe: ProtectedEquitySleeve | None,
    target_fn: TargetFn,
    regimes: np.ndarray,
    initial: float,
    contributions: np.ndarray,
    box3: Box3Params,
    rebalance: RebalancePolicy,
    box3_category: tuple[str, ...] = ("other", "other", "other", "other"),
) -> PortfolioResult:
    """Simulate the pot.

    returns: simple monthly returns (P, T) for 'equity', 'short_bonds', 'cash'.
    pe: protected-equity sleeve object (handles its own valuation).
    contributions: (P, T+1); column 0 is ignored (use `initial`).
    box3_category: per sleeve, 'bank' or 'other' (deemed-return category).
    """
    rebalance.validate()
    P, T = returns["cash"].shape
    sl = np.zeros((P, T + 1, len(SLEEVES)))
    tgt = np.zeros((P, T + 1, len(SLEEVES)))
    nav = np.ones((P, T + 1))
    contrib = np.zeros((P, T + 1))
    tax = np.zeros((P, T + 1))
    costs = np.zeros((P, T + 1))
    reb = np.zeros((P, T + 1), bool)

    w0 = target_fn(0, np.full(P, float(initial)), regimes[:, 0])
    tgt[:, 0] = w0
    sl[:, 0] = initial * w0
    contrib[:, 0] = initial
    if pe is not None:
        pe.set_value(sl[:, 0, PE])
    bank_mask = np.array([c == "bank" for c in box3_category])

    for t in range(1, T + 1):
        prev_total = sl[:, t - 1].sum(axis=1)
        cur = sl[:, t - 1].copy()
        for name in ("equity", "short_bonds", "cash"):
            k = SLEEVES.index(name)
            cur[:, k] *= 1.0 + returns[name][:, t - 1]
        if pe is not None:
            cur[:, PE] = pe.advance(t)
        total = cur.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            nav[:, t] = nav[:, t - 1] * np.where(prev_total > 0, total / np.where(prev_total > 0, prev_total, 1.0), 1.0)

        if box3.enabled and t % 12 == 0:
            bank = cur[:, bank_mask].sum(axis=1)
            other = cur[:, ~bank_mask].sum(axis=1)
            due = np.minimum(box3_tax(bank, other, box3, years_elapsed=t / 12), total)
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = np.where(total > 0, due / np.where(total > 0, total, 1.0), 0.0)
            cur *= (1.0 - frac)[:, None]
            tax[:, t] = due
            total = cur.sum(axis=1)

        w_t = target_fn(t, total, regimes[:, t])
        tgt[:, t] = w_t
        c = contributions[:, t]
        contrib[:, t] = c
        cur += c[:, None] * w_t
        total = cur.sum(axis=1)

        with np.errstate(invalid="ignore", divide="ignore"):
            w_cur = np.where(total[:, None] > 0, cur / np.where(total > 0, total, 1.0)[:, None], w_t)
        drift = np.abs(w_cur - w_t).sum(axis=1)
        do = (t % rebalance.every_months == 0) | (drift > rebalance.drift_threshold_l1)
        if np.any(do):
            new = total[:, None] * w_t
            traded = np.abs(new - cur).sum(axis=1)
            cost = np.where(do, rebalance.transaction_cost * traded, 0.0)
            new = new * np.where(total > 0, 1.0 - cost / np.where(total > 0, total, 1.0), 0.0)[:, None]
            cur = np.where(do[:, None], new, cur)
            costs[:, t] = cost
            reb[:, t] = do
        if pe is not None:
            pe.set_value(cur[:, PE])
        sl[:, t] = cur

    return PortfolioResult(
        value=sl.sum(axis=2), sleeves=sl, target=tgt, nav=nav, contributions=contrib,
        tax_paid=tax, costs_paid=costs, rebalanced=reb, locked=np.full(P, -1),
    )
