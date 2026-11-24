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
    tracked_share: np.ndarray | None = None   # (P,) final share of the pot held in a tracked sub-account
    fund_tax_state: dict | None = None        # DE: basis, Vorabpauschalen, deemed-sale gains, exemption at the end


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
    tracked: np.ndarray | None = None,
    fund_tax=None,
) -> PortfolioResult:
    """Simulate the pot.

    returns: simple monthly returns (P, T) for 'equity', 'short_bonds', 'cash'.
    pe: protected-equity sleeve object (handles its own valuation).
    contributions: (P, T+1); column 0 is ignored (use `initial`).
    box3_category: per sleeve, 'bank' or 'other' (deemed-return category).
    tracked: optional (P, T+1) part of each contribution that belongs to a sub-account holding a
      pro-rata slice of every sleeve (UK Lifetime ISA). Returns, tax and costs hit it pro rata, so its
      share only changes when money comes in: share = (share * pot before + tracked) / (pot before + c).
    fund_tax: optional affordability_de.FundTax (German fund / savings-interest taxation, see its docstring).
      Hooks: deemed sale on an exemption change (before the month's payment), payment bookkeeping, and the
      12-month tax after the payment (before rebalancing). The exit tax at purchase is applied by the engine.
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
    share = np.zeros(P)
    ft = fund_tax
    if ft is not None:
        v_start = np.full(P, float(initial))
        basis = np.full(P, float(initial))
        vp_cum = np.zeros(P)
        seg = np.zeros(P)                      # taxable deemed-sale gains (after exemption), taxed at sale
        c_year = np.zeros(P)
        base_w = np.zeros(P)
        tf_cur = ft.exemption(np.asarray(w0, float).reshape(P, -1))

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
        if ft is not None and ft.mode == "fund":
            tf_new = ft.exemption(w_t)
            chg = tf_new != tf_cur
            if np.any(chg):                    # § 22 InvStG deemed sale at the category change
                seg = np.where(chg, seg + (total - basis - vp_cum) * (1.0 - tf_cur), seg)
                basis = np.where(chg, total, basis)
                vp_cum = np.where(chg, 0.0, vp_cum)
                tf_cur = tf_new
        if ft is not None:
            m_in_year = (t - 1) % 12 + 1
            basis = basis + c
            c_year = c_year + c
            base_w = base_w + c * (13 - m_in_year) / 12.0
        if tracked is not None:
            denom = total + c
            with np.errstate(invalid="ignore", divide="ignore"):
                share = np.where(denom > 0, (share * total + tracked[:, t]) / np.where(denom > 0, denom, 1.0), share)
        cur += c[:, None] * w_t
        total = cur.sum(axis=1)

        if ft is not None and t % 12 == 0 and (ft.mode == "interest" or t < T):
            gain = total - v_start - c_year
            if ft.mode == "fund":
                vp = np.minimum(ft.factor * ft.base_rate * (v_start + base_w), np.maximum(gain, 0.0))
                due = ft.tax_on(vp * (1.0 - tf_cur))
                vp_cum = vp_cum + vp
            else:
                due = ft.tax_on(gain)
            due = np.minimum(due, total)
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = np.where(total > 0, due / np.where(total > 0, total, 1.0), 0.0)
            cur *= (1.0 - frac)[:, None]
            basis *= (1.0 - frac); vp_cum *= (1.0 - frac); seg *= (1.0 - frac)
            tax[:, t] += due
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
        if ft is not None and t % 12 == 0:
            v_start = cur.sum(axis=1)
            c_year = np.zeros(P)
            base_w = np.zeros(P)

    return PortfolioResult(
        value=sl.sum(axis=2), sleeves=sl, target=tgt, nav=nav, contributions=contrib,
        tax_paid=tax, costs_paid=costs, rebalanced=reb, locked=np.full(P, -1),
        tracked_share=share if tracked is not None else None,
        fund_tax_state=({"basis": basis, "vp_cum": vp_cum, "seg": seg, "tf": tf_cur} if ft is not None else None),
    )
