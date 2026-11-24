"""UK (England) affordability and Lifetime ISA rules.

Method (docs/METHODS.md, section 12). Statutory numbers come from
data/uk/policy_2026.yaml, which cites a GOV.UK / FCA source for each.

  income loan  L_inc = loan-to-income limit (4.5) x gross income  (FPC flow limit; FCA FG26/11)
  LTV cap      L = min(L_inc, 95% x price)   or 95% x price when the income test is off
  SDLT         first-time buyer relief if price <= 500,000: 0% to 300,000, 5% on 300,001-500,000;
               otherwise the standard bands (0 / 2 / 5 / 10 / 12%) on the whole price
  costs        SDLT + other buyer costs supplied by the user (conveyancing, survey; CPI-indexed)
  gap          D = max(price + costs - L, 0)

Projection to the purchase date (declared assumption): the pound thresholds (SDLT bands,
the 500,000 relief cap, the Lifetime ISA's 4,000 limit and 450,000 threshold) stay fixed in
nominal pounds, as they are in statute; nothing is indexed. This differs from the NL module,
where thresholds are scaled because Dutch law indexes them.

Lifetime ISA (when enabled; saver aged 18-39 when the account is opened):
  each month min(payment, 4,000 / 12) goes into the Lifetime ISA, the rest into a Stocks and
  Shares ISA in the same fund; a 25% bonus on the Lifetime ISA payment is credited one month later.
  At purchase, if the price exceeds 450,000 a 20% withdrawal charge applies to the Lifetime ISA
  value (GOV.UK, 2026). Payments stop at 50 (not reached within the horizons modelled).
"""
from __future__ import annotations

from importlib import resources
from typing import Any, Mapping

import numpy as np
import yaml

from .affordability_nl import AffordabilityResult, BuyerProfile


def load_policy_uk(year: int = 2026) -> dict[str, Any]:
    with resources.files("homepiggy.data.uk").joinpath(f"policy_{year}.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def sdlt(price: np.ndarray, policy: Mapping[str, Any], first_time_buyer: bool = True) -> np.ndarray:
    """Stamp Duty Land Tax on a single residential purchase (England / NI), slice by slice."""
    p = np.asarray(price, float)
    s = policy["stamp_duty_land_tax"]
    up = np.array([float(x) for x in s["standard_bands"]["upper_gbp"]])
    rt = np.array([float(x) for x in s["standard_bands"]["rates"]])
    lo = np.r_[0.0, up[:-1]]
    standard = sum(r * np.clip(p - a, 0.0, b - a) for a, b, r in zip(lo, up, rt))
    if not first_time_buyer:
        return standard
    f = s["first_time_buyer_relief"]
    relief = f["reduced_rate"] * np.clip(p - f["nil_rate_upper_gbp"], 0.0, f["reduced_rate_upper_gbp"] - f["nil_rate_upper_gbp"])
    return np.where(p <= f["max_price_gbp"], relief, standard)


def affordability_gap_uk(
    price: np.ndarray,
    income: np.ndarray,
    mortgage_rate: np.ndarray,
    cpi_factor: np.ndarray,
    buyer: BuyerProfile,
    policy: Mapping[str, Any],
    apply_income_test: bool = True,
) -> AffordabilityResult:
    """Vectorised UK affordability gap at a purchase date (same result fields as the NL module).

    transfer_tax holds SDLT; nhg_premium is zero (no NHG equivalent); flp holds the LTI limit.
    """
    price = np.asarray(price, float)
    income = np.asarray(income, float)
    lend = policy["lending"]
    lti = float(lend["loan_to_income_limit"])
    income_loan = lti * income
    ltv_loan = float(lend["max_ltv"]) * price
    max_loan = np.minimum(income_loan, ltv_loan) if apply_income_test else ltv_loan
    tax = sdlt(price, policy, buyer.first_time_buyer)
    other = buyer.other_costs_eur * np.asarray(cpi_factor, float) * np.ones_like(price)
    gap = np.maximum(price + tax + other - max_loan, 0.0)
    zero = np.zeros_like(price)
    return AffordabilityResult(np.asarray(mortgage_rate, float) * np.ones_like(price), np.full_like(price, lti),
                               income_loan, max_loan, tax, zero, other, gap)


def lisa_flows(contributions: np.ndarray, age_at_start: float, policy: Mapping[str, Any],
               enabled: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(Lifetime ISA payment, bonus credited) per path and month, shape like ``contributions`` (P, T+1).

    Column 0 (the initial amount) goes to the Stocks and Shares ISA. Payment in month t is
    min(contribution, annual limit / 12); the 25% bonus on it is credited in month t + 1.
    Disabled (zeros) if switched off or the saver is not aged 18-39 at the start.
    """
    c = np.asarray(contributions, float)
    pay, bonus = np.zeros_like(c), np.zeros_like(c)
    L = policy["lifetime_isa"]
    if not enabled or not (L["open_age_min"] <= age_at_start < L["open_age_max_exclusive"]):
        return pay, bonus
    T = c.shape[1] - 1
    ages = age_at_start + np.arange(T + 1) / 12.0
    pay[:, 1:] = np.minimum(c[:, 1:], L["annual_limit_gbp"] / 12.0)
    pay[:, ages >= L["contribute_until_age_exclusive"]] = 0.0
    bonus[:, 2:] = L["bonus_rate"] * pay[:, 1:-1]
    return pay, bonus


def lisa_charge(lisa_value: np.ndarray, price: np.ndarray, policy: Mapping[str, Any]) -> np.ndarray:
    """Withdrawal charge at purchase: 20% of the Lifetime ISA value if the price is above 450,000."""
    L = policy["lifetime_isa"]
    above = np.asarray(price, float) > L["property_price_threshold_gbp"]
    return np.where(above, L["withdrawal_charge_above_threshold"] * np.asarray(lisa_value, float), 0.0)
