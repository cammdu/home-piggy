"""NL affordability gap: what a first-time buyer must bring in cash.

Method (docs/METHODS.md, section 5). Statutory rules come from
data/nl/policy_2026.yaml, which cites a source for every number.

  toetsrente   i_test = m if fixed period >= 10y else max(m, AFM test rate 5%)
  FLP          financing-cost % from the 2026 Nibud/regulation table, looked up
               by (toetsinkomen bracket, toetsrente bracket); toetsrente is
               rounded to 3 decimals in % as the regulation prescribes
  budget       monthly = income * FLP / 12 - DUO instalment * bruteringsfactor(i_test)
  income loan  L_inc = budget * (1 - (1 + i/12)^-360) / (i/12)   (30y annuity at i_test)
               + EUR 17,000 for singles with toetsinkomen > EUR 30,000
               + energy-label amount
  LTV cap      L = min(L_inc, 100% * price)
  costs        transfer tax: 0 if starter exemption applies (18 <= age < 35,
               price <= limit, first use, owner-occupied), else 2% of price;
               NHG premium 0.4% of the loan if the loan <= NHG limit (optional);
               other buyer costs (notary, valuation, advice) supplied by the user
  gap          D = max(price + costs - L, 0)

Projection to the purchase date (declared assumption): the 2026 rules are
frozen in *real* terms. Euro thresholds (table income brackets, singles
threshold and amount, energy-label amounts) are scaled by simulated CPI, and
the starter and NHG price limits by the simulated house price index (they
are raised in line with house prices: 525k -> 555k in 2025 -> 2026).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib import resources
from typing import Any, Mapping

import numpy as np
import yaml

RATE_UPPER_PCT = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5])


def load_policy(year: int = 2026) -> dict[str, Any]:
    with resources.files("homepiggy.data.nl").joinpath(f"policy_{year}.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class FinancingCostTable:
    income_rows: np.ndarray   # (R,) lower bounds; row 0 (=0) covers incomes below the first bracket
    pct: np.ndarray           # (R, 12) financing-cost percentages as fractions

    @classmethod
    def load(cls, policy: Mapping[str, Any]) -> "FinancingCostTable":
        fname = policy["mortgage_income_test"]["financing_cost_table"]["file"]
        with resources.files("homepiggy.data.nl").joinpath(fname).open("r", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        header, body = rows[0], rows[1:]
        if len(header) != 13:
            raise ValueError("Financing-cost table must have 1 income + 12 rate columns.")
        inc = np.array([float(r[0]) for r in body])
        pct = np.array([[float(x) / 100.0 for x in r[1:]] for r in body])
        if np.any(np.diff(inc) <= 0):
            raise ValueError("Income rows must be strictly increasing.")
        if np.any((pct <= 0) | (pct >= 1)):
            raise ValueError("Financing-cost percentages out of range.")
        return cls(inc, pct)

    def lookup(self, income: np.ndarray, test_rate: np.ndarray) -> np.ndarray:
        """FLP for each (income, test rate). Income in EUR (2026 terms), rate as a decimal."""
        income = np.asarray(income, float)
        r_pct = np.round(np.asarray(test_rate, float) * 100.0, 3)
        col = np.searchsorted(RATE_UPPER_PCT, r_pct, side="left")      # <=1.500 -> 0, 1.501 -> 1, >6.500 -> 11
        row = np.searchsorted(self.income_rows, income, side="right") - 1
        row = np.clip(row, 0, len(self.income_rows) - 1)                # above top row -> top row (assumption)
        return self.pct[row, col]


def annuity_factor(monthly_rate: np.ndarray, n: int) -> np.ndarray:
    """Present value of 1 per month for n months; the limit n as the rate -> 0."""
    i = np.asarray(monthly_rate, float)
    small = np.abs(i) < 1e-10
    safe = np.where(small, 1.0, i)
    return np.where(small, float(n), (1.0 - (1.0 + safe) ** (-n)) / safe)


def annuity_payment(principal: np.ndarray, annual_rate: np.ndarray | float, n: int) -> np.ndarray:
    return np.asarray(principal, float) / annuity_factor(np.asarray(annual_rate, float) / 12.0, n)


def grossup_factor(policy: Mapping[str, Any], test_rate: np.ndarray) -> np.ndarray:
    g = policy["mortgage_income_test"]["student_loan_grossup"]
    upper = np.array([float(x) for x in g["brackets_upper_pct"]])
    fac = np.array([float(x) for x in g["factors"]])
    r_pct = np.round(np.asarray(test_rate, float) * 100.0, 3)
    return fac[np.searchsorted(upper, r_pct, side="left")]


@dataclass(frozen=True)
class BuyerProfile:
    """Saver/buyer inputs. These are the user's scenario inputs, not market data."""

    age_at_start: float = 20.0
    single: bool = True
    first_time_buyer: bool = True
    owner_occupier: bool = True
    fixed_period_years: int = 10
    energy_label: str = "EFG"            # key into policy energy_label_extra_loan_eur.table
    use_nhg: bool = True
    other_costs_eur: float = 0.0         # notary, valuation, advice: user input (2026 euros, CPI-indexed)


@dataclass
class AffordabilityResult:
    test_rate: np.ndarray
    flp: np.ndarray
    income_loan: np.ndarray
    max_loan: np.ndarray
    transfer_tax: np.ndarray
    nhg_premium: np.ndarray
    other_costs: np.ndarray
    required_deposit: np.ndarray        # max(price + costs - loan, 0)


def affordability_gap(
    price: np.ndarray,
    income: np.ndarray,
    mortgage_rate: np.ndarray,
    student_instalment: np.ndarray,
    age: float,
    cpi_factor: np.ndarray,
    house_factor: np.ndarray,
    buyer: BuyerProfile,
    policy: Mapping[str, Any],
    table: FinancingCostTable,
) -> AffordabilityResult:
    """Vectorised affordability gap at a purchase date.

    cpi_factor / house_factor: simulated index level relative to the 2026 base
    (used to keep 2026 euro thresholds constant in real terms).
    """
    mi = policy["mortgage_income_test"]
    price = np.asarray(price, float)
    income = np.asarray(income, float)
    m = np.asarray(mortgage_rate, float)

    tr = mi["test_rate"]
    if buyer.fixed_period_years >= tr["short_fixed_period_threshold_years"]:
        test = m
    else:
        test = np.maximum(m, tr["short_fixed_period_minimum_rate"])

    income_2026 = income / cpi_factor
    flp = table.lookup(income_2026, test)
    budget = income * flp / 12.0 - np.asarray(student_instalment, float) * grossup_factor(policy, test)
    budget = np.maximum(budget, 0.0)
    loan = budget * annuity_factor(test / 12.0, int(mi["annuity_term_months"]))

    sp = mi["single_person_extra_loan"]
    if buyer.single:
        eligible = income_2026 > sp["min_toetsinkomen_eur"]
        loan = loan + np.where(eligible, sp["amount_eur"] * cpi_factor, 0.0)
    label_table = mi["energy_label_extra_loan_eur"]["table"]
    if buyer.energy_label not in label_table:
        raise ValueError(f"Unknown energy label key '{buyer.energy_label}'; choose from {list(label_table)}.")
    loan = loan + label_table[buyer.energy_label] * cpi_factor

    max_loan = np.minimum(loan, policy["loan_to_value"]["max_ltv"] * price)

    tt = policy["transfer_tax"]
    se = tt["starters_exemption"]
    limit = se["house_value_limit_eur"] * house_factor
    exempt = (
        (se["min_age"] <= age) & (age < se["max_age_exclusive"]) & buyer.first_time_buyer
        & buyer.owner_occupier & (price <= limit)
    )
    rate = tt["owner_occupier_rate"] if buyer.owner_occupier else tt["non_owner_occupier_rate"]
    transfer_tax = np.where(exempt, 0.0, rate * price)

    nhg = policy["nhg"]
    if buyer.use_nhg:
        nhg_ok = max_loan <= nhg["cost_limit_eur"] * house_factor
        nhg_premium = np.where(nhg_ok, nhg["one_off_premium_rate"] * max_loan, 0.0)
    else:
        nhg_premium = np.zeros_like(price)
    other = buyer.other_costs_eur * np.asarray(cpi_factor, float) * np.ones_like(price)
    gap = np.maximum(price + transfer_tax + nhg_premium + other - max_loan, 0.0)
    return AffordabilityResult(test, flp, loan, max_loan, transfer_tax, nhg_premium, other, gap)
