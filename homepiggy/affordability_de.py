"""German affordability and investment-fund tax (2026 rules).

Method (docs/METHODS.md, section 13). Numbers come from data/de/policy_2026.yaml, each with a source.

Affordability at the purchase date
  costs        Grunderwerbsteuer (rate of the chosen Land) + notary (1.0% + 19% VAT) + land registry (0.5%)
               + other costs entered by the user. No first-time buyer exemption exists.
  loan limit   income test (optional, off by default as for NL/UK): monthly annuity
               (mortgage rate + 2% initial repayment) / 12 x loan <= 35% of monthly net income, so
               L_inc = 0.35 x net income / (rate + 0.02); LTV cap 100% of the price.
  net income   gross - income tax (§ 32a EStG 2026, single) - solidarity surcharge (SolZG) - employee social
               insurance (pension, unemployment, health incl. average additional rate, care incl. childless
               surcharge from 23), with the contribution ceilings. Taxable income = gross - 1,230 - 36 -
               pension share - 0.96 x health share - care share (simplified Vorsorgeaufwendungen).
  projection   euro amounts in the tax and social-insurance rules are scaled with simulated CPI (rules kept in
               real terms), like the NL module.

Fund tax (InvStG 2018 / EStG, private investor, no church tax): see `FundTax` and portfolio.run_portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any, Mapping

import numpy as np
import yaml

from .affordability_nl import AffordabilityResult, BuyerProfile


def load_policy_de(year: int = 2026) -> dict[str, Any]:
    with resources.files("homepiggy.data.de").joinpath(f"policy_{year}.yaml").open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def income_tax_2026(x: np.ndarray, policy: Mapping[str, Any]) -> np.ndarray:
    """§ 32a EStG tariff (single), x = taxable income in 2026 euros; floor to full euros as the law prescribes."""
    t = policy["income_tax"]
    x = np.floor(np.maximum(np.asarray(x, float), 0.0))
    z1, z2, z3, z4 = t["zones_upper_eur"]
    y = (x - z1) / 10000.0
    z = (x - z2) / 10000.0
    tax = np.where(x <= z1, 0.0,
          np.where(x <= z2, (t["zone2"]["a"] * y + t["zone2"]["b"]) * y,
          np.where(x <= z3, (t["zone3"]["a"] * z + t["zone3"]["b"]) * z + t["zone3"]["c"],
          np.where(x <= z4, t["zone4"]["rate"] * x - t["zone4"]["minus"], t["zone5"]["rate"] * x - t["zone5"]["minus"]))))
    return np.floor(tax)


def net_income(gross: np.ndarray, age: float, cpi_factor: np.ndarray, policy: Mapping[str, Any]) -> np.ndarray:
    """Annual net income of a single employee (gross in current euros; rules scaled by cpi_factor)."""
    g = np.asarray(gross, float)
    cf = np.asarray(cpi_factor, float)
    si = policy["social_insurance"]
    base_p = np.minimum(g, si["ceiling_pension_eur"] * cf)
    base_h = np.minimum(g, si["ceiling_health_eur"] * cf)
    pension = base_p * si["pension"]
    unemployment = base_p * si["unemployment"]
    health = base_h * (si["health"] + si["health_average_additional"])
    care = base_h * (si["care"] + (si["care_childless_surcharge"] if age >= 23 else 0.0))
    it = policy["income_tax"]
    zve = g - (it["employee_lump_sum_eur"] + it["special_expenses_lump_sum_eur"]) * cf - pension - 0.96 * health - care
    tax = income_tax_2026(zve / cf, policy) * cf
    so = policy["solidarity_surcharge"]
    lim = so["exemption_limit_single_eur"] * cf
    soli = np.where(tax <= lim, 0.0, np.minimum(so["rate"] * tax, so["phase_in_rate"] * (tax - lim)))
    return g - tax - soli - pension - unemployment - health - care


def transfer_tax_rate(policy: Mapping[str, Any], state: str) -> float:
    rates = policy["transfer_tax"]["rates"]
    s = state or policy["transfer_tax"]["default_state"]
    if s not in rates:
        raise ValueError(f"Unknown Land '{s}'; choose from {sorted(rates)}.")
    return float(rates[s])


def affordability_gap_de(price: np.ndarray, income: np.ndarray, mortgage_rate: np.ndarray, age: float,
                         cpi_factor: np.ndarray, buyer: BuyerProfile, policy: Mapping[str, Any],
                         apply_income_test: bool = True) -> AffordabilityResult:
    """Vectorised German affordability gap. Result fields as NL; `flp` holds annual net income,
    `transfer_tax` the Grunderwerbsteuer and `nhg_premium` the notary + land-registry fees."""
    price = np.asarray(price, float)
    m = np.asarray(mortgage_rate, float) * np.ones_like(price)
    net = net_income(np.asarray(income, float), age, cpi_factor, policy) * np.ones_like(price)
    L = policy["lending"]
    income_loan = L["payment_share_of_net_income"] * net / (m + L["initial_repayment_rate"])
    ltv_loan = L["max_ltv"] * price
    max_loan = np.minimum(income_loan, ltv_loan) if apply_income_test else ltv_loan
    tt = transfer_tax_rate(policy, buyer.region) * price
    n = policy["notary_land_registry"]
    fees = (n["notary_rate"] * (1.0 + n["notary_vat"]) + n["land_registry_rate"]) * price
    other = buyer.other_costs_eur * np.asarray(cpi_factor, float) * np.ones_like(price)
    gap = np.maximum(price + tt + fees + other - max_loan, 0.0)
    return AffordabilityResult(m, net, income_loan, max_loan, tt, fees, other, gap)


@dataclass(frozen=True)
class FundTax:
    """German taxation of the pot, applied inside portfolio.run_portfolio.

    mode 'fund' (the FirstHome fund, one investment fund whose mix follows the glidepath):
      * Vorabpauschale each 12-month period: min(70% x Basiszins x (value at start + this period's payments
        x (13 - m)/12 for a payment in month m), the period's gain), taxed after partial exemption;
      * partial exemption by the fund's equity share (equity + protected equity): >= 51% -> 30%,
        >= 25% -> 15%, else 0%; when the category changes the units count as sold and re-bought and the gain
        so far is taxed at sale with the old exemption (§ 22 InvStG);
      * at purchase all units are sold: (gain since the last deemed sale - Vorabpauschalen) x (1 - exemption)
        plus the deemed-sale gains, taxed after the saver allowance.
    mode 'interest' (savings account): the period's interest is taxed every 12 months and at purchase.
    Tax = 25% + 5.5% solidarity surcharge on the amount above the 1,000 allowance per 12-month period.
    Simplifications (declared): 12-month periods from the start stand in for calendar years; payments enter
    the Vorabpauschale base at their amount (not units x start-of-year price); tax is paid out of the pot,
    reducing cost basis and past Vorabpauschalen pro rata, and the small gain realised on units sold for tax
    is ignored.
    """
    mode: str
    rate: float
    allowance: float
    base_rate: float
    factor: float
    tf_equity: float
    tf_mixed: float
    th_equity: float
    th_mixed: float

    @classmethod
    def from_policy(cls, policy: Mapping[str, Any], mode: str = "fund") -> "FundTax":
        c, f = policy["capital_income_tax"], policy["investment_fund_tax"]
        pe, vp = f["partial_exemption"], f["vorabpauschale"]
        return cls(mode, c["rate"] * (1.0 + c["solidarity"]), float(c["saver_allowance_eur"]), vp["base_rate_2026"],
                   vp["factor"], pe["equity_fund"], pe["mixed_fund"], pe["equity_fund_threshold"],
                   pe["mixed_fund_threshold"])

    def exemption(self, weights: np.ndarray) -> np.ndarray:
        """Partial exemption from target weights (P, 4) [equity, protected equity, bonds, cash]."""
        eq = weights[:, 0] + weights[:, 1]
        return np.where(eq >= self.th_equity - 1e-12, self.tf_equity, np.where(eq >= self.th_mixed - 1e-12, self.tf_mixed, 0.0))

    def tax_on(self, taxable: np.ndarray) -> np.ndarray:
        return self.rate * np.maximum(np.asarray(taxable, float) - self.allowance, 0.0)
