"""Dutch box 3 wealth tax, 2026 deemed-return system.

Method (docs/METHODS.md, section 6.2). The statutory steps, in order:
  1. rendementsgrondslag G = bank + other assets - (debts above threshold)
  2. deemed return    R = bank * f_bank + other * f_other - debts * f_debt
  3. effective %      p = R / G
  4. grondslag sparen en beleggen B = max(G - allowance, 0)
  5. voordeel         V = p * B
  6. tax              = rate * max(V, 0)

Model simplifications (declared):
  * The peildatum (1 January) is approximated by each 12-month anniversary of
    the start date. With a 1 January start date this is exact.
  * Tax is deducted from the pot at the peildatum, not with next year's
    assessment, which is slightly conservative.
  * The tegenbewijsregeling (taxing the actual return when it is lower) is NOT
    modelled, so the tax is an upper bound on what a saver would pay.
  * The allowance is held at its 2026 nominal value unless an indexation
    rate is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class Box3Params:
    tax_rate: float
    allowance: float
    f_bank: float
    f_other: float
    f_debt: float = 0.0
    other_box3_assets: float = 0.0          # saver's assets outside the fund (assumed 'other')
    other_box3_bank: float = 0.0
    allowance_indexation: float = 0.0       # annual nominal indexation of the allowance (assumption)
    enabled: bool = True

    @classmethod
    def from_policy(cls, policy: Mapping[str, Any], partners: bool = False, **kw: Any) -> "Box3Params":
        b = policy["box3"]
        return cls(
            tax_rate=float(b["tax_rate"]),
            allowance=float(b["allowance_partners_eur"] if partners else b["allowance_single_eur"]),
            f_bank=float(b["deemed_return"]["bank_deposits"]),
            f_other=float(b["deemed_return"]["other_assets"]),
            f_debt=float(b["deemed_return"]["debts"]),
            **kw,
        )


def box3_tax(bank: np.ndarray, other: np.ndarray, p: Box3Params, years_elapsed: float = 0.0) -> np.ndarray:
    """Tax due for the given box 3 assets. Arrays are per path; result is >= 0."""
    if not p.enabled:
        return np.zeros_like(np.asarray(bank, float))
    bank = np.asarray(bank, float) + p.other_box3_bank
    other = np.asarray(other, float) + p.other_box3_assets
    G = bank + other
    R = bank * p.f_bank + other * p.f_other
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(G > 0, R / np.where(G > 0, G, 1.0), 0.0)
    allowance = p.allowance * (1.0 + p.allowance_indexation) ** years_elapsed
    B = np.maximum(G - allowance, 0.0)
    total = p.tax_rate * np.maximum(pct * B, 0.0)
    # Attribute the tax to the fund pro rata to its share of box 3 assets.
    fund = np.asarray(bank, float) - p.other_box3_bank + np.asarray(other, float) - p.other_box3_assets
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(G > 0, fund / np.where(G > 0, G, 1.0), 0.0)
    return total * share
