"""German rules: § 32a EStG tariff, net income, costs, and the fund / interest taxation against hand calculations."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from homepiggy.affordability_de import (FundTax, affordability_gap_de, income_tax_2026, load_policy_de, net_income,
                                        transfer_tax_rate)
from homepiggy.affordability_nl import BuyerProfile
from homepiggy.engine import RunConfig, SaverProfile, run
from homepiggy.glidepath import GlidepathSpec
from homepiggy.sleeves import ProductParams, cash_returns, equity_returns

from .conftest import hand_market

POL = load_policy_de(2026)
RATE = 0.25 * 1.055


def test_tariff_formula_and_continuity():
    # zone 3 by hand: z = (30,000 - 17,799) / 10,000
    z = (30000 - 17799) / 10000
    assert income_tax_2026(np.array([30000.0]), POL)[0] == np.floor((173.10 * z + 2397) * z + 1034.87)
    assert income_tax_2026(np.array([12348.0]), POL)[0] == 0.0
    assert income_tax_2026(np.array([100000.0]), POL)[0] == np.floor(0.42 * 100000 - 11135.63)
    # the tariff is continuous at the zone boundaries (within 1 euro of rounding)
    for b in (12348, 17799, 69878, 277825):
        lo, hi = income_tax_2026(np.array([b, b + 1.0]), POL)
        assert 0 <= hi - lo <= 1.0


def test_net_income_plausible_and_monotone():
    g = np.array([30000.0, 60000.0, 120000.0])
    net = net_income(g, 20.0, np.ones(3), POL)
    assert np.all(np.diff(net) > 0) and np.all(net < g)
    assert 0.55 < net[1] / g[1] < 0.70          # single, no children, 2026
    # childless surcharge from 23 lowers net pay slightly
    assert net_income(g[1:2], 30.0, np.ones(1), POL)[0] < net[1]


def test_costs_by_state_and_income_switch():
    b = replace(BuyerProfile(), region="NW")
    a = affordability_gap_de(np.array([400000.0]), np.array([60000.0]), np.array([0.04]), 30.0, np.ones(1), b, POL,
                             apply_income_test=False)
    fees = (0.010 * 1.19 + 0.005) * 400000
    assert a.transfer_tax[0] == pytest.approx(0.065 * 400000)
    assert a.nhg_premium[0] == pytest.approx(fees)
    assert a.required_deposit[0] == pytest.approx(0.065 * 400000 + fees)        # 100% LTV: costs only
    on = affordability_gap_de(np.array([400000.0]), np.array([60000.0]), np.array([0.04]), 30.0, np.ones(1), b, POL)
    assert on.income_loan[0] == pytest.approx(0.35 * on.flp[0] / 0.06)
    assert transfer_tax_rate(POL, "") == 0.035                                   # default Bavaria
    with pytest.raises(ValueError):
        transfer_tax_rate(POL, "XX")


def _de_cfg(market, table, initial, monthly=0.0, years=3):
    saver = SaverProfile(horizon_years=years, initial_eur=initial, monthly_contribution_eur=monthly,
                         real_wage_growth=0.0)
    return RunConfig(market=market, saver=saver, glidepath=GlidepathSpec(kind="step", table=table), n_paths=2,
                     country="de")


def test_fund_tax_lump_sum_matches_hand_calculation(market):
    """100% equity, one lump sum, deterministic index: Vorabpauschale each year and the exit tax by hand."""
    T = 36
    eq = np.cumprod(np.r_[1.0, np.full(T, 1.006)])[None, :].repeat(2, axis=0)
    m = hand_market(eq, rate=0.02)
    table = [{"from_years": 15, "to_years": 0, "weights": [1.0, 0.0, 0.0, 0.0]}]
    r = run(_de_cfg(market, table, 1_000_000.0), market_paths=m)
    rets = equity_returns(m, ProductParams())[0]
    ft = FundTax.from_policy(POL, "fund")
    v, basis, vp_cum = 1_000_000.0, 1_000_000.0, 0.0
    for y in range(3):
        v0 = v
        for t in range(12 * y, 12 * y + 12):
            v *= 1 + rets[t]
        if y < 2:
            vp = min(0.7 * 0.032 * v0, max(v - v0, 0.0))
            tax = RATE * max(vp * 0.7 - 1000, 0.0)
            frac = tax / v
            v -= tax; basis *= 1 - frac; vp_cum = (vp_cum + vp) * (1 - frac)
    exit_tax = RATE * max((v - basis - vp_cum) * 0.7 - 1000, 0.0)
    assert r.exit_tax[0] == pytest.approx(exit_tax, rel=1e-10)
    assert r.portfolio.value[0, -1] == pytest.approx(v - exit_tax, rel=1e-10)


def test_deemed_sale_on_category_change(market):
    """Equity fund (30% exemption) becomes a cash fund (0%) one year before purchase: the gain to that date keeps
    the 30% exemption (§ 22 InvStG); later gains have none."""
    T = 36
    eq = np.cumprod(np.r_[1.0, np.full(T, 1.006)])[None, :].repeat(2, axis=0)
    m = hand_market(eq, rate=0.02)
    table = [{"from_years": 15, "to_years": 1, "weights": [1.0, 0.0, 0.0, 0.0]},
             {"from_years": 1, "to_years": 0, "weights": [0.0, 0.0, 0.0, 1.0]}]
    r = run(_de_cfg(market, table, 1_000_000.0), market_paths=m)
    st = r.portfolio.fund_tax_state
    assert st["tf"][0] == 0.0 and st["seg"][0] > 0
    V = r.portfolio.value[0, -1] + r.exit_tax[0]
    assert r.exit_tax[0] == pytest.approx(RATE * max(st["seg"][0] + (V - st["basis"][0] - st["vp_cum"][0]) - 1000, 0))


def test_savings_interest_taxed_every_year(market):
    T = 24
    m = hand_market(np.ones((2, T + 1)), rate=0.04)
    table = [{"from_years": 15, "to_years": 0, "weights": [1.0, 0.0, 0.0, 0.0]}]
    r = run(_de_cfg(market, table, 200_000.0, years=2), market_paths=m)
    rc = cash_returns(m, 0.005)[0]
    b = 200_000.0
    for y in range(2):
        b0 = b
        for t in range(12 * y, 12 * y + 12):
            b *= 1 + rc[t]
        b -= RATE * max(b - b0 - 1000, 0.0)
    assert r.benchmark.value[0, -1] == pytest.approx(b, rel=1e-12)


def test_policy_sections_sourced():
    for name, d in POL.items():
        if name == "meta":
            continue
        assert any("source" in k for k in d), name
        assert any(k.endswith("verification") for k in d), name
