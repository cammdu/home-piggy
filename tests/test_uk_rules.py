"""UK (England) rules: SDLT against GOV.UK's worked figures, Lifetime ISA flows and charge, engine wiring."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from homepiggy.affordability_nl import BuyerProfile
from homepiggy.affordability_uk import affordability_gap_uk, lisa_charge, lisa_flows, load_policy_uk, sdlt
from homepiggy.engine import RunConfig, SaverProfile, run

POL = load_policy_uk(2026)


@pytest.mark.parametrize("price, ftb, expected", [
    (300_000, True, 0.0),            # first-time buyer: nil band to 300k
    (400_000, True, 5_000.0),        # 5% of 100k
    (500_000, True, 10_000.0),       # GOV.UK-quoted figure for a 500k first-time purchase
    (500_001, True, 15_000.05),      # relief lost above 500k: standard rates on the whole price
    (300_000, False, 5_000.0),       # standard: 2% of 125k + 5% of 50k
    (125_000, False, 0.0),
    (1_000_000, False, 43_750.0),    # 2,500 + 33,750 (5% of 675k) + 10% of 75k
    (2_000_000, False, 153_750.0),   # 2,500 + 33,750 + 57,500 + 12% of 500k
])
def test_sdlt(price, ftb, expected):
    assert float(sdlt(np.array([price]), POL, ftb)[0]) == pytest.approx(expected, abs=1e-6)


def test_affordability_income_test_switch():
    b = BuyerProfile()
    price, income = np.array([400_000.0]), np.array([50_000.0])
    off = affordability_gap_uk(price, income, np.array([0.05]), np.array([1.0]), b, POL, apply_income_test=False)
    on = affordability_gap_uk(price, income, np.array([0.05]), np.array([1.0]), b, POL, apply_income_test=True)
    assert off.max_loan[0] == pytest.approx(0.95 * 400_000)
    assert off.required_deposit[0] == pytest.approx(0.05 * 400_000 + 5_000)          # LTV gap + SDLT
    assert on.max_loan[0] == pytest.approx(4.5 * 50_000)
    assert on.required_deposit[0] == pytest.approx(400_000 + 5_000 - 225_000)


def test_lisa_flows_and_age_limit():
    c = np.full((2, 13), 460.0); c[:, 0] = 0.0
    pay, bonus = lisa_flows(c, 20.0, POL)
    assert np.allclose(pay[:, 1:], 4000 / 12)
    assert pay[0, 0] == 0 and bonus[0, 1] == 0
    assert np.allclose(bonus[:, 2:], 0.25 * 4000 / 12)
    small = np.full((1, 13), 100.0); small[:, 0] = 0
    assert np.allclose(lisa_flows(small, 20.0, POL)[0][:, 1:], 100.0)        # below the monthly limit
    assert not lisa_flows(c, 40.0, POL)[0].any()                              # too old to open
    assert not lisa_flows(c, 20.0, POL, enabled=False)[1].any()


def test_lisa_charge_threshold():
    v = np.array([10_000.0, 10_000.0])
    assert np.allclose(lisa_charge(v, np.array([450_000.0, 450_001.0]), POL), [0.0, 2_000.0])


def _uk_cfg(market, **kw):
    saver = SaverProfile(horizon_years=8, monthly_contribution_eur=460.0, gross_income_eur=50_000.0,
                         house_price_eur=335_000.0, initial_eur=834.0)
    return RunConfig(market=market, saver=saver, n_paths=300, seed=11, country="uk", **kw)


def test_engine_uk_wiring(market):
    on = run(_uk_cfg(market, lisa=True))
    off = run(_uk_cfg(market, lisa=False))
    # same own payments, no tax; the bonus only adds money (before any charge)
    assert np.allclose(on.portfolio.contributions, off.portfolio.contributions)
    assert on.portfolio.tax_paid.sum() == 0.0
    gross_on = on.portfolio.value[:, -1] + on.lisa_charge
    assert np.all(gross_on > off.portfolio.value[:, -1])
    # charge only where the price at purchase is above 450k, and equal to 20% of the LISA slice
    above = on.price[:, -1] > 450_000
    assert np.all(on.lisa_charge[~above] == 0.0)
    share = on.portfolio.tracked_share
    assert np.allclose(on.lisa_charge[above], 0.2 * share[above] * gross_on[above])
    assert np.all((share > 0) & (share < 1))
    # the deposit target uses SDLT, not Dutch transfer tax
    t = on.target_path[:, -1]
    expect = 0.10 * on.price[:, -1] + sdlt(on.price[:, -1], POL)
    assert np.allclose(t, np.maximum(expect, 0.05 * on.price[:, -1] + sdlt(on.price[:, -1], POL)))


def test_tracked_share_exact_without_costs(market):
    """Cash-only fund (no trades, no costs): the Lifetime ISA slice equals its own accumulation at the cash rate."""
    from homepiggy.engine import simulate_market_paths
    from homepiggy.glidepath import GlidepathSpec
    from homepiggy.sleeves import cash_returns
    cash = GlidepathSpec(kind="step", table=[{"from_years": 15, "to_years": 0, "weights": [0, 0, 0, 1]}], name="cash")
    cfg = replace(_uk_cfg(market, lisa=True), glidepath=cash)
    r = run(cfg)
    m = simulate_market_paths(cfg)
    rc = cash_returns(m, cfg.product.cash_spread)                        # (P, T)
    T = m.n_months
    c = np.zeros((m.n_paths, T + 1))
    from homepiggy.engine import _income_index
    c[:, 1:] = cfg.saver.monthly_contribution_eur * _income_index(m, cfg.saver.real_wage_growth)[:, 1:]
    pay, bonus = lisa_flows(c, cfg.saver.buyer.age_at_start, POL)
    v = np.zeros(m.n_paths)
    for t in range(1, T + 1):
        v = v * (1 + rc[:, t - 1]) + pay[:, t] + bonus[:, t]
    gross = r.portfolio.value[:, -1] + r.lisa_charge
    assert np.allclose(r.portfolio.tracked_share * gross, v, rtol=1e-10)


def test_uk_policy_sections_all_sourced():
    """Every section of the UK policy file carries a verification level and a source (or is an assumption)."""
    levels = ("primary", "secondary", "assumption")
    for name, d in POL.items():
        if name == "meta":
            continue
        ver = [v for k, v in d.items() if k.endswith("verification")]
        assert ver and all(v in levels for v in ver), name
        assert any("source" in k for k in d), name
