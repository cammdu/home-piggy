import numpy as np
import pytest

from homepiggy.options import bs_price, implied_vol, solve_call_strike
from homepiggy.sleeves import ProductParams, ProtectedEquitySleeve, bond_returns, cash_returns

from .conftest import hand_market

R_SIMPLE_5PCT_CONT = np.expm1(0.05)   # simple rate whose continuous equivalent is 5%


def test_black_scholes_textbook_value():
    # S=K=100, r=5% (cont.), sigma=20%, T=1: call = 10.4506, put = 5.5735 (Hull)
    c = bs_price(100.0, 100.0, 1.0, R_SIMPLE_5PCT_CONT, 0.2, "call")
    p = bs_price(100.0, 100.0, 1.0, R_SIMPLE_5PCT_CONT, 0.2, "put")
    assert abs(c - 10.4506) < 1e-4 and abs(p - 5.5735) < 1e-4


@pytest.mark.parametrize("K", [60.0, 90.0, 100.0, 130.0])
def test_put_call_parity(K):
    S, tau, r, v = 100.0, 0.75, 0.03, 0.25
    c = bs_price(S, K, tau, r, v, "call")
    p = bs_price(S, K, tau, r, v, "put")
    assert abs((c - p) - (S - K * np.exp(-np.log1p(r) * tau))) < 1e-10


def test_expiry_is_intrinsic():
    assert bs_price(90.0, 100.0, 0.0, 0.02, 0.3, "put") == 10.0
    assert bs_price(110.0, 100.0, 0.0, 0.02, 0.3, "call") == 10.0


def test_smile_puts_richer_than_atm():
    assert implied_vol(0.2, 0.85, 0.3, 0.05) > implied_vol(0.2, 1.0, 0.3, 0.05) > implied_vol(0.2, 1.2, 0.3, 0.05)


def test_call_strike_solver_hits_target():
    S = np.array([100.0, 100.0])
    target = np.array([2.0, 4.0])
    k = solve_call_strike(S, target, 1.0, np.array([0.02, 0.02]), np.array([0.18, 0.18]), 0.3, 0.05)
    c = bs_price(S, S * k, 1.0, 0.02, implied_vol(np.array([0.18, 0.18]), k, 0.3, 0.05), "call")
    np.testing.assert_allclose(c, target, atol=1e-8)
    assert k[0] > k[1] > 1.0   # cheaper target -> higher cap


def _buffer_tranche_value(final_ratio, prod, market):
    """One tranche, 12-month outcome, path from 100 to 100*final_ratio; return V_T / V_0."""
    T = 12
    path = np.linspace(1.0, final_ratio, T + 1)[None, :] * 100.0
    m = hand_market(path, rate=0.02)
    pe = ProtectedEquitySleeve(m, market, prod)
    pe.set_value(np.array([1000.0]))
    for t in range(1, T + 1):
        v = pe.advance(t)
    return v[0] / 1000.0, pe


@pytest.mark.parametrize("final", [0.95, 0.90, 0.80, 0.60, 1.20])
def test_buffer_payoff_at_expiry(final, market):
    prod = ProductParams(n_tranches=1, protected_equity_ter=0.0, option_cost=0.0, buffer_lower_strike=0.90)
    ratio, _ = _buffer_tranche_value(final, prod, market)
    # Units bought at t=0: n = V0 / (S0 + put spread). Payoff per unit at expiry:
    # S_T + max(K1 - S_T, 0) - max(K2 - S_T, 0)
    S0, ST = 100.0, 100.0 * final
    pe0 = ProtectedEquitySleeve(hand_market(np.full((1, 13), 100.0), rate=0.02), market, prod)
    unit_cost = pe0._unit_values(0)[0, 0]
    payoff = ST + max(100.0 - ST, 0) - max(90.0 - ST, 0)
    assert abs(ratio - payoff / unit_cost) < 1e-9
    if final < 1.0:
        # downside: index loss of (1 - final) is reduced by exactly the 10% buffer (or fully absorbed)
        expected_loss_on_notional = max(1.0 - final - 0.10, 0.0)
        assert abs(payoff / S0 - (1.0 - expected_loss_on_notional)) < 1e-12


def test_zero_cost_collar_caps_upside(market):
    prod = ProductParams(n_tranches=1, protected_equity_ter=0.0, option_cost=0.0, call_funding=1.0)
    ratio_up, pe = _buffer_tranche_value(1.50, prod, market)
    pe0 = ProtectedEquitySleeve(hand_market(np.full((1, 13), 100.0), rate=0.02), market, prod)
    assert abs(pe0._unit_values(0)[0, 0] - 100.0) < 1e-6        # zero-cost: unit costs exactly S
    cap = pe0.Kc[0, 0] / 100.0
    assert 1.0 < cap < 1.5
    assert abs(ratio_up - cap) < 1e-9                            # gain capped at the call strike


def test_ladder_staggered_and_value_consistent(market):
    prod = ProductParams()
    path = 100.0 * np.exp(np.cumsum(np.r_[0, np.full(36, 0.004)]))[None, :]
    m = hand_market(np.repeat(path, 3, axis=0))
    pe = ProtectedEquitySleeve(m, market, prod)
    np.testing.assert_array_equal(pe.expiry, [3, 6, 9, 12])
    pe.set_value(np.array([0.0, 500.0, 1000.0]))
    np.testing.assert_allclose(pe.value(), [0.0, 500.0, 1000.0])
    for t in range(1, 37):
        v = pe.advance(t)
        assert np.all(v >= 0)
        pe.set_value(v * 1.01)
        np.testing.assert_allclose(pe.value(), v * 1.01, rtol=1e-12)
    assert v[0] == 0.0


def test_bond_and_cash_returns_closed_form():
    m = hand_market(np.full((1, 4), 100.0), rate=0.03)
    m.term_spread[:] = 0.0
    prod = ProductParams(short_bonds_ter=0.0)
    np.testing.assert_allclose(bond_returns(m, prod), 0.03 / 12)        # flat curve: carry only
    m.term_spread[0, 2:] = 0.01                                          # 2y yield +100 bp at month 2
    r = bond_returns(m, prod)[0]
    y1_w = (1 - 0.25) / 1.75
    slope_after = 0.01 * (1 - y1_w)
    assert abs(r[1] - (0.03 / 12 - prod.bond_duration * 0.01)) < 1e-12    # price hit = -D * dy
    assert abs(r[2] - (0.04 + prod.bond_duration * slope_after) / 12) < 1e-12   # carry + roll-down
    np.testing.assert_allclose(cash_returns(m, 0.001), (0.03 - 0.001) / 12)


def test_product_validation():
    with pytest.raises(ValueError):
        ProductParams(buffer_lower_strike=1.0).validate()
    with pytest.raises(ValueError):
        ProductParams(outcome_months=12, n_tranches=5).validate()
