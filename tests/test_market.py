import numpy as np
import pytest

from homepiggy.config import STATE_VARS, market_params_from_dict
from homepiggy.market import ShockEvent, nearest_correlation, simulate_market

from .conftest import deterministic_dict


def _single_regime_run(params, k=0, n=20000, T=24, seed=3, shocks=()):
    reg = np.full((n, T + 1), k, dtype=np.int8)
    normals = np.random.default_rng(seed).standard_normal((n, T, len(STATE_VARS)))
    return simulate_market(params, reg, normals, shocks), normals, reg


def test_ar1_deterministic_half_life(synth_dict):
    d = deterministic_dict(synth_dict)
    d["short_rate"]["initial"] = 0.05
    d["short_rate"]["theta"] = [0.01] * 4
    d["short_rate"]["half_life_months"] = 6
    p = market_params_from_dict(d)
    m, _, _ = _single_regime_run(p, n=2, T=12)
    dev = m.short_rate[0] - 0.01
    np.testing.assert_allclose(dev[6] / dev[0], 0.5, rtol=1e-12)
    np.testing.assert_allclose(dev[12] / dev[0], 0.25, rtol=1e-12)


def test_equity_moments_match_parameters(market):
    m, _, _ = _single_regime_run(market, k=2)
    x = m.equity_logret.ravel()
    mu, sd = market.equity.mu[2] / 12, market.equity.sigma[2] / np.sqrt(12)
    se_mean = sd / np.sqrt(x.size)
    assert abs(x.mean() - mu) < 4 * se_mean
    assert abs(x.std() / sd - 1) < 4 / np.sqrt(2 * x.size)


def test_innovation_correlation_matches_target(market):
    """Sample correlation of equity vs mortgage innovations ~ target (Fisher-z 4 SE)."""
    m, _, _ = _single_regime_run(market, k=0)
    eq = m.equity_logret.ravel()
    dm = np.diff(m.mortgage_rate, axis=1)  # increments = kappa-term + innovation
    kappa = 1 - 0.5 ** (1 / market.levels["mortgage_rate"].half_life_months)
    innov = (dm - kappa * (market.levels["mortgage_rate"].theta[0] - m.mortgage_rate[:, :-1])).ravel()
    r = np.corrcoef(eq, innov)[0, 1]
    target = nearest_correlation(market.corr[0])[0, STATE_VARS.index("mortgage_rate")]
    assert abs(np.arctanh(r) - np.arctanh(target)) < 4 / np.sqrt(eq.size - 3)


def test_house_growth_autocorrelation(synth_dict):
    p = market_params_from_dict(synth_dict)
    m, _, _ = _single_regime_run(p, k=0, n=4000, T=120)
    g = np.diff(np.log(m.house_index), axis=1)[:, 20:]          # drop burn-in
    g = g - g.mean()
    rho = (g[:, 1:] * g[:, :-1]).sum() / (g * g).sum()
    assert abs(rho - p.house_ar1) < 0.02


def test_shocks_are_exact_and_paired(market):
    shock = ShockEvent(month=5, duration_months=3, equity_log=np.log(0.7), house_log=np.log(0.8), mortgage_rate=0.01)
    base, _, _ = _single_regime_run(market, n=50, T=24)
    shk, _, _ = _single_regime_run(market, n=50, T=24, shocks=[shock])
    np.testing.assert_allclose(shk.equity_index[:, -1] / base.equity_index[:, -1], 0.7, rtol=1e-12)
    np.testing.assert_allclose(shk.house_index[:, -1] / base.house_index[:, -1], 0.8, rtol=1e-12)
    np.testing.assert_allclose(shk.equity_index[:, :6], base.equity_index[:, :6])     # untouched before
    # +100 bp spread over months 5-7, then mean-reverting: just under 100 bp at level index 8
    kappa = 1 - 0.5 ** (1 / market.levels["mortgage_rate"].half_life_months)
    expected = 0.01 / 3 * (1 + (1 - kappa) + (1 - kappa) ** 2)
    np.testing.assert_allclose(shk.mortgage_rate[:, 8] - base.mortgage_rate[:, 8], expected, rtol=1e-9)


def test_shock_outside_horizon_rejected(market):
    with pytest.raises(ValueError):
        _single_regime_run(market, n=2, T=12, shocks=[ShockEvent(month=11, duration_months=2)])


def test_nearest_correlation_repairs_non_psd():
    C = np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1.0]])
    A = nearest_correlation(C)
    assert np.linalg.eigvalsh(A).min() > 0
    np.testing.assert_allclose(np.diag(A), 1.0)
    np.linalg.cholesky(A)


def test_cpi_compounds_inflation(synth_dict):
    p = market_params_from_dict(deterministic_dict(synth_dict, infl=0.024))
    m, _, _ = _single_regime_run(p, n=2, T=12)
    np.testing.assert_allclose(m.cpi[0, -1], (1 + 0.024 / 12) ** 12, rtol=1e-12)
