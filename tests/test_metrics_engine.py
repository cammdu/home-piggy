import copy
from dataclasses import replace

import numpy as np
import pytest

from homepiggy import metrics as M
from homepiggy.config import ConfigError, load_market_params, market_params_from_dict
from homepiggy.engine import RunConfig, SaverProfile, compare, paired_difference, run
from homepiggy.glidepath import GlidepathSpec
from homepiggy.portfolio import RebalancePolicy
from homepiggy.regimes import RegimeSegment
from homepiggy.scenarios import Scenario, scenario_from_dict
from homepiggy.market import ShockEvent
from homepiggy.sleeves import ProductParams

from .conftest import SYNTH, deterministic_dict

ALL_CASH = GlidepathSpec(table=[{"from_years": 15, "to_years": 0, "weights": [0, 0, 0, 1]}], name="all_cash")


# ----------------------------------------------------------------- metrics
def test_wilson_known_values():
    e = M.wilson(5, 10)
    assert e.lo == pytest.approx(0.2366, abs=1e-4) and e.hi == pytest.approx(0.7634, abs=1e-4)
    assert M.wilson(0, 10).hi == pytest.approx(0.2775, abs=1e-4)


def test_quantile_interval_coverage():
    rng = np.random.default_rng(0)
    hits = sum(
        (lambda e: e.lo <= 0.0 <= e.hi)(M.quantile(rng.standard_normal(301), 0.5)) for _ in range(1000)
    )
    assert 0.93 <= hits / 1000 <= 0.98       # nominal 95% (discrete order statistics are slightly conservative)


def test_irr_recovers_known_rate():
    i = 0.005
    T = 60
    flows = -np.full((1, T + 1), 100.0)
    flows[0, -1] = 0.0
    fv = sum(100 * (1 + i) ** (T - t) for t in range(T))
    flows[0, -1] += fv
    assert M.irr_monthly(flows)[0] == pytest.approx(i, abs=1e-12)


def test_max_drawdown():
    assert M.max_drawdown(np.array([[1.0, 1.2, 0.9, 1.3, 1.0]]))[0] == pytest.approx(0.9 / 1.2 - 1)


# ----------------------------------------------------------------- config
def test_config_fails_closed(synth_dict):
    d = copy.deepcopy(synth_dict)
    del d["equity"]["sigma"]
    with pytest.raises(ConfigError):
        market_params_from_dict(d)
    d = copy.deepcopy(synth_dict)
    d["correlation"][0][1] = 0.5        # asymmetric
    with pytest.raises(ConfigError):
        market_params_from_dict(d)


def test_unsourced_parameter_file_rejected(tmp_path):
    txt = SYNTH.read_text().replace("synthetic: true", "synthetic: false")
    p = tmp_path / "x.yaml"
    p.write_text(txt)
    with pytest.raises(ConfigError, match="sources"):
        load_market_params(p)


# ----------------------------------------------------------------- engine
@pytest.fixture(scope="module")
def mp():
    return load_market_params(SYNTH)


def _det_cfg(synth_dict, rate=0.03, **kw):
    p = market_params_from_dict(deterministic_dict(synth_dict, rate=rate))
    saver = SaverProfile(horizon_years=5, initial_eur=1000, monthly_contribution_eur=100, real_wage_growth=0.0)
    return RunConfig(market=p, saver=saver, n_paths=4, box3_enabled=False,
                     rebalance=RebalancePolicy(transaction_cost=0.0),
                     product=ProductParams(cash_spread=0.0), benchmark_savings_spread=0.0, **kw)


def test_conservation_all_cash_closed_form(synth_dict):
    r = run(_det_cfg(synth_dict, glidepath=ALL_CASH))
    i, T = 0.03 / 12, 60
    fv = 1000 * (1 + i) ** T + 100 * sum((1 + i) ** (T - t) for t in range(1, T + 1))
    np.testing.assert_allclose(r.portfolio.value[:, -1], fv, rtol=1e-12)
    np.testing.assert_allclose(r.benchmark.value[:, -1], fv, rtol=1e-12)


def test_weights_valid_and_sleeves_nonnegative(mp):
    r = run(RunConfig(market=mp, n_paths=300))
    assert np.all(r.portfolio.sleeves >= -1e-9)
    np.testing.assert_allclose(r.portfolio.target.sum(axis=2), 1.0)
    assert np.all(np.isfinite(r.portfolio.value))


def test_determinism_and_seed_sensitivity(mp):
    a = run(RunConfig(market=mp, n_paths=200, seed=1)).portfolio.value
    b = run(RunConfig(market=mp, n_paths=200, seed=1)).portfolio.value
    c = run(RunConfig(market=mp, n_paths=200, seed=2)).portfolio.value
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_compare_uses_common_random_numbers(mp):
    res = compare(RunConfig(market=mp, n_paths=200),
                  [GlidepathSpec(), GlidepathSpec(kind="time_warp", gamma=0.5, name="late")])
    a, b = res["step"], res["late"]
    np.testing.assert_array_equal(a.market.house_index, b.market.house_index)
    np.testing.assert_array_equal(a.required_path, b.required_path)
    d = paired_difference(b, a)
    assert 0 <= d["p_a_higher_pot"]["value"] <= 1


def test_forced_recession_scenario(mp):
    sc = Scenario(name="deep", segments=(RegimeSegment(0, 120, "Recession"),))
    r = run(RunConfig(market=mp, n_paths=100, scenario=sc))
    assert r.summary["regime_share"]["Recession"] == 1.0


def test_purchase_anchored_scenario_resolves():
    sc = scenario_from_dict({"name": "late shock", "anchor": "purchase", "sources": ["test"],
                             "segments": [{"start": 0, "end": 24, "regime": "Recession"}],
                             "shocks": [{"month": 0, "duration_months": 6, "equity_log": -0.3}]}).resolve(120)
    assert sc.segments[0].start == 96 and sc.segments[0].end == 120
    assert sc.shocks[0].month == 114


def test_scenario_shocks_require_sources():
    with pytest.raises(ValueError, match="sourced"):
        scenario_from_dict({"name": "x", "shocks": [{"month": 1, "equity_log": -0.2}]})


def test_shock_in_final_years_hurts_less_with_glidepath(mp):
    shock = Scenario(name="crash", shocks=(ShockEvent(month=110, duration_months=3, equity_log=np.log(0.6)),),
                     sources=("test",))
    base = compare(RunConfig(market=mp, n_paths=300),
                   [GlidepathSpec(), GlidepathSpec(table=[{"from_years": 10, "to_years": 0, "weights": [1, 0, 0, 0]}], name="all_equity")])
    hit = compare(RunConfig(market=mp, n_paths=300, scenario=shock),
                  [GlidepathSpec(), GlidepathSpec(table=[{"from_years": 10, "to_years": 0, "weights": [1, 0, 0, 0]}], name="all_equity")])
    loss_glide = 1 - hit["step"].portfolio.value[:, -1].mean() / base["step"].portfolio.value[:, -1].mean()
    loss_equity = 1 - hit["all_equity"].portfolio.value[:, -1].mean() / base["all_equity"].portfolio.value[:, -1].mean()
    assert loss_glide < 0.02 < 0.25 < loss_equity      # 80% cash + 20% bonds is near-immune to a late crash


def test_lock_in_fires_when_target_met(mp):
    r = run(RunConfig(market=mp, n_paths=100, glidepath=GlidepathSpec(kind="lock_in", lock_multiple=0.0)))
    assert np.all(r.portfolio.locked == 1)
    np.testing.assert_allclose(r.portfolio.target[:, 1:], np.broadcast_to([0, 0, 0.2, 0.8], (100, 120, 4)))


def test_target_vol_runs_and_derisks(mp):
    res = compare(RunConfig(market=mp, n_paths=200),
                  [GlidepathSpec(), GlidepathSpec(kind="target_vol", vol_start=0.05, vol_end=0.01)])
    eq_step = res["step"].portfolio.target[:, :, 0].mean()
    eq_tv = res["target_vol"].portfolio.target[:, :, 0].mean()
    assert eq_tv < eq_step


def test_mc_error_shrinks_with_paths(mp):
    w = []
    for n in (400, 1600):
        e = run(RunConfig(market=mp, n_paths=n)).summary["p_outpace_cash"]
        w.append(e["hi"] - e["lo"])
    assert w[1] < 0.6 * w[0]          # ~1/sqrt(n): expect about 0.5x
