"""Calibration methods: each estimator is checked on data with known truth (no network needed)."""
import numpy as np
import pytest

from calibration.bootstrap import politis_white_block_length, stationary_indices
from calibration.classify import (
    REGIMES, align_states, classify, cohens_kappa, expanding_z, fit_hmm,
)
from calibration.estimate import _house_kalman, _level_ar, estimate, ledoit_wolf_corr
from calibration.scenarios_hist import EVENTS, find_window
from calibration.sources import parse_cboe, parse_eurostat
from homepiggy.config import STATE_VARS, load_market_params, market_params_from_dict
from homepiggy.market import simulate_market
from homepiggy.regimes import simulate_regimes

from .conftest import SYNTH


# ------------------------------------------------------------------ classifier (repo semantics)
def test_expanding_z_has_no_look_ahead():
    x = np.r_[np.random.default_rng(0).standard_normal(60)]
    z1 = expanding_z(x)
    x2 = x.copy(); x2[40:] += 100.0                       # change the future
    z2 = expanding_z(x2)
    np.testing.assert_array_equal(z1[:40], z2[:40])
    assert np.all(np.isnan(z1[:23])) and not np.isnan(z1[23])   # 24 observations required


@pytest.mark.parametrize("scores, expected", [
    ([1.0, 0.5, 0.0, 0.0], "Expansion"),                  # repo table: Expansion meets all 4
    ([0.5, 1.5, 1.0, 0.0], "Late-cycle"),
    ([-1.0, 0.0, 0.0, 1.0], "Recession"),
    ([-0.5, -0.5, -0.5, 0.0], "Recovery"),
])
def test_classify_matches_repo_threshold_table(scores, expected):
    lab, p = classify(np.array([scores]))
    assert REGIMES[lab[0]] == expected
    assert np.isclose(p[0].sum(), 1.0)


def test_classify_tie_break_is_first_regime():
    # s_g = 0 satisfies both ">= 0" and "<= 0": construct an exact tie between Expansion and Recovery
    lab, p = classify(np.array([[0.0, 0.0, 0.0, 0.0]]))
    assert p[0][0] == p[0].max() and REGIMES[lab[0]] == "Expansion"


def test_hmm_recovers_separated_regimes():
    rng = np.random.default_rng(1)
    A = np.array([[0.95, 0.05], [0.05, 0.95]])
    s = [0]
    for _ in range(399):
        s.append(rng.choice(2, p=A[s[-1]]))
    s = np.array(s)
    X = np.where(s[:, None] == 0, -2.0, 2.0) + rng.standard_normal((400, 2))
    fit = fit_hmm(X, k=2, n_starts=5, seed=0)
    mapped, conf, _ = align_states(fit.states, s, k=2)
    assert (mapped == s).mean() > 0.97 and cohens_kappa(conf) > 0.9


# ------------------------------------------------------------------ estimators recover known parameters
def test_level_ar_recovers_kappa_and_theta():
    rng = np.random.default_rng(2)
    n, kappa, theta = 4000, 1 - 0.5 ** (1 / 12), np.array([0.01, 0.04])
    lab = (np.arange(n) // 200) % 2
    y = np.empty(n); y[0] = 0.02
    for t in range(1, n):
        y[t] = y[t - 1] + kappa * (theta[lab[t]] - y[t - 1]) + 0.001 * rng.standard_normal()
    vp = np.ones(n, bool); vp[0] = False
    k_hat, th_hat, _, hl, info = _level_ar(y, lab, vp, 2)
    assert abs(hl - 12) < 1.5
    np.testing.assert_allclose(th_hat, theta, atol=0.002)
    assert not any(info["theta_bound_binding"])


def test_level_ar_bounds_theta_to_observed_range():
    rng = np.random.default_rng(3)
    n = 300
    lab = (np.arange(n) // 30) % 2
    y = np.cumsum(0.001 * rng.standard_normal(n)) + 0.02 + 0.0005 * lab   # near random walk
    vp = np.ones(n, bool); vp[0] = False
    _, th, _, _, info = _level_ar(y, lab, vp, 2)
    lo, hi = info["range"]
    assert np.all((th >= lo - 1e-15) & (th <= hi + 1e-15))


def test_house_state_space_recovers_persistence_through_noise():
    rng = np.random.default_rng(4)
    n, phi, sig, se, mu = 600, 0.95, 0.002, 0.004, 0.004
    d = np.zeros(n); e = se * rng.standard_normal(n)
    for t in range(1, n):
        d[t] = phi * d[t - 1] + sig * rng.standard_normal()
    g = mu + d + e - np.r_[0.0, e[:-1]]
    lab = np.zeros(n, int); vp = np.ones(n, bool); vp[0] = False
    naive = np.corrcoef(g[1:], g[:-1])[0, 1]
    m_hat, p_hat, s_hat, se_hat, *_ = _house_kalman(g, lab, vp, 1)
    assert naive < 0.5                         # noise hides the persistence from a naive AR(1)
    assert abs(p_hat - phi) < 0.05
    assert abs(se_hat - se) / se < 0.2


def test_round_trip_engine_to_estimator():
    """Simulate with the engine's own model, then re-estimate: equity and rate parameters come back."""
    mp = load_market_params(SYNTH)
    n, T = 1, 6000
    rng = np.random.default_rng(5)
    reg = simulate_regimes(mp.P, n, T, np.zeros(n, int), rng.random((n, T)), (), mp.regimes)
    m = simulate_market(mp, reg, rng.standard_normal((n, T, len(STATE_VARS))))
    lab = reg[0, 1:].astype(int)
    rows = {"equity": m.equity_logret[0], "house": np.diff(np.log(m.house_index[0]))}
    for v in ("short_rate", "term_spread", "mortgage_rate", "inflation"):
        rows[v] = getattr(m, v)[0, 1:]
    vp = np.ones(T, bool); vp[0] = False
    est = estimate(rows, lab, vp, mp.regimes)
    np.testing.assert_allclose(est.params["equity"]["mu"], mp.equity.mu, atol=0.05)
    np.testing.assert_allclose(est.params["equity"]["sigma"], mp.equity.sigma, rtol=0.1)
    np.testing.assert_allclose(est.params["transition_matrix"], mp.P, atol=0.03)
    hl = est.params["short_rate"]["half_life_months"]
    assert abs(hl - mp.levels["short_rate"].half_life_months) / mp.levels["short_rate"].half_life_months < 0.35


# ------------------------------------------------------------------ shrinkage and bootstrap
def test_ledoit_wolf_limits():
    rng = np.random.default_rng(6)
    Z = rng.standard_normal((5000, 4))
    C, d = ledoit_wolf_corr(Z)
    assert 0 <= d <= 1 and np.allclose(C, np.eye(4), atol=0.05)
    L = np.linalg.cholesky(np.array([[1, 0.8], [0.8, 1]]))
    Z = rng.standard_normal((5000, 2)) @ L.T
    C, d = ledoit_wolf_corr(Z)
    assert d < 0.05 and abs(C[0, 1] - 0.8) < 0.03          # strong signal: little shrinkage
    C2, _ = ledoit_wolf_corr(Z[:20], target=np.array([[1, 0.8], [0.8, 1]]))
    assert np.allclose(np.diag(C2), 1.0)


def test_politis_white_orders_persistence():
    rng = np.random.default_rng(7)
    iid = rng.standard_normal(1000)
    ar = np.zeros(1000)
    for t in range(1, 1000):
        ar[t] = 0.9 * ar[t - 1] + rng.standard_normal()
    assert politis_white_block_length(iid) < 3 < politis_white_block_length(ar)


def test_stationary_indices_valid_mask():
    rng = np.random.default_rng(8)
    idx, valid = stationary_indices(100, 10.0, rng)
    assert not valid[0]
    j = np.where(valid)[0]
    assert np.all(idx[j] == idx[j - 1] + 1)                 # valid links are true successions
    assert valid.sum() > 50                                  # mean block 10 -> ~90% of links continue a block


def test_bootstrap_estimate_never_counts_block_joins():
    lab = np.array([0, 0, 1, 1, 0, 0])
    vp = np.array([False, True, False, True, False, True])   # three blocks: no observed 0->1 or 1->0
    from calibration.estimate import _transition
    C = _transition(lab, vp, 2)
    assert C[0, 1] == 0 and C[1, 0] == 0 and C[0, 0] == 2 and C[1, 1] == 1


# ------------------------------------------------------------------ scenarios and parsers
def test_find_window_drawdown_and_rise():
    months = np.arange(np.datetime64("2007-01", "M"), np.datetime64("2010-01", "M"))
    x = np.r_[np.linspace(1, 1.2, 10), np.linspace(1.2, 0.6, 15), np.linspace(0.6, 0.9, 11)]
    a, b = find_window(x, months, EVENTS[0])
    assert a == 9 and b == 24
    y = np.r_[np.zeros(20), np.linspace(0, 0.02, 12), np.full(4, 0.02)]
    from calibration.scenarios_hist import EventRule
    a, b = find_window(y, months, EventRule("r", "", "y", "max_rise_12m", "2007-01", "2009-12"))
    assert b - a == 12 and y[b] - y[a] == pytest.approx(0.02 * 11 / 11, abs=0.002)


def test_parsers_on_fixtures():
    d, v = parse_cboe(b"DATE,SKEW\n01/02/1990,126.09\n12/31/2025,140.5\n")
    assert list(d) == ["1990-01-02", "2025-12-31"] and v[1] == 140.5
    js = (b'{"size":[1,1,1,1,2],"dimension":{"time":{"category":{"index":{"2026-09":0,"2026-10":1}}}},'
          b'"value":{"0":3.0,"1":3.3}}')
    d, v = parse_eurostat(js)
    assert list(d) == ["2026-09", "2026-10"] and list(v) == [3.0, 3.3]


# ------------------------------------------------------------------ uncertainty bands and replay
def _perturbed_sets(n=6):
    import copy
    from homepiggy.config import load_yaml
    base = load_yaml(SYNTH)["market"]
    out = []
    for i in range(n):
        d = copy.deepcopy(base)
        d["house"]["mu"] = [m + 0.01 * (i - n / 2) for m in d["house"]["mu"]]
        out.append(market_params_from_dict(d))
    return out


def test_band_runner_orders_by_driver():
    from homepiggy.engine import RunConfig
    from homepiggy.uncertainty import run_band
    sets = _perturbed_sets()
    band = run_band(RunConfig(market=sets[0], n_paths=300), sets, n_paths=300)
    b = band["p_reach_deposit"]
    assert b["p05"] <= b["median"] <= b["p95"]
    # higher house-price growth -> larger required deposit -> lower success: driver is negative
    assert band["required_deposit.p50"]["p95"] > band["required_deposit.p50"]["p05"]
    assert b["drivers_spearman"].get("house_mu (10y occupancy avg)", -1) <= 0


def test_historical_scenario_replays_observed_move_exactly(synth_dict):
    from .conftest import deterministic_dict
    from homepiggy.engine import RunConfig, SaverProfile, simulate_market_paths
    from homepiggy.scenarios import historical_scenario
    p = market_params_from_dict(deterministic_dict(synth_dict))
    event = {"name": "x", "window": ["2008-01", "2009-01"], "duration_months": 12,
             "regime_path": ["Recession"] * 6 + ["Recovery"] * 6,
             "shocks": {"equity_log": -0.4, "house_log": -0.1, "short_rate": 0.0, "term_spread": 0.0,
                        "mortgage_rate": 0.0, "inflation": 0.0}}
    sc = historical_scenario(event, 120, end_months_before_purchase=0)
    assert sc.segments[0].start == 108 and sc.segments[-1].end == 120
    m = simulate_market_paths(RunConfig(market=p, n_paths=2, scenario=sc, saver=SaverProfile(horizon_years=10)))
    mu_e = np.asarray(p.equity.mu) / 12
    expected = -0.4 + 6 * mu_e[2] + 6 * mu_e[3]
    got = np.log(m.equity_index[0, 120] / m.equity_index[0, 108])
    assert got == pytest.approx(expected, abs=1e-12)
    with pytest.raises(ValueError):
        historical_scenario(event, 10, 0)


# ------------------------------------------------------------------ committed outputs
@pytest.mark.parametrize("suffix", ["", "_hmm"])
def test_committed_calibration_files_are_valid_and_sourced(suffix):
    import json
    from pathlib import Path
    from homepiggy.uncertainty import load_bootstrap
    root = Path(__file__).resolve().parents[1] / "params"
    mp = load_market_params(root / f"nl_calibrated{suffix}.yaml")
    assert not mp.provenance.get("synthetic") and len(mp.provenance["sources"]) >= 17
    assert all(s.get("sha256") for s in mp.provenance["sources"] if s["name"] != "eonia_estr_spread")
    reps, prov = load_bootstrap(root / f"nl_bootstrap{suffix}.json")
    assert len(reps) == 200 and prov["mean_block_length_months"] > 1
    scen = json.loads((root / "nl_scenarios.json").read_text())
    assert {e["name"] for e in scen["events"]["repo_classifier"]} >= {"gfc_2008", "rate_shock_2022", "nl_housing_slump"}


def test_house_mean_constrained_to_sample_mean():
    rng = np.random.default_rng(9)
    n = 400
    lab = (np.arange(n) // 40) % 3
    g = 0.003 + 0.001 * lab + 0.002 * rng.standard_normal(n)
    vp = np.ones(n, bool); vp[0] = False
    mu, *_ = _house_kalman(g, lab, vp, 3)
    freq = np.array([np.mean(lab == r) for r in range(3)])
    assert freq @ mu == pytest.approx(g.mean(), abs=1e-12)
