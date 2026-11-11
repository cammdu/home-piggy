import numpy as np
import pytest

from homepiggy.engine import RunConfig, run
from homepiggy.frontier import (BASELINE, Candidate, analyse, evaluate, grid, outcome_metrics, pareto_mask)
from homepiggy.glidepath import DEFAULT_TABLE, GlidepathSpec, static_weights


def test_baseline_candidate_reproduces_agreed_glidepath():
    a = static_weights(Candidate(*BASELINE).spec(), 120)
    b = static_weights(GlidepathSpec(kind="step", table=DEFAULT_TABLE), 120)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_grid_is_monotone_and_contains_baseline():
    g = grid()
    assert all(c.c_late >= c.c_mid for c in g)
    assert Candidate(*BASELINE) in g
    assert len(g) == 4 * 66
    assert len({c.label for c in g}) == len(g)


@pytest.mark.parametrize("start", [6.0, 5.0, 3.0])
def test_candidate_tables_are_valid(start):
    w = static_weights(Candidate(start, 0.3, 0.9).spec(), 120)
    np.testing.assert_allclose(w.sum(axis=1), 1.0)
    months_mid = np.sum(np.isclose(w[:, 3], 0.3))
    assert months_mid == pytest.approx((start - 2) * 12, abs=1)


def test_pareto_mask():
    g = np.array([0.5, 0.6, 0.4, 0.6])
    s = np.array([-10, -20, -30, -20])
    assert pareto_mask(g, s).tolist() == [True, True, False, True]


def test_outcome_metrics():
    V = np.arange(100.0); G = np.full(100, 50.0)
    m = outcome_metrics(V, G, G)
    assert m["p_target"] == 0.5 and m["bad_case"] == pytest.approx(np.quantile(V - 50, 0.05))


def test_evaluate_matches_engine_and_analyse(market):
    cands = [Candidate(*BASELINE), Candidate(4.0, 1.0, 1.0), Candidate(4.0, 0.0, 0.0)]
    cfg = RunConfig(market=market)
    res = evaluate(cfg, cands, [market] * 4, n_paths=200, n_jobs=1)
    assert res.shape == (4, 3, 5)
    r = run(cfg.__class__(market=market, n_paths=200, glidepath=cands[0].spec()))
    ref = outcome_metrics(r.portfolio.value[:, -1], r.target_path[:, -1], r.required_path[:, -1],
                         r.benchmark.value[:, -1])
    assert res[0, 0, 0] == pytest.approx(ref["p_target"]) and res[0, 0, 1] == pytest.approx(ref["bad_case"])
    assert res[0, 0, 4] == pytest.approx(ref["bad_vs_bank"])
    out = analyse(res, cands, safety="bad_vs_bank")
    assert out["rows"][0]["is_baseline"] and out["rows"][0]["dg_median_test"] == 0.0
