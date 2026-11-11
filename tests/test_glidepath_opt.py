import numpy as np
import pytest

from homepiggy.engine import RunConfig
from homepiggy.glidepath import DEFAULT_TABLE, GlidepathSpec, static_weights
from homepiggy.glidepath_opt import (METRICS, Relay, analyse, candidates, choose, cvar_low, evaluate)


def test_candidate_count_and_validity():
    c = candidates()
    assert len(c) == 106 and len({x.label for x in c}) == 106
    for r in c:
        W = static_weights(r.spec(), 120)
        np.testing.assert_allclose(W.sum(axis=1), 1.0)
        assert np.all(np.diff(W[:, 0]) <= 1e-12)          # equity never increases toward purchase


def test_relay_phases():
    W = static_weights(Relay(6, 3, 1, 0.0).spec(), 120)
    assert W[0].tolist() == [1, 0, 0, 0]                  # 10y out: equity
    assert W[120 - 5 * 12].tolist() == [0, 1, 0, 0]       # 5y out: protected equity
    assert W[120 - 2 * 12].tolist() == [0, 0, 1, 0]       # 2y out: short bonds
    assert W[120].tolist() == [0, 0, 0, 1]                # purchase: cash
    Wb = static_weights(Relay(6, 3, 0, 0.25).spec(), 120)
    np.testing.assert_allclose(Wb[120], [0, 0.25, 0.75, 0])   # no cash phase, soft handoff


def test_cvar_low():
    x = np.arange(1, 101, dtype=float)
    assert cvar_low(x, 0.10) == pytest.approx(5.5)        # mean of 1..10
    assert cvar_low(np.array([3.0]), 0.10) == 3.0


def test_choose_respects_bar_under_all_methods():
    def mk(growth, bar):
        return {"rows": [{"growth": g, "passes_safety_bar": b} for g, b in zip(growth, bar)]}
    sel = {"repo": mk([1.0, 2.0, 3.0, 9.0], [True, True, True, True]),
           "hmm": mk([1.0, 2.0, 3.0, 9.0], [True, True, False, True])}
    assert choose(sel) == 1          # row 2 fails under hmm; rows 0 and 3 are references (today, bank)
    sel["hmm"]["rows"][1]["passes_safety_bar"] = False
    assert choose(sel) is None


def test_evaluate_smoke(market):
    specs = [GlidepathSpec(kind="step", table=DEFAULT_TABLE, name="today"), Relay(6, 3, 1, 0.0).spec()]
    res = evaluate(RunConfig(market=market), specs, [market] * 2, n_paths=100, n_jobs=1)
    assert res.shape == (2, 3, len(METRICS))
    bank = res[0, -1]
    assert bank[METRICS.index("p_beat_bank")] == 0.0
    out = analyse(res, ["today", "relay", "bank"], np.arange(2))
    assert out["rows"][0]["growth_minus_today_median"] == 0.0
    assert out["rows"][2]["safety_minus_bank_median"] == 0.0 and out["rows"][2]["passes_safety_bar"]
