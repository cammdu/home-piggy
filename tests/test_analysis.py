import numpy as np
import pytest

from homepiggy.analysis import band_study, path_detail, subset
from homepiggy.config import load_market_params
from homepiggy.engine import RunConfig
from homepiggy.glidepath import GlidepathSpec
from streamlit_app.presets import build_glidepaths, default_table_rows, table_from_rows

from .conftest import SYNTH
from .test_calibration import _perturbed_sets


def test_subset_is_deterministic_and_spread():
    xs = list(range(200))
    s = subset(xs, 50)
    assert s == subset(xs, 50) and len(s) == 50 and s[0] == 0 and s[-1] == 199
    assert subset(xs, 500) == xs


def test_parallel_equals_serial():
    sets = _perturbed_sets(4)
    gps = [GlidepathSpec(name="a"), GlidepathSpec(kind="time_warp", gamma=0.6, name="b")]
    cfg = RunConfig(market=sets[0])
    s1 = band_study(cfg, gps, sets, n_paths=200, n_jobs=1)
    s2 = band_study(cfg, gps, sets, n_paths=200, n_jobs=2)
    assert s1["bands"] == s2["bands"] and s1["paired_vs_first"] == s2["paired_vs_first"]
    assert 0 <= s1["median_replicate"] < 4


def test_path_detail_shapes_and_order():
    mp = load_market_params(SYNTH)
    d = path_detail(RunConfig(market=mp), [GlidepathSpec(name="a")], mp, n_paths=300)["a"]
    T = 120
    assert len(d["months"]) == T + 1 and len(d["weights_mean"]) == T + 1
    p = d["pot"]
    assert all(np.all(np.array(p[lo]) <= np.array(p[hi]) + 1e-9) for lo, hi in (("p05", "p25"), ("p25", "p50"), ("p50", "p75"), ("p75", "p95")))
    np.testing.assert_allclose(np.sum(d["weights_mean"], axis=1), 1.0)


def test_custom_table_round_trip_and_validation():
    rows = default_table_rows()
    t = table_from_rows(rows)
    assert t[2]["weights"] == pytest.approx([0.0, 0.2, 0.3, 0.5])
    bad = [dict(r) for r in rows]
    bad[0]["Cash"] = 50.0                         # row no longer sums to 100%
    gps = build_glidepaths(["Custom table"], bad)
    mp = load_market_params(SYNTH)
    with pytest.raises(ValueError):
        path_detail(RunConfig(market=mp), gps, mp, n_paths=50)
