import numpy as np
import pytest

from homepiggy.glidepath import GlidepathSpec, static_weights, target_vol_lambda


def test_default_table_matches_agreed_glidepath():
    W = static_weights(GlidepathSpec(), 120)
    tau = lambda years: 120 - int(years * 12)                       # noqa: E731  month index for years-to-purchase
    np.testing.assert_allclose(W[tau(10)], [0.80, 0.20, 0.00, 0.00])
    np.testing.assert_allclose(W[tau(8)], [0.80, 0.20, 0.00, 0.00])
    np.testing.assert_allclose(W[tau(5)], [0.30, 0.60, 0.10, 0.00])
    np.testing.assert_allclose(W[tau(3)], [0.00, 0.20, 0.30, 0.50])  # 4-2y: 50% cash
    np.testing.assert_allclose(W[tau(1)], [0.00, 0.00, 0.20, 0.80])  # 2-0y: 80% cash
    np.testing.assert_allclose(W[-1], [0.00, 0.00, 0.20, 0.80])
    np.testing.assert_allclose(W.sum(axis=1), 1.0)


def test_eight_year_horizon_starts_in_first_segment():
    W = static_weights(GlidepathSpec(), 96)
    np.testing.assert_allclose(W[0], [0.80, 0.20, 0.0, 0.0])


def test_time_warp_gamma_one_equals_step_and_late_derisk():
    step = static_weights(GlidepathSpec(), 120)
    np.testing.assert_allclose(static_weights(GlidepathSpec(kind="time_warp", gamma=1.0), 120), step)
    late = static_weights(GlidepathSpec(kind="time_warp", gamma=0.6), 120)
    assert late[:, 0].sum() > step[:, 0].sum()                       # more equity-months when de-risking late


def test_linear_is_between_neighbours():
    W = static_weights(GlidepathSpec(kind="linear"), 120)
    assert np.all(W >= -1e-12) and np.allclose(W.sum(axis=1), 1.0)
    assert np.all(np.diff(W[:, 0]) <= 1e-12)                          # equity never increases


@pytest.mark.parametrize("table", [
    [{"from_years": 10, "to_years": 5, "weights": [1, 0, 0, 0]}, {"from_years": 4, "to_years": 0, "weights": [0, 0, 0, 1]}],
    [{"from_years": 10, "to_years": 0, "weights": [0.5, 0.5, 0.1, 0]}],
    [{"from_years": 10, "to_years": 1, "weights": [1, 0, 0, 0]}],
    [{"from_years": 6, "to_years": 0, "weights": [1, 0, 0, 0]}],      # shorter than horizon
])
def test_invalid_tables_rejected(table):
    with pytest.raises(ValueError):
        static_weights(GlidepathSpec(table=table), 120)


def test_target_vol_lambda():
    cov = np.diag([0.2**2, 0.1**2, 0.02**2, 0.0])[None].repeat(2, axis=0)
    wb = np.array([[1.0, 0, 0, 0], [0.1, 0, 0, 0.9]])
    lam = target_vol_lambda(wb, cov, 0.10)
    assert abs(lam[0] - 0.5) < 1e-9          # 0.5 * 20% = 10%
    assert lam[1] == 1.0                     # base mix vol 2% < target: never lever up
