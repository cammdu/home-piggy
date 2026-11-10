import numpy as np
import pytest

from homepiggy.regimes import (
    RegimeSegment, empirical_transition_matrix, expected_durations_months, simulate_regimes,
    stationary_distribution, validate_transition_matrix,
)


@pytest.mark.parametrize("bad", [
    [[0.5, 0.5]],                      # not square
    [[1.2, -0.2], [0.5, 0.5]],         # negative
    [[0.5, 0.4], [0.5, 0.5]],          # row does not sum to 1
    [[np.nan, 1.0], [0.5, 0.5]],       # non-finite
])
def test_validate_rejects_bad_matrices(bad):
    with pytest.raises(ValueError):
        validate_transition_matrix(np.array(bad))


def test_stationary_two_state_closed_form():
    a, b = 0.1, 0.3
    P = np.array([[1 - a, a], [b, 1 - b]])
    pi = stationary_distribution(P)
    np.testing.assert_allclose(pi, [b / (a + b), a / (a + b)], atol=1e-12)
    np.testing.assert_allclose(pi @ P, pi, atol=1e-12)


def test_expected_durations():
    P = np.array([[0.9, 0.1], [0.25, 0.75]])
    np.testing.assert_allclose(expected_durations_months(P), [10.0, 4.0])


def test_empirical_transitions_match_within_sampling_error(market):
    P = market.P
    n, T = 20000, 120
    rng = np.random.default_rng(1)
    init = rng.integers(0, 4, n)
    paths = simulate_regimes(P, n, T, init, rng.random((n, T)))
    P_hat = empirical_transition_matrix(paths, 4)
    counts = np.zeros(4)
    np.add.at(counts, paths[:, :-1].ravel(), 1)
    se = np.sqrt(P * (1 - P) / counts[:, None])         # binomial standard error per cell
    assert np.all(np.abs(P_hat - P) <= 4 * se + 1e-12)


def test_forcing_and_common_random_numbers(market):
    n, T = 500, 60
    rng = np.random.default_rng(2)
    init = np.zeros(n, int)
    u = rng.random((n, T))
    free = simulate_regimes(market.P, n, T, init, u)
    forced = simulate_regimes(market.P, n, T, init, u, [RegimeSegment(24, 36, "Recession")], market.regimes)
    assert np.all(forced[:, 25:37] == market.regimes.index("Recession"))
    np.testing.assert_array_equal(free[:, :25], forced[:, :25])   # identical before the forced window


def test_forcing_outside_horizon_rejected(market):
    with pytest.raises(ValueError):
        simulate_regimes(market.P, 10, 12, np.zeros(10, int), np.random.default_rng(0).random((10, 12)),
                         [RegimeSegment(6, 13, "Recession")], market.regimes)
