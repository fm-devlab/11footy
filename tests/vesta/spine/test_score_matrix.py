"""The spine's score matrix and its calibration, against reference numbers.

The reference values were computed on 24 October 2026 on the same inputs (this code gave a
difference of 0.0 on 900 matrices, and the same calibrations to every digit); they are frozen
here.
"""

import numpy as np
import pytest
from scipy.stats import poisson

from vesta.forecast_reading import read_line
from vesta.spine.calibration import fit_calibration
from vesta.spine.score_matrix import (
    MAXIMUM_GOALS,
    Calibration,
    calibrate_intensities,
    find_score_probability,
    format_published_matrix,
    score_matrices,
    sum_result_probabilities,
)

REFERENCE_CALIBRATION = Calibration(-0.06, 0.08, 1.03, 0.98, 0.2, 1.08)


def synthetic_matches() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """900 matches drawn with numpy's legacy generator, whose stream never changes."""
    state = np.random.RandomState(20261024)
    count = 900
    home = np.exp(state.normal(np.log(1.45), 0.35, count))
    away = np.exp(state.normal(np.log(1.15), 0.35, count))
    return home, away, state.poisson(home * 1.05), state.poisson(away * 0.97)


def test_every_matrix_sums_to_one():
    home, away, _, _ = synthetic_matches()
    matrices = score_matrices(home, away, REFERENCE_CALIBRATION)
    assert matrices.shape == (900, MAXIMUM_GOALS + 1, MAXIMUM_GOALS + 1)
    np.testing.assert_allclose(matrices.sum(axis=(1, 2)), 1.0, atol=1e-12)
    assert (matrices >= 0).all()


def test_the_neutral_calibration_is_two_independent_poissons():
    matrix = score_matrices(np.array([1.6]), np.array([0.9]), Calibration())[0]
    expected = np.outer(poisson.pmf(np.arange(11), 1.6), poisson.pmf(np.arange(11), 0.9))
    np.testing.assert_allclose(matrix[:10, :10], expected[:10, :10], atol=1e-12)


def test_the_matrix_reproduces_the_previous_projects_numbers():
    home, away, _, _ = synthetic_matches()
    matrix = score_matrices(home[:1], away[:1], REFERENCE_CALIBRATION)
    assert sum_result_probabilities(matrix)[0].tolist() == pytest.approx(
        [0.6367241485195303, 0.2118738838382979, 0.15140196764217204], abs=1e-15
    )
    assert matrix[0, 1, 1] == pytest.approx(0.09802832979944714, abs=1e-15)
    assert matrix[0, 10, 0] == pytest.approx(3.0451107892179837e-05, abs=1e-17)


@pytest.mark.parametrize(
    ("fit_spread", "fit_dispersion", "expected"),
    [
        (False, False, Calibration(0.14436481962285724, 0.23339840527225444,
                                   1.1109707567992748, 0.948023791337866, 0.0, 1.0)),
        (True, True, Calibration(0.09822199991132362, 0.12675403267934365, 1.1654995549927163,
                                 0.9684473660702897, -0.1999992594521797, 1.062916635109489)),
    ],
)  # fmt: skip
def test_the_calibration_fit_reproduces_the_previous_projects_numbers(
    fit_spread, fit_dispersion, expected
):
    home, away, home_goals, away_goals = synthetic_matches()
    fitted = fit_calibration(home, away, home_goals, away_goals, fit_spread, fit_dispersion)
    for name, value in expected.__dict__.items():
        assert getattr(fitted, name) == pytest.approx(value, abs=1e-9), name


def test_a_spread_of_zero_is_the_plain_product_and_a_positive_one_widens_the_gap():
    home, away = np.array([2.0]), np.array([0.8])
    assert calibrate_intensities(home, away, Calibration(home_scale=1.1)) == (
        pytest.approx(home * 1.1), pytest.approx(away),
    )  # fmt: skip
    wider_home, wider_away = calibrate_intensities(home, away, Calibration(spread=0.5))
    assert wider_home[0] > 2.0 and wider_away[0] < 0.8
    assert wider_home[0] * wider_away[0] == pytest.approx(2.0 * 0.8)


def test_the_probability_of_a_score_folds_ten_goals_and_more_into_the_last_cell():
    matrices = score_matrices(np.array([1.3, 1.3]), np.array([1.1, 1.1]), Calibration())
    probabilities = find_score_probability(matrices, [2, 12], [1, 0])
    assert probabilities[0] == matrices[0, 2, 1]
    assert probabilities[1] == matrices[1, 10, 0]


def test_the_published_matrix_answers_the_lines_as_the_matrix_does():
    matrix = score_matrices(np.array([1.6]), np.array([1.0]), REFERENCE_CALIBRATION)[0]
    published = format_published_matrix(matrix)
    home_win, draw, away_win = sum_result_probabilities(matrix[None])[0]
    assert read_line("joint_count", published, "home_win") == pytest.approx(home_win, abs=1e-12)
    assert read_line("joint_count", published, "draw") == pytest.approx(draw, abs=1e-12)
    assert read_line("joint_count", published, "away_win") == pytest.approx(away_win, abs=1e-12)


def test_a_forecast_rebuilds_the_matrix_it_was_read_from():
    from vesta.spine.match_forecast import MatchForecast

    matrix = score_matrices(np.array([1.6]), np.array([1.0]), REFERENCE_CALIBRATION)[0]
    forecast = MatchForecast(1, "PL", 2026, "2026-10-24T15:00:00Z", 1.6, 1.0,
                             tuple(sum_result_probabilities(matrix[None])[0]), "home",
                             REFERENCE_CALIBRATION)  # fmt: skip
    assert np.array_equal(forecast.matrix, matrix)
