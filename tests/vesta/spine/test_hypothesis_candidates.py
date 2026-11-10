"""The candidates of H1-H3 on fixtures: the fit of extra time and the shoot-out, the minute of a
goal by the profile, the first-half share by band; the published model untouched when absent."""

import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vesta.spine.goals_in_time import (
    GoalsInTimeParameters,
    TimeShares,
    fit_shares_by_band,
    simulate,
)
from vesta.spine.match_forecast import MatchForecast
from vesta.spine.score_matrix import Calibration, score_matrices
from vesta.spine.ties import TieParameters
from vesta.spine.ties_check import TieRecord, fit_tie_parameters

SPINE = tomllib.loads((Path(__file__).resolve().parents[3] / "config/vesta/spine.toml")
                      .read_text(encoding="utf-8"))  # fmt: skip
BASE = TieParameters(30.0, 1.0, 0.5)


def test_h1_fits_the_pace_of_extra_time_and_the_shoot_outs_home_share():
    ties, forecasts, extra = [], {}, {}
    for number in range(8):  # eight ties to extra time in 2018, the last four to penalties
        deciding = 1000 + number
        forecasts[deciding] = MatchForecast(deciding, "CL", 2018, "2018-10-01T19:00:00Z", 1.2,
                                            0.9, (0.4, 0.3, 0.3), "draw")  # fmt: skip
        extra[deciding] = 1 if number % 2 else 0  # four goals in eight extra times
        penalties = number >= 4
        ties.append(TieRecord(number, "CL", 2018, 2000 + number, deciding,
                              10 if (penalties and number < 7) or not penalties else 20,
                              20, "2018-10-01T19:00:00Z",
                              "shootout" if penalties else "extra_time", 10))  # fmt: skip
    fitted = fit_tie_parameters(ties, forecasts, extra, (2019,), BASE)[2019]
    expected = 8 * (1.2 + 0.9) * 30.0 / 90.0  # the goals the intensities expect in extra time
    assert fitted.extra_time_pace == pytest.approx((4 + 0.5) / (expected + 0.5))
    assert fitted.shootout_home_share == pytest.approx((3 + 0.5) / (4 + 1.0))  # 3 of 4 at home
    nothing = fit_tie_parameters([], forecasts, extra, (2019,), BASE)[2019]
    assert nothing.extra_time_pace == 1.0 and nothing.shootout_home_share == 0.5


PARAMETERS = GoalsInTimeParameters.from_config(SPINE)
MATRIX = score_matrices(np.array([1.6]), np.array([1.1]), Calibration())[0]
SHARES = TimeShares(0.45, 0.45, (0.1, 0.03, 0.04, 0.15, 0.15, 0.53))


def test_h2_moves_goals_late_in_their_half_and_leaves_the_halves_alone():
    late = tuple([0.1] * 30 + [3.0] * 15)  # most goals of each half in its last 15 minutes
    published = simulate(MATRIX, SHARES, PARAMETERS, 7)
    tried = simulate(MATRIX, replace(SHARES, regular_minutes=(late, late)), PARAMETERS, 7)
    np.testing.assert_array_equal(published.home_half, tried.home_half)  # the halves unchanged
    used = tried.home_minute > 0
    first = used & (tried.home_half == 1) & (tried.home_minute < 45)
    assert (tried.home_minute[first] > 30).mean() > 0.6
    assert (published.home_minute[first] > 30).mean() < 0.45


def test_h3_fits_and_uses_a_share_per_band_of_expected_goals():
    home_goals = np.array([2, 2, 2, 2])
    home_half = np.array([2, 0, 1, 1])
    expected = np.array([0.8, 2.0, 1.2, 1.3])
    shares = fit_shares_by_band(home_goals, home_goals, home_half, home_half, expected, expected,
                                0.45)  # fmt: skip
    assert shares[0] == (1.0, 0.5, 0.0)  # below 1.0: all in the first half; above 1.8: none
    banded = replace(SHARES, shares_by_band=((1.0, 1.0, 1.0), (0.0, 0.0, 0.0)))
    population = simulate(MATRIX, banded, PARAMETERS, 7)
    assert (population.home_half[population.home_half > 0] == 1).all()
    assert (population.away_half[population.away_half > 0] == 2).all()
