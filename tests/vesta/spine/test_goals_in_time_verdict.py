"""The verdict of goals in time: what a match's story says happened, and the check itself."""

import math
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np

from vesta.quantities import load_quantities
from vesta.spine.goals_in_time import GoalsInTimeParameters, TimeShares
from vesta.spine.goals_in_time_verdict import check_goals_in_time, find_outcomes
from vesta.spine.match_forecast import MatchForecast
from vesta.spine.score_matrix import Calibration, score_matrices, sum_result_probabilities
from vesta.spine.world import MatchStory

ROOT = Path(__file__).resolve().parents[3]
SPINE = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
QUICK = replace(GoalsInTimeParameters.from_config(SPINE), simulations=300, minimum_per_score=2)
SHARES = TimeShares(0.44, 0.45, (0.11, 0.03, 0.04, 0.15, 0.16, 0.51))


def test_a_comeback_with_an_own_goal_reads_as_it_happened():
    # 0-1 at 20' (header), 1-1 at 50' (an own goal by the away side), 2-1 at 88' (penalty)
    goals = ((20, "away", "header"), (50, "home", "own_goal"), (88, "home", "penalty"))
    story = MatchStory(9, 13, 14, (0, 1), (2, 1), goals)
    happened = find_outcomes(story)
    assert happened[("M7.first_goal", "match")] == "away"
    assert happened[("M7.last_goal", "match")] == "home"
    assert happened[("M8.first_goal_time", "match")] == 20
    assert happened[("M8.goal_in_band.band=81-90", "match")] is True
    assert happened[("M9.half_time_full_time", "match")] == "away/home"
    assert happened[("M10.score.second_half", "match")] == (2, 0)
    assert happened[("M10.highest_scoring_half", "match")] == "second"
    assert happened[("M11.comes_from_behind_to_win.team", "home")] is True
    assert happened[("M11.leads_at_half_time_and_wins.team", "home")] is False
    assert happened[("M13.first_goal_method", "match")] == "header"
    assert happened[("M13.goal_by_method.method=penalty", "match")] is True
    assert happened[("M23.own_goal.team", "away")] is True  # the away side scored into its net
    assert happened[("M23.own_goal.team", "home")] is False


def test_a_goal_without_its_manner_leaves_the_manners_unchecked():
    story = MatchStory(9, 13, 14, (1, 0), (1, 0), ((10, "home", None),))
    happened = find_outcomes(story)
    assert ("M13.first_goal_method", "match") not in happened
    assert happened[("M7.first_goal", "match")] == "home"


def synthetic(season_count: int = 3, per_season: int = 40):
    """Matches of a few seasons with their forecast and a story drawn from the same model."""
    state = np.random.RandomState(13)
    forecasts, stories = [], {}
    match_id = 0
    for season in range(2016, 2016 + season_count):
        for _ in range(per_season):
            match_id += 1
            home, away = float(state.uniform(1.0, 2.0)), float(state.uniform(0.7, 1.5))
            matrix = score_matrices(np.array([home]), np.array([away]), Calibration())[0]
            goals = []
            full = (int(state.poisson(home)), int(state.poisson(away)))
            for side, count in (("home", full[0]), ("away", full[1])):
                goals += [(int(state.randint(1, 91)), side, "open_play_inside_box")
                          for _ in range(count)]  # fmt: skip
            goals.sort()
            half = (sum(1 for m, s, _ in goals if m <= 45 and s == "home"),
                    sum(1 for m, s, _ in goals if m <= 45 and s == "away"))  # fmt: skip
            result = "home" if full[0] > full[1] else "draw" if full[0] == full[1] else "away"
            forecasts.append(
                MatchForecast(
                    match_id,
                    "PL",
                    season,
                    f"{season}-09-{1 + match_id % 28:02d}T15:00:00Z",
                    home,
                    away,
                    tuple(sum_result_probabilities(matrix[None])[0]),
                    result,
                )
            )
            stories[match_id] = MatchStory(match_id, 100 + match_id, 200 + match_id, half, full,
                                           tuple(goals))  # fmt: skip
    return forecasts, stories


def test_the_check_scores_every_quantity_against_its_reference_and_is_reproduced():
    forecasts, stories = synthetic()
    shares = {season: SHARES for season in (2016, 2017, 2018)}
    season_of = {forecast.match_id: forecast.season for forecast in forecasts}
    arguments = (forecasts, stories, season_of, shares, load_quantities(), QUICK, 200, 7, 2016)
    results = check_goals_in_time(*arguments)
    keys = {result.key for result in results}
    assert "M9.half_time_full_time" in keys and "M11.comes_from_behind_to_win.team" in keys
    for result in results:
        assert result.seasons == 2  # 2016 has no training season before it: no reference
        assert all(math.isfinite(value) for value in (result.model, result.reference,
                                                      result.difference))  # fmt: skip
    assert check_goals_in_time(*arguments) == results
