"""Ties (M14): extra time, shoot-outs, the aggregate and the away goals, on matrices whose answer
is known."""

import tomllib
from pathlib import Path

import numpy as np
import pytest

from vesta.spine.score_matrix import SIZE, Calibration, score_matrices
from vesta.spine.ties import (
    TieParameters,
    TieRule,
    forecast_first_leg_through,
    read_tie_rule,
    resolve_second_leg,
    resolve_single_match,
)

ROOT = Path(__file__).resolve().parents[3]
PARAMETERS = TieParameters.from_config(
    tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
)
AWAY_GOALS = TieRule(True, True)
NO_AWAY_GOALS = TieRule(False, False)


def one_score(home: int, away: int) -> np.ndarray:
    matrix = np.zeros((SIZE, SIZE))
    matrix[home, away] = 1.0
    return matrix


def test_the_rules_are_read_per_competition_and_season():
    assert read_tie_rule("CL", 2019) == AWAY_GOALS
    assert read_tie_rule("CL", 2021) == NO_AWAY_GOALS
    assert read_tie_rule("ECLQ", 2023) == NO_AWAY_GOALS
    with pytest.raises(ValueError, match="no rule"):
        read_tie_rule("WCQ", 2021)


def test_a_single_match_goes_to_extra_time_when_level_and_to_penalties_when_still_level():
    matrix = score_matrices(np.array([1.4]), np.array([1.1]), Calibration())[0]
    outcomes = resolve_single_match(matrix, 1.4, 1.1, PARAMETERS)
    assert outcomes.extra_time == pytest.approx(np.trace(matrix))
    through = sum(outcomes.through["home"].values()) + sum(outcomes.through["away"].values())
    assert through == pytest.approx(1.0)
    assert outcomes.shootout < outcomes.extra_time
    assert outcomes.through["home"]["on_penalties"] == pytest.approx(outcomes.shootout * 0.5)


def test_a_one_nil_first_leg_and_a_one_nil_second_leg_go_to_extra_time_by_either_rule():
    # first leg 1-0 to the side that is away now; second leg 1-0 to the home side
    for rule in (AWAY_GOALS, NO_AWAY_GOALS):
        outcomes = resolve_second_leg(one_score(1, 0), 1.3, 1.1, (1, 0), rule, PARAMETERS)
        assert outcomes.extra_time == pytest.approx(1.0)


def test_the_away_goals_decide_a_level_aggregate_only_in_the_seasons_of_the_rule():
    # first leg 1-0 to the visitors of today; today 2-1: aggregate 2-2, their away goal decides
    with_rule = resolve_second_leg(one_score(2, 1), 1.3, 1.1, (1, 0), AWAY_GOALS, PARAMETERS)
    assert with_rule.through["away"]["in_regular_time"] == pytest.approx(1.0)
    assert with_rule.extra_time == 0.0
    without = resolve_second_leg(one_score(2, 1), 1.3, 1.1, (1, 0), NO_AWAY_GOALS, PARAMETERS)
    assert without.extra_time == pytest.approx(1.0)


def test_an_away_goal_in_extra_time_decides_where_the_rule_counts_it():
    # 1-0, then 1-0 today, then 1-1 in extra time: the visitors' extra-time goal is an away goal
    with_rule = resolve_second_leg(one_score(1, 0), 1.3, 1.1, (1, 0), AWAY_GOALS, PARAMETERS)
    without = resolve_second_leg(one_score(1, 0), 1.3, 1.1, (1, 0), NO_AWAY_GOALS, PARAMETERS)
    assert with_rule.shootout < without.shootout


def test_the_first_leg_reads_who_goes_through_from_the_second_leg():
    through = forecast_first_leg_through(
        one_score(3, 0), one_score(0, 0), (1.2, 1.2), NO_AWAY_GOALS, PARAMETERS
    )
    assert through == {"home": pytest.approx(1.0), "away": pytest.approx(0.0)}
    level = score_matrices(np.array([1.3]), np.array([1.3]), Calibration())[0]
    split = forecast_first_leg_through(level, level, (1.3, 1.3), NO_AWAY_GOALS, PARAMETERS)
    assert split["home"] + split["away"] == pytest.approx(1.0)


