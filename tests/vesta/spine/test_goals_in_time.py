"""Goals in time: the population, its agreement with the matrix, and the step-3 distributions."""

import tomllib
from pathlib import Path

import numpy as np
import pytest

from vesta.distributions import check_distribution
from vesta.forecast_reading import read_line
from vesta.quantities import load_quantities
from vesta.spine.goals_in_time import (
    METHODS,
    GoalsInTimeParameters,
    TimeShares,
    fit_first_half_shares,
    fit_method_shares,
    simulate,
)
from vesta.spine.goals_in_time_publication import build_goals_in_time_distributions
from vesta.spine.score_matrix import SIZE, Calibration, score_matrices, sum_result_probabilities
from vesta.spine.world import read_goal_methods, read_half_time_scores

ROOT = Path(__file__).resolve().parents[3]
SPINE = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
PARAMETERS = GoalsInTimeParameters.from_config(SPINE)
SHARES = TimeShares(0.44, 0.45, (0.11, 0.03, 0.04, 0.15, 0.16, 0.51))
MATCH, HOME, AWAY = 77120, 13, 14
MATRIX = score_matrices(np.array([1.6]), np.array([1.1]),
                        Calibration(-0.05, 0.08, 1.0, 1.0, 0.1, 1.08))[0]  # fmt: skip


@pytest.fixture(scope="module")
def published():
    population = simulate(MATRIX, SHARES, PARAMETERS, MATCH)
    return by_key(population)


def by_key(population) -> dict:
    distributions = build_goals_in_time_distributions(population, MATCH, HOME, AWAY, "v")
    return {(item.quantity_key, item.subject_id): item.values for item in distributions}


def one_score(home: int, away: int) -> np.ndarray:
    matrix = np.zeros((SIZE, SIZE))
    matrix[home, away] = 1.0
    return matrix


def test_the_shares_are_read_from_the_training_matches():
    home, away = fit_first_half_shares(np.array([2, 1, 0]), np.array([1, 3, 0]),
                                       np.array([1, 0, 0]), np.array([0, 2, 0]), 0.45)  # fmt: skip
    assert (home, away) == (pytest.approx(1 / 3), pytest.approx(2 / 4))
    assert fit_first_half_shares(np.array([0]), np.array([0]), np.array([0]), np.array([0]),
                                 0.45) == (0.45, 0.45)  # fmt: skip
    shares = fit_method_shares(["header", "penalty", "header", "own_goal"])
    assert dict(zip(METHODS, shares, strict=True))["header"] == 0.5


def test_the_population_keeps_the_final_score_of_the_matrix_exactly():
    population = simulate(MATRIX, SHARES, PARAMETERS, MATCH)
    home_win = float(np.sum(population.weight * (population.home_goals > population.away_goals)))
    assert home_win == pytest.approx(sum_result_probabilities(MATRIX[None])[0][0], abs=1e-12)
    assert population.weight.sum() == pytest.approx(1.0, abs=1e-12)


def test_every_published_distribution_sums_to_one_and_answers_its_lines(published):
    definitions = load_quantities()
    for (key, _), values in published.items():
        quantity = definitions[key].quantity
        check_distribution(quantity.distribution_kind, values)
        for line in quantity.lines:
            assert 0.0 <= read_line(quantity.distribution_kind, values, line) <= 1.0, (key, line)
    step_three = {
        key for key, item in definitions.items()
        if item.family_member == "goals in time" and not key.startswith("M14.")
    }  # fmt: skip
    assert step_three == {key for key, _ in published}


def test_the_numbers_of_step_three_agree_with_the_matrix_and_with_one_another(published):
    goalless = MATRIX[0, 0]
    assert published[("M7.first_goal", MATCH)]["none"] == pytest.approx(goalless, abs=1e-12)
    assert published[("M8.first_goal_time", MATCH)]["never"] == pytest.approx(goalless, abs=1e-12)
    first_method = published[("M13.first_goal_method", MATCH)]
    assert first_method["none"] == pytest.approx(goalless, abs=1e-12)
    half_full = published[("M9.half_time_full_time", MATCH)]
    home_win, _, _ = sum_result_probabilities(MATRIX[None])[0]
    assert sum(p for name, p in half_full.items() if name.endswith("/home")) == pytest.approx(
        home_win, abs=1e-12
    )
    first_half = dict(published[("M9.result.first_half", MATCH)])
    assert sum(p for name, p in half_full.items() if name.startswith("home/")) == pytest.approx(
        first_half["home"], abs=1e-12
    )
    first_half_goals = published[("M12.goals.first_half", MATCH)]
    assert read_line("count", first_half_goals, "under=0.5") == pytest.approx(
        read_line("joint_count", published[("M9.score.first_half", MATCH)], "score=0-0"), abs=1e-12
    )


def test_a_one_nil_has_no_comeback_and_a_first_half_lead_as_often_as_the_goal_falls_there():
    population = simulate(one_score(1, 0), SHARES, PARAMETERS, MATCH)
    values = by_key(population)
    assert values[("M11.comes_from_behind_to_win.team", HOME)]["yes"] == 0.0
    assert values[("M11.leads_at_half_time_and_wins.team", HOME)]["yes"] == pytest.approx(
        SHARES.first_half_home, abs=0.02
    )  # a Monte Carlo share of 10,000 draws
    assert values[("M7.first_goal", MATCH)] == {
        "home": pytest.approx(1.0, abs=1e-12), "away": 0.0, "none": 0.0
    }


def test_a_two_one_comeback_happens_only_when_the_away_side_scores_first():
    population = simulate(one_score(2, 1), SHARES, PARAMETERS, MATCH)
    values = by_key(population)
    away_first = values[("M7.first_goal", MATCH)]["away"]
    assert values[("M11.comes_from_behind_to_win.team", HOME)]["yes"] == pytest.approx(away_first)
    assert values[("M11.comes_from_behind_to_win.team", AWAY)]["yes"] == 0.0


def test_an_own_goal_credited_to_one_side_is_scored_by_the_other():
    all_own_goals = TimeShares(0.45, 0.45, (0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
    population = simulate(one_score(1, 0), all_own_goals, PARAMETERS, MATCH)
    values = by_key(population)
    assert values[("M23.own_goal.team", AWAY)]["yes"] == pytest.approx(1.0)
    assert values[("M23.own_goal.team", HOME)]["yes"] == 0.0


def test_a_match_is_reproduced_by_its_seed_and_another_match_draws_its_own():
    first = simulate(MATRIX, SHARES, PARAMETERS, MATCH)
    again = simulate(MATRIX, SHARES, PARAMETERS, MATCH)
    other = simulate(MATRIX, SHARES, PARAMETERS, MATCH + 1)
    assert np.array_equal(first.home_minute, again.home_minute)
    assert not np.array_equal(first.home_minute, other.home_minute)
    assert ((first.home_minute >= 0) & (first.home_minute <= 90)).all()


def test_the_shares_are_read_from_the_world(world):
    world.execute("UPDATE matches SET home_goals_half_time = 0, away_goals_half_time = 0")
    world.execute("UPDATE matches SET home_goals_half_time = home_goals WHERE match_id <= 10")
    scores = read_half_time_scores(world, ["PL"], 2014, 2014)
    assert len(scores.home_goals) == 88  # of the 90: one awarded, one cancelled
    first = [(1, "penalty_goal", None), (2, "own_goal", None), (3, "goal", "head"),
             (4, "goal", "right_foot"), (5, "goal", None)]  # fmt: skip
    for sequence, kind, body_part in first:
        shot_id = None
        if body_part:
            shot_id = world.execute(
                "INSERT INTO shots (match_id, team_id, period, minute, x, y, result, situation,"
                " body_part, is_penalty, zone, source, layer, also_in_commentary) VALUES (1, 101,"
                " 1, 10, 0.9, 0.5, 'goal', 'open_play', ?, 0, 'penalty_area', 'tracked',"
                " 'tracked', 0)",
                (body_part,),
            ).lastrowid
        world.execute(
            "INSERT INTO match_events (match_id, sequence, period, minute, kind, team_id,"
            " shot_id, source) VALUES (1, ?, 1, 10, ?, 101, ?, 'test')",
            (sequence, kind, shot_id),
        )
    assert sorted(read_goal_methods(world, ["PL"], 2014, 2014)) == [
        "header", "open_play_inside_box", "own_goal", "penalty",
    ]  # the goal without its shot has no manner
