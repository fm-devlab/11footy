"""The scores of one forecast, by kind (`docs/FORECAST_LOG.md` §9)."""

import math

import pytest

from vesta.research.scores import (
    BRIER,
    LOG_LOSS,
    RANKED_PROBABILITY,
    OutcomeError,
    score,
)

RESULT = [["home", 0.5], ["draw", 0.3], ["away", 0.2]]
GOALS = {"probabilities": [0.2, 0.3, 0.25, 0.15, 0.1], "tail_from": 5, "tail": 0.0}
SCORE = {"matrix": [[0.3, 0.1], [0.2, 0.1]], "tail": 0.3}


def test_result_scores_match_the_reference_formulas():
    scores = score("ordered", RESULT, "home")
    assert scores[LOG_LOSS] == pytest.approx(-math.log(0.5))
    assert scores[BRIER] == pytest.approx(0.25 + 0.09 + 0.04)
    assert scores[RANKED_PROBABILITY] == pytest.approx((0.25 + 0.04) / 2)


def test_a_sure_and_right_forecast_scores_zero():
    scores = score("ordered", [["home", 1.0], ["draw", 0.0], ["away", 0.0]], "home")
    assert scores == {LOG_LOSS: 0.0, BRIER: 0.0, RANKED_PROBABILITY: 0.0}


def test_a_forecast_that_gave_zero_to_what_happened_scores_a_finite_loss():
    loss = score("binary", {"yes": 0.0}, True)[LOG_LOSS]
    assert 30 < loss < 40


def test_count_ranked_probability_score_is_in_counts():
    assert score("count", GOALS, 2)[RANKED_PROBABILITY] == pytest.approx(
        0.04 + 0.25 + 0.0625 + 0.01
    )


def test_count_beyond_the_published_counts_falls_in_the_tail():
    with_tail = {"probabilities": [0.5, 0.3], "tail_from": 2, "tail": 0.2}
    assert score("count", with_tail, 7)[LOG_LOSS] == pytest.approx(-math.log(0.2))


def test_joint_count_scores_the_cell_or_the_tail():
    assert score("joint_count", SCORE, (1, 0))[LOG_LOSS] == pytest.approx(-math.log(0.2))
    assert score("joint_count", SCORE, (4, 1))[LOG_LOSS] == pytest.approx(-math.log(0.3))


def test_time_to_event_scores_the_band_or_never():
    first_goal = {"bands": [[1, 45, 0.6], [46, 90, 0.3]], "never": 0.1}
    assert score("time_to_event", first_goal, 50)[LOG_LOSS] == pytest.approx(-math.log(0.3))
    assert score("time_to_event", first_goal, None)[LOG_LOSS] == pytest.approx(-math.log(0.1))


def test_rank_ranked_probability_score_is_between_zero_and_one():
    worst = score("rank", [1.0, 0.0, 0.0, 0.0], 4)[RANKED_PROBABILITY]
    assert worst == pytest.approx(1.0)


def test_binary_and_categorical_scores():
    assert score("binary", {"yes": 0.8}, False)[BRIER] == pytest.approx(0.64 + 0.64)
    formation = score("categorical", {"4-3-3": 0.7, "4-4-2": 0.3}, "4-4-2")
    assert formation[LOG_LOSS] == pytest.approx(-math.log(0.3))


@pytest.mark.parametrize(
    ("kind", "values", "happened"),
    [
        ("binary", {"yes": 0.5}, 1),
        ("ordered", RESULT, "abandoned"),
        ("count", GOALS, -1),
        ("count", GOALS, True),
        ("rank", [0.5, 0.5], 3),
        ("time_to_event", {"bands": [[1, 45, 1.0]], "never": 0.0}, 60),
        ("histogram", {}, 1),
    ],
)
def test_outcomes_the_kind_cannot_score_are_refused(kind, values, happened):
    with pytest.raises(OutcomeError):
        score(kind, values, happened)


