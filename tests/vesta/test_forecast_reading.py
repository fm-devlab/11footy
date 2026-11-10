"""The reader of lines and the check of distributions (`docs/FORECAST_LOG.md` §6)."""

import math

import pytest

from vesta.distributions import DistributionError, check_distribution, compute_expected_count
from vesta.forecast_reading import LineError, read_line, round_to_whole_percentages

GOALS = {"probabilities": [0.2, 0.3, 0.25, 0.15, 0.1], "tail_from": 5, "tail": 0.0}
CORNERS = {"probabilities": [0.1, 0.2, 0.3, 0.2], "tail_from": 4, "tail": 0.2}
SCORE = {"matrix": [[0.1, 0.1, 0.05], [0.2, 0.1, 0.05], [0.15, 0.15, 0.1]], "tail": 0.0}
RESULT = [["home", 0.5], ["draw", 0.3], ["away", 0.2]]
FIRST_GOAL = {"bands": [[1, 15, 0.3], [16, 30, 0.25], [31, 90, 0.35]], "never": 0.1}
PLACE = [0.5, 0.3, 0.15, 0.05]


def test_distributions_of_every_kind_sum_to_one():
    check_distribution("binary", {"yes": 0.3})
    check_distribution("categorical", {"4-3-3": 0.6, "4-4-2": 0.4})
    check_distribution("ordered", RESULT)
    check_distribution("count", GOALS)
    check_distribution("count", CORNERS)
    check_distribution("joint_count", SCORE)
    check_distribution("time_to_event", FIRST_GOAL)
    check_distribution("rank", PLACE)


def test_distribution_that_does_not_sum_to_one_is_refused():
    with pytest.raises(DistributionError, match="sums to"):
        check_distribution("ordered", [["home", 0.5], ["draw", 0.3], ["away", 0.3]])


def test_distribution_with_a_probability_outside_zero_one_is_refused():
    with pytest.raises(DistributionError, match="not in"):
        check_distribution("rank", [1.2, -0.2])


def test_count_whose_tail_does_not_follow_its_counts_is_refused():
    with pytest.raises(DistributionError, match="tail"):
        check_distribution("count", {"probabilities": [0.5, 0.5], "tail_from": 5, "tail": 0.0})


def test_expected_count_counts_the_tail_at_its_start():
    assert compute_expected_count(GOALS) == pytest.approx(0.3 + 0.5 + 0.45 + 0.4)
    assert compute_expected_count(CORNERS) == pytest.approx(0.2 + 0.6 + 0.6 + 0.8)


@pytest.mark.parametrize(
    ("line_key", "expected"),
    [
        ("over=2.5", 0.25),
        ("under=2.5", 0.75),
        ("exactly=1", 0.3),
        ("at_least=2", 0.5),
        ("at_most=1", 0.5),
        ("between=1,3", 0.7),
        ("odd", 0.45),
        ("even", 0.55),
        ("exactly=9", 0.0),
    ],
)
def test_count_lines_read_the_published_counts(line_key, expected):
    assert read_line("count", GOALS, line_key) == pytest.approx(expected)


def test_count_over_a_line_includes_the_tail():
    assert read_line("count", CORNERS, "over=2.5") == pytest.approx(0.4)
    assert read_line("count", CORNERS, "over=3.5") == pytest.approx(0.2)


@pytest.mark.parametrize("line_key", ["over=4.5", "exactly=4", "odd", "between=2,5"])
def test_count_line_inside_a_tail_with_probability_is_refused(line_key):
    with pytest.raises(LineError, match="tail"):
        read_line("count", CORNERS, line_key)


@pytest.mark.parametrize("line_key", ["over=2", "over=x", "between=3,1", "spread=1", "exactly=-1"])
def test_malformed_count_lines_are_refused(line_key):
    with pytest.raises(LineError):
        read_line("count", GOALS, line_key)


@pytest.mark.parametrize(
    ("line_key", "expected"),
    [
        ("home_win", 0.5),
        ("draw", 0.3),
        ("away_win", 0.2),
        ("score=1-0", 0.2),
        ("score=7-7", 0.0),
        ("both_score", 0.4),
        ("home_scores", 0.75),
        ("away_scores", 0.55),
        ("total_over=2.5", 0.3),
        ("total_under=2.5", 0.7),
        ("home_over=1.5", 0.4),
        ("away_over=0.5", 0.55),
        ("margin=home,2", 0.15),
        ("margin=away,1", 0.15),
        ("margin_at_least=home,1", 0.5),
        ("margin_at_least=home,2", 0.15),
        ("margin_at_least=away,1", 0.2),
        ("margin_at_least=away,2", 0.05),
    ],
)
def test_joint_count_lines_read_the_score_matrix(line_key, expected):
    assert read_line("joint_count", SCORE, line_key) == pytest.approx(expected)


def test_joint_count_result_is_refused_while_the_tail_holds_probability():
    with_tail = {"matrix": [[0.5, 0.2], [0.2, 0.05]], "tail": 0.05}
    with pytest.raises(LineError, match="tail"):
        read_line("joint_count", with_tail, "home_win")


def test_result_lines_of_the_score_agree_with_the_result_distribution():
    for outcome, line_key in (("home", "home_win"), ("draw", "draw"), ("away", "away_win")):
        from_result = read_line("ordered", RESULT, f"outcome={outcome}")
        assert read_line("joint_count", SCORE, line_key) == pytest.approx(from_result)


def test_named_lines_read_one_or_several_outcomes():
    assert read_line("ordered", RESULT, "outcome=draw") == pytest.approx(0.3)
    assert read_line("ordered", RESULT, "any=home,draw") == pytest.approx(0.8)
    assert read_line("categorical", {"4-3-3": 0.6, "4-4-2": 0.4}, "outcome=4-4-2") == 0.4
    with pytest.raises(LineError, match="no outcome"):
        read_line("ordered", RESULT, "outcome=abandoned")


def test_binary_lines_are_yes_and_no():
    assert read_line("binary", {"yes": 0.3}, "yes") == 0.3
    assert read_line("binary", {"yes": 0.3}, "no") == pytest.approx(0.7)


def test_time_lines_read_whole_bands():
    assert read_line("time_to_event", FIRST_GOAL, "by=30") == pytest.approx(0.55)
    assert read_line("time_to_event", FIRST_GOAL, "band=16-30") == 0.25
    assert read_line("time_to_event", FIRST_GOAL, "never") == 0.1
    with pytest.raises(LineError, match="band"):
        read_line("time_to_event", FIRST_GOAL, "by=20")


def test_rank_lines_read_places():
    assert read_line("rank", PLACE, "place=1") == 0.5
    assert read_line("rank", PLACE, "top=2") == pytest.approx(0.8)
    assert read_line("rank", PLACE, "bottom=1") == 0.05
    assert read_line("rank", PLACE, "between=2,3") == pytest.approx(0.45)
    with pytest.raises(LineError, match="places run"):
        read_line("rank", PLACE, "top=9")


def test_unknown_kind_has_no_lines():
    with pytest.raises(LineError, match="no lines"):
        read_line("histogram", {}, "yes")


def test_a_winning_margin_is_at_least_one():
    with pytest.raises(LineError, match="at least 1"):
        read_line("joint_count", SCORE, "margin_at_least=home,0")


def test_a_line_over_the_whole_distribution_never_strays_beyond_zero_or_one():
    # these floats sum to 1.0000000000000002, one bit above one, inside the check's tolerance
    goals = {"probabilities": [0.6, 0.4000000000000002], "tail_from": 2, "tail": 0.0}
    check_distribution("count", goals)
    assert math.fsum(goals["probabilities"]) > 1.0
    assert read_line("count", goals, "at_least=0") == 1.0
    assert read_line("count", goals, "over=1.5") == 0.0
    assert read_line("ordered", [["home", 0.6], ["draw", 0.4000000000000002]],
                     "any=home,draw") == 1.0  # fmt: skip
    assert 0.0 <= read_line("count", goals, "under=0.5") <= 1.0


def test_a_set_is_shown_by_largest_remainder_adding_up_to_a_hundred():
    """The percentages rule, point 2: rounded down, the missing points to the largest
    remainders, a tie of remainders to the first outcome."""
    # a tie shown that rounding alone misses (18, 41, 42), and the other way (47, 47, 7)
    assert round_to_whole_percentages([0.1772, 0.40725, 0.41555]) == [18, 41, 41]
    assert round_to_whole_percentages([0.46531, 0.46515, 0.06954]) == [47, 46, 7]
    # equal remainders: home first
    assert round_to_whole_percentages([0.355, 0.355, 0.29]) == [36, 35, 29]
    # a float's last bit
    assert round_to_whole_percentages([0.29, 0.29, 0.42]) == [29, 29, 42]
    assert round_to_whole_percentages([1 / 3, 1 / 3, 1 / 3]) == [34, 33, 33]
    assert round_to_whole_percentages([0.996, 0.003, 0.001]) == [100, 0, 0]
    assert sum(round_to_whole_percentages([0.123, 0.456, 0.421])) == 100
