"""The league season simulation (T1) and the round in numbers (C4): the tie-break rules, the
simulated tables, the decided lines, the exact totals of a round."""

from dataclasses import replace

import numpy as np
import pytest

from vesta.forecast_reading import read_line
from vesta.quantities import load_quantities
from vesta.season.league import LeagueSeason, build_league_distributions, simulate_league
from vesta.season.standings import (
    SeasonMatches,
    TieBreakRule,
    count_season_totals,
    rank,
    read_tie_break_rule,
)
from vesta.spine.score_matrix import Calibration, score_matrices, sum_result_probabilities

DEFINITIONS = load_quantities()
# the leagues have 18 or 20 teams; the test league has four, so its position lines stop at four
_POSITION = DEFINITIONS["T1.final_position"]
FOUR_TEAMS = {**DEFINITIONS, "T1.final_position": replace(_POSITION, quantity=replace(
    _POSITION.quantity, lines=("place=1", "place=2", "place=3", "place=4", "top=2", "bottom=1",
                               "bottom=3")))}  # fmt: skip
A, B, C = 0, 1, 2
# A and B level on 5 points: B has the better goal difference, A won their meetings; C far below
PLAYED = [(A, B, 1, 0), (B, A, 0, 0), (A, C, 0, 0), (C, A, 1, 0), (B, C, 5, 0), (C, B, 0, 0)]


def _matches(played, draws=1) -> SeasonMatches:
    rows = np.array(played)
    return SeasonMatches(rows[:, 0], rows[:, 1], np.tile(rows[:, 2], (draws, 1)),
                         np.tile(rows[:, 3], (draws, 1)))  # fmt: skip


def test_the_rules_are_read_per_competition_and_season_and_never_assumed(tmp_path):
    assert read_tie_break_rule("SA", 2021).criteria[1] == "head_to_head_points"
    assert read_tie_break_rule("SA", 2022).criteria[1] == "play_off=1,17"
    assert read_tie_break_rule("PL", 2025).verified and not read_tie_break_rule("PL", 2026).verified
    with pytest.raises(ValueError, match="no tie-break rules"):
        read_tie_break_rule("XX", 2025)
    wrong = tmp_path / "rules.csv"
    wrong.write_text("competition,first_season,last_season,criteria,verified,source\n"
                     "PL,2014,2099,points|form,yes,test\n", encoding="utf-8")  # fmt: skip
    with pytest.raises(ValueError, match="unknown tie-break criterion"):
        read_tie_break_rule("PL", 2025, wrong)


def test_goal_difference_or_the_meetings_decide_as_the_rules_say():
    matches = _matches(PLAYED)
    totals = count_season_totals(matches, 3, np.array([0, 0, 10]))
    assert totals["points"][0].tolist() == [5, 5, -5]
    generator = np.random.default_rng(1)
    by_goal_difference = rank(totals, matches, read_tie_break_rule("PL", 2025), generator)
    by_meetings = rank(totals, matches, read_tie_break_rule("LL", 2025), generator)
    assert by_goal_difference[0].tolist() == [1, 0, 2]
    assert by_meetings[0].tolist() == [0, 1, 2]


def test_lots_and_a_play_off_are_drawn_at_even_chances():
    level = [(A, B, 1, 1), (B, A, 1, 1)]
    matches = _matches(level, draws=4000)
    totals = count_season_totals(matches, 2, np.zeros(2))
    first = rank(totals, matches, TieBreakRule(("points", "lots"), True),
                 np.random.default_rng(2))[:, A] == 0  # fmt: skip
    assert first.mean() == pytest.approx(0.5, abs=0.03)
    played_off = rank(totals, matches, TieBreakRule(("points", "play_off=1", "wins"), True),
                      np.random.default_rng(3))[:, A] == 0  # fmt: skip
    assert played_off.mean() == pytest.approx(0.5, abs=0.03)
    # a play-off for another place: the next criterion, here none, keeps the order
    kept = rank(totals, matches, TieBreakRule(("points", "play_off=2"), True),
                np.random.default_rng(4))  # fmt: skip
    assert (kept[:, A] == 0).all()


def _season(played, fixtures) -> LeagueSeason:
    strengths = {101: 0.5, 102: 0.2, 103: -0.2, 104: -0.5}
    matrices = score_matrices(
        np.array([np.exp(0.3 + strengths[home] - strengths[away]) for home, away in fixtures]),
        np.array([np.exp(strengths[away] - strengths[home]) for home, away in fixtures]),
        Calibration(),
    ) if fixtures else np.zeros((0, 11, 11))  # fmt: skip
    return LeagueSeason((101, 102, 103, 104), tuple(played), {}, tuple(fixtures), matrices)


ALL_PAIRS = [(home, away) for home in (101, 102, 103, 104) for away in (101, 102, 103, 104)
             if home != away]  # fmt: skip


def test_a_season_is_drawn_to_its_end_and_published_with_its_decided_lines():
    # 104 lost its first five matches: with one left it can no longer reach 101's points
    played = [(101, 104, 3, 0), (104, 102, 0, 1), (103, 104, 2, 0), (104, 101, 0, 2),
              (102, 104, 1, 0), (101, 102, 1, 1), (102, 101, 0, 0), (101, 103, 2, 0),
              (103, 101, 1, 1), (102, 103, 0, 0), (103, 102, 1, 0)]  # fmt: skip
    fixtures = [(104, 103)]
    season = _season(played, fixtures)
    rule = read_tie_break_rule("PL", 2025)
    draws = simulate_league(season, rule, 3000, seed=7)
    again = simulate_league(season, rule, 3000, seed=7)
    np.testing.assert_array_equal(draws.places, again.places)  # seeded
    published = {
        (item.quantity_key, item.subject_id): item
        for item in build_league_distributions(season, draws, "test", FOUR_TEAMS)
    }
    bottom = published[("T1.final_position", 104)]
    assert "place=1" in bottom.decided_lines and "bottom=3" in bottom.decided_lines
    for item in published.values():
        kind = FOUR_TEAMS[item.quantity_key].quantity.distribution_kind
        for line in item.decided_lines:
            value = read_line(kind, item.values, line)
            assert value in (0.0, 1.0) or abs(value - round(value)) < 1e-9
    leader = published[("T1.points", 101)]
    # 101 has 12 points and nothing left to play: every line of its points is decided
    assert leader.decided_lines == DEFINITIONS["T1.points"].quantity.lines
    assert leader.values["probabilities"][12] == 1.0
    assert sum(published[("T1.final_position", team)].values[0]
               for team in (101, 102, 103, 104)) == pytest.approx(1.0)  # fmt: skip


def test_a_whole_season_ahead_follows_the_strengths():
    season = _season([], ALL_PAIRS)
    draws = simulate_league(season, read_tie_break_rule("LL", 2025), 4000, seed=8)
    champion = (draws.places == 0).mean(axis=0)
    assert champion[0] > champion[1] > champion[2] > champion[3]
    assert champion.sum() == pytest.approx(1.0)
    expected = draws.points.mean(axis=0)
    home_points = sum_result_probabilities(season.matrices) @ np.array([3, 1, 0])
    assert expected.sum() == pytest.approx(
        home_points.sum() + (sum_result_probabilities(season.matrices) @ np.array([0, 1, 3])).sum(),
        rel=0.01,
    )




def test_the_history_check_lists_the_teams_placed_differently():
    from vesta.season.standings_check import compare_season

    deductions = {C: 10}
    by_meetings = {A: 1, B: 2, C: 3}
    assert compare_season([A, B, C], PLAYED, deductions, by_meetings,
                          read_tie_break_rule("LL", 2025)) == []
    assert compare_season([A, B, C], PLAYED, deductions, by_meetings,
                          read_tie_break_rule("PL", 2025)) == [(A, 2, 1, False), (B, 1, 2, False)]
    level = [(A, B, 1, 1), (B, A, 1, 1)]
    marked = compare_season([A, B], level, {}, {A: 2, B: 1}, TieBreakRule(("points", "lots"), True))
    assert all(by_lots for *_, by_lots in marked)


def test_the_new_rules_restart_share_and_play_off_as_the_regulations_say(tmp_path):
    # A, B, C level on 3 points (after deductions); among the three A and B have the same
    # head-to-head goal difference (+1), A scored more among the three, B beat A
    played = [(A, B, 0, 1), (A, C, 2, 0), (B, C, 0, 0)]
    matches = _matches(played)
    totals = count_season_totals(matches, 3, np.array([0, 1, -2]))
    assert totals["points"][0].tolist() == [3, 3, 3]

    def order(*criteria):
        return rank(totals, matches, TieBreakRule(("points", *criteria), True),
                    np.random.default_rng(1))[0].tolist()  # fmt: skip

    # one table among all three (no restart): A over B by goals among the three
    assert order("head_to_head_goal_difference", "head_to_head_goals_for", "lots") == [0, 1, 2]
    # LaLiga: C separated, A and B start again between themselves: B beat A
    assert order("head_to_head_goal_difference", "head_to_head_goals_for",
                 "restart_on_separation", "lots") == [1, 0, 2]  # fmt: skip
    # UEFA: still level after the run, the run again between A and B alone
    assert order("head_to_head_goal_difference", "restart_head_to_head", "lots") == [1, 0, 2]
    # the Premier League: level teams share a place unless it is one played off
    level = [(A, B, 1, 1), (B, A, 1, 1)]
    even = _matches(level, draws=400)
    even_totals = count_season_totals(even, 3, np.array([0, 0, 9]))  # C far below
    shared = rank(even_totals, even, TieBreakRule(("points", "play_off=2", "shared"), True),
                  np.random.default_rng(2))  # fmt: skip
    assert (shared[:, A] == 0).all() and (shared[:, B] == 0).all()  # level for 1st: shared
    played_off = rank(even_totals, even, TieBreakRule(("points", "play_off=1", "shared"), True),
                      np.random.default_rng(3))  # fmt: skip
    assert sorted(set(played_off[:, A].tolist())) == [0, 1]
    assert (played_off[:, A] != played_off[:, B]).all()
    wrong = tmp_path / "rules.csv"
    wrong.write_text("competition,first_season,last_season,criteria,verified,source\n"
                     "PL,2014,2099,points|play_off=1,4|shared,yes,test\n",
                     encoding="utf-8")  # fmt: skip
    with pytest.raises(ValueError, match="malformed"):
        read_tie_break_rule("PL", 2025, wrong)


def test_a_uefa_finalist_plays_no_play_off_in_serie_a_from_2026():
    from dataclasses import replace

    rule = read_tie_break_rule("SA", 2026)
    assert rule.verified and "play_off_unless_uefa_finalist" in rule.criteria
    # A and B level on points for the title; A won their meetings
    level = [(A, B, 1, 0), (B, A, 1, 1), (A, C, 0, 0), (C, A, 0, 0), (B, C, 1, 0), (C, B, 0, 0)]
    matches = _matches(level, draws=2000)
    totals = count_season_totals(matches, 3, np.array([0, -1, 0]))  # B one point back, level with A
    assert totals["points"][0].tolist()[:2] == [6, 6]
    played_off = rank(totals, matches, rule, np.random.default_rng(1))
    assert (played_off[:, A] == 0).mean() == pytest.approx(0.5, abs=0.04)
    finalist = rank(totals, matches, replace(rule, play_off_exempt=frozenset({B})),
                    np.random.default_rng(1))  # fmt: skip
    assert (finalist[:, A] == 0).all()  # no play-off: the head-to-head table decides
