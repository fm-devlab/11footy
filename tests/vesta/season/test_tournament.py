"""The cups and national tournaments (T4, T6): formats, bracket rules, ties drawn, the round each
team reaches and its decided lines."""

import numpy as np
import pytest

from vesta.forecast_reading import read_line
from vesta.quantities import load_quantities
from vesta.season.brackets import build
from vesta.season.knockout import Tie, TieSampler, list_leaves
from vesta.season.league import LeagueSeason
from vesta.season.standings import read_tie_break_rule
from vesta.season.tournament import (
    find_possible_outcomes,
    read_tournament_format,
    round_reached_distributions,
    simulate_knockout,
    simulate_tournament,
)
from vesta.spine.score_matrix import SIZE, Calibration, score_matrices
from vesta.spine.ties import TieParameters, TieRule

DEFINITIONS = load_quantities()
PARAMETERS = TieParameters(extra_time_minutes=30.0, extra_time_pace=1.0, shootout_home_share=0.5)
NO_AWAY_GOALS = TieRule(away_goals=False, away_goals_in_extra_time=False)


def _strength(ratings: dict[int, float]):
    def strength(home: int, away: int, neutral: bool):
        advantage = 0.0 if neutral else 0.25
        home_rate = float(np.exp(0.2 + advantage + ratings[home] - ratings[away]))
        away_rate = float(np.exp(0.2 + ratings[away] - ratings[home]))
        matrix = score_matrices(np.array([home_rate]), np.array([away_rate]), Calibration())[0]
        return matrix, home_rate, away_rate

    return strength


def test_formats_are_read_per_competition_and_season_and_never_assumed():
    form = read_tournament_format("ECL", 2025)
    assert form.phase == "league_phase" and form.bracket == "uefa_league_phase_2024"
    assert form.outcomes[0] == "league_phase" and form.outcomes[-2:] == ("final", "winner")
    assert read_tournament_format("CL", 2019).phase == "group_stage"
    assert read_tournament_format("EL", 2019).drop_in.source == "CL"
    with pytest.raises(ValueError, match="no format"):
        read_tournament_format("EL", 2018)  # the 2018-19 round of 32 is not written


def test_the_uefa_bracket_from_2024_follows_the_places():
    order = list(range(1, 37))
    tree = build("uefa_league_phase_2024", {"": order}, read_tournament_format("CL", 2025).rounds,
                 np.random.default_rng(5))  # fmt: skip
    assert sorted(list_leaves(tree)) == list(range(1, 25))  # places 25-36 are out
    halves = [set(list_leaves(tree.first)), set(list_leaves(tree.second))]
    for pair in ((1, 2), (3, 4), (5, 6), (7, 8)):  # each seeded pair in opposite halves
        assert sum(pair[0] in half for half in halves) == 1
        assert sum(pair[1] in half for half in halves) == 1
        assert all(len(set(pair) & half) == 1 for half in halves)

    def ties(node, name):
        if isinstance(node, int):
            return []
        own = [node] if node.round_name == name else []
        return own + ties(node.first, name) + ties(node.second, name)

    for play_off in ties(tree, "knockout_play_offs"):  # the better placed at home second
        assert {play_off.first, play_off.second} <= set(range(17, 25)) | set(range(9, 17))
        assert play_off.second < play_off.first
        assert play_off.first + play_off.second in (32, 33, 34)  # 9/10 v 23/24 and so on
    for last_16 in ties(tree, "round_of_16"):
        assert isinstance(last_16.second, int) and last_16.second <= 8


def test_the_world_cup_and_the_group_era_brackets():
    groups = {letter: [10 * index + 1, 10 * index + 2, 10 * index + 3, 10 * index + 4]
              for index, letter in enumerate("ABCDEFGH")}  # fmt: skip
    rounds = read_tournament_format("WC", 2022).rounds
    tree = build("world_cup_32", groups, rounds, np.random.default_rng(1))
    first = tree.first.first.first
    assert (first.first, first.second) == (1, 12)  # 1A v 2B opens the bracket
    assert len(list_leaves(tree)) == 16 and tree.neutral
    drawn = build("group_winners_v_runners_up", groups, read_tournament_format("CL", 2019).rounds,
                  np.random.default_rng(2))  # fmt: skip

    def last_16(node):
        if node.round_name == "round_of_16":
            return [node]
        return last_16(node.first) + last_16(node.second)

    for tie in last_16(drawn):
        assert tie.second % 10 == 1 and tie.first % 10 == 2  # the winner at home second
        assert tie.second // 10 != tie.first // 10  # never its own group


def test_away_goals_decide_a_level_tie_only_where_the_rules_say_so():
    one_nil = np.zeros((SIZE, SIZE))
    one_nil[0, 1] = 1.0  # the second leg ends 0-1 every time

    def strength(home, away, neutral):
        return one_nil, 0.0, 0.0  # no goal in extra time: a level tie goes to penalties

    tie = Tie("round_of_16", 2, 101, 202, first_leg=(1, 2))  # 202 scored two away
    with_rule = TieSampler(strength, TieRule(True, False), PARAMETERS, np.random.default_rng(3))
    assert {with_rule.draw_winner(tie, 101, 202) for _ in range(50)} == {202}
    without = TieSampler(strength, NO_AWAY_GOALS, PARAMETERS, np.random.default_rng(3))
    winners = [without.draw_winner(tie, 101, 202) for _ in range(2000)]
    assert np.mean(np.array(winners) == 202) == pytest.approx(0.5, abs=0.04)  # penalties


def _league_phase(ratings: dict[int, float], generator) -> LeagueSeason:
    teams = sorted(ratings)
    fixtures = []
    for index, team in enumerate(teams):  # eight opponents each, four at home
        for step in (1, 2, 3, 4):
            fixtures.append((team, teams[(index + step * 7) % len(teams)]))
    strength = _strength(ratings)
    matrices = np.array([strength(home, away, False)[0] for home, away in fixtures])
    return LeagueSeason(tuple(teams), (), {}, tuple(fixtures), matrices)


def test_a_cup_is_played_from_its_league_phase_to_its_final():
    generator = np.random.default_rng(4)
    ratings = {100 + index: float(rating)
               for index, rating in enumerate(np.linspace(0.8, -0.8, 36))}  # fmt: skip
    form = read_tournament_format("CL", 2025)
    arguments = (form, {"": _league_phase(ratings, generator)}, read_tie_break_rule("CL", 2025),
                 _strength(ratings), NO_AWAY_GOALS, PARAMETERS, 400, 9)  # fmt: skip
    draws = simulate_tournament(*arguments)
    again = simulate_tournament(*arguments)
    winner = form.outcomes.index("winner")
    winners = np.array([reached == winner for reached in draws.reached.values()])
    assert (winners.sum(axis=0) == 1).all()  # one winner in every simulation
    assert winners[:6].sum() > winners[-6:].sum()  # the strongest win more often
    assert all((draws.reached[team] == again.reached[team]).all() for team in draws.reached)
    published = round_reached_distributions("T4.round_reached", draws, {}, "test", DEFINITIONS)
    for item in published:
        assert sum(share for _, share in item.values) == pytest.approx(1.0)
        assert item.decided_lines == ()


def test_a_knockout_under_way_decides_what_is_already_known():
    form = read_tournament_format("WC", 2022)
    semi_1 = Tie("semi_finals", 1, 1, 2, neutral=True, winner=1)
    semi_2 = Tie("semi_finals", 1, 3, 4, neutral=True)
    tree = Tie("final", 1, semi_1, semi_2, neutral=True)
    out = {5: "quarter_finals", 6: "group_stage"}
    ratings = {team: 0.0 for team in range(1, 7)}
    draws = simulate_knockout(form, tree, out, _strength(ratings), NO_AWAY_GOALS, PARAMETERS,
                              300, 1)  # fmt: skip
    known = find_possible_outcomes(form, tree, out, list(range(1, 7)))
    published = {item.subject_id: item for item in round_reached_distributions(
        "T6.round_reached", draws, known, "test", DEFINITIONS)}  # fmt: skip
    assert "any=final,winner" in published[1].decided_lines  # 1 is in the final
    assert "outcome=winner" not in published[1].decided_lines
    assert set(published[2].decided_lines) == set(DEFINITIONS["T6.round_reached"].quantity.lines)
    assert read_line("ordered", published[2].values, "any=semi_finals,final,winner") == 1.0
    assert set(published[6].decided_lines) == set(DEFINITIONS["T6.round_reached"].quantity.lines)
    assert published[3].decided_lines == ("any=semi_finals,final,winner",
                                          "any=quarter_finals,semi_finals,final,winner",
                                          "any=round_of_16,quarter_finals,semi_finals,final,"
                                          "winner")  # fmt: skip
