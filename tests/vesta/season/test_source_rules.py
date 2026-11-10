"""The rules read from the second primary-source check: the Asian Cup's bracket and best thirds,
the comparison of runners-up without their results against the fifth, hosts, the penalties step
and the list for two teams level."""

import numpy as np
import pytest

from vesta.season.brackets import ASIAN_CUP_2027_THIRDS, build
from vesta.season.knockout import list_leaves
from vesta.season.league import LeagueSeason
from vesta.season.standings import TieBreakRule, count_season_totals, rank, read_tie_break_rule
from vesta.season.tournament import read_tournament_format, simulate_tournament
from vesta.spine.score_matrix import Calibration, score_matrices
from vesta.spine.ties import TieParameters, TieRule

TABLE_RULE = TieBreakRule(("points", "goal_difference", "goals_for", "lots"), False)


def _group(teams, played=(), all_left=True) -> LeagueSeason:
    done = {(home, away) for home, away, _, _ in played}
    fixtures = [(home, away) for home in teams for away in teams
                if home != away and (home, away) not in done and all_left]  # fmt: skip
    matrices = (score_matrices(np.full(len(fixtures), 1.3), np.full(len(fixtures), 1.1),
                               Calibration()) if fixtures else np.zeros((0, 11, 11)))  # fmt: skip
    return LeagueSeason(tuple(teams), tuple(played), {}, tuple(fixtures), matrices)


def test_the_asian_cup_table_keeps_the_round_of_16_apart():
    allowed = {"A": "CDE", "B": "ACD", "C": "ABF", "D": "BEF"}  # art. 9.8: whom 1A..1D can meet
    assert len(ASIAN_CUP_2027_THIRDS) == 15
    for groups, met in ASIAN_CUP_2027_THIRDS.items():
        assert sorted(met) == sorted(groups)
        assert all(third in allowed[winner] for winner, third in zip("ABCD", met, strict=True))
    places = {letter: [10 * index + place for place in (1, 2, 3, 4)]
              for index, letter in enumerate("ABCDEF")}  # fmt: skip
    rounds = read_tournament_format("ASIAN", 2027).rounds
    tree = build("afc_asian_cup_2027", places, rounds, np.random.default_rng(1),
                 ("A", "B", "C", "D"))  # fmt: skip
    assert len(list_leaves(tree)) == 16
    assert list_leaves(tree)[:4] == [2, 22, 31, 13]  # 2A v 2C, then 1D v 3B (the table's row ABCD)


def test_an_asian_cup_is_played_from_its_groups_with_the_best_thirds():
    strength_of = {}
    tables = {}
    for index, letter in enumerate("ABCDEF"):
        teams = [10 * index + place for place in (1, 2, 3, 4)]
        tables[letter] = _group(teams)
        strength_of.update({team: 0.3 - 0.2 * place for place, team in enumerate(teams)})

    def strength(home, away, neutral):
        home_rate = float(np.exp(0.2 + strength_of[home] - strength_of[away]))
        away_rate = float(np.exp(0.2 + strength_of[away] - strength_of[home]))
        matrix = score_matrices(np.array([home_rate]), np.array([away_rate]), Calibration())[0]
        return matrix, home_rate, away_rate

    form = read_tournament_format("ASIAN", 2027)
    draws = simulate_tournament(form, tables, read_tie_break_rule("ASIAN", 2027), strength,
                                TieRule(False, False), TieParameters(30.0, 1.0, 0.5), 300,
                                2)  # fmt: skip
    reached = np.stack(list(draws.reached.values()))
    assert ((reached >= 1).sum(axis=0) == 16).all()  # 12 first and second, 4 thirds
    assert ((reached == form.outcomes.index("winner")).sum(axis=0) == 1).all()






def test_penalties_for_two_level_who_met_last_and_a_list_for_two():
    # A and B level on everything; their match is the last of both
    matches_played = [(0, 2, 1, 0), (1, 2, 1, 0), (0, 1, 1, 1)]
    rows = np.array(matches_played)
    from vesta.season.standings import SeasonMatches

    draws = 2000
    matches = SeasonMatches(rows[:, 0], rows[:, 1], np.tile(rows[:, 2], (draws, 1)),
                            np.tile(rows[:, 3], (draws, 1)))  # fmt: skip
    totals = count_season_totals(matches, 3, np.zeros(3))
    by_penalties = rank(totals, matches, TieBreakRule(("points", "penalties_if_last_match",
                                                       "shared"), True),
                        np.random.default_rng(4))  # fmt: skip
    assert (by_penalties[:, 0] == 0).mean() == pytest.approx(0.5, abs=0.04)
    earlier = SeasonMatches(rows[[2, 0, 1], 0], rows[[2, 0, 1], 1],
                            np.tile(rows[[2, 0, 1], 2], (draws, 1)),
                            np.tile(rows[[2, 0, 1], 3], (draws, 1)))  # fmt: skip
    not_last = rank(count_season_totals(earlier, 3, np.zeros(3)), earlier,
                    TieBreakRule(("points", "penalties_if_last_match", "shared"), True),
                    np.random.default_rng(4))  # fmt: skip
    assert (not_last[:, 0] == 0).all() and (not_last[:, 1] == 0).all()  # shared instead
    # a list of its own for two teams level
    two = TieBreakRule(("points", "lots"), True, ("points", "shared"))
    shared = rank(totals, matches, two, np.random.default_rng(5))
    assert (shared[:, 0] == shared[:, 1]).all()
