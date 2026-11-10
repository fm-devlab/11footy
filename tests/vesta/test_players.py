"""The scorers: the shares add up to the side's goals, the first goal follows the score, a player
with no history gets his group's rates, and the reference reads only his last matches."""

import math

import numpy as np

from vesta.players.model import (
    Appearance,
    GroupRates,
    compute_side_means,
    estimate_own_rates,
    estimate_reference_rates,
    make_poisson_cells,
    normalise_shares,
    split_first_goal,
)
from vesta.spine.score_matrix import Calibration, score_matrices

GROUP = GroupRates(0.2, 0.01, 0.1, 0.15, 0.1, 0.12)


def _appearance(season=2020, minutes=90.0, xg=0.5, goals=1, team_goals=2):
    return Appearance("2020-01-01T12:00:00Z", season, minutes, xg, 0, 0.2, goals, 0, team_goals)


def test_the_pools_expected_goals_add_up_to_the_sides():
    rng = np.random.default_rng(7)
    raw, plays = rng.random(25), rng.random(25)
    side = 1.37
    expected = normalise_shares(raw, plays) * side
    assert abs(float(plays @ expected) - side) < 1e-12
    half = np.full(5, 0.5)  # no weight at all: shared evenly
    assert abs(float(half @ normalise_shares(np.zeros(5), half)) * side - side) < 1e-12


def test_the_first_goal_and_the_sides_goals_follow_the_matrix():
    matrix = score_matrices(np.array([1.6]), np.array([0.9]), Calibration())[0]
    home, away = split_first_goal(matrix)
    assert abs(home + away + matrix[0, 0] - matrix.sum()) < 1e-12
    assert home > away
    means = compute_side_means(matrix)
    assert abs(means[0] - 1.6) < 0.05 and abs(means[1] - 0.9) < 0.05
    one_nil = np.zeros((3, 3))
    one_nil[1, 0] = 1.0
    assert split_first_goal(one_nil) == (1.0, 0.0)
    one_all = np.zeros((3, 3))
    one_all[1, 1] = 1.0
    assert split_first_goal(one_all) == (0.5, 0.5)


def test_a_player_without_history_takes_his_groups_rates():
    assert estimate_own_rates([], 2020, GROUP) == (0.2, 0.01, 0.1)
    assert estimate_reference_rates([], GROUP) == (0.15, 0.1, 0.12)


def test_the_rates_read_two_seasons_and_shrink_to_the_group():
    old = [_appearance(season=2017, xg=5.0)] * 10  # three seasons back: not read
    recent = [_appearance(season=2019, xg=0.9)] * 40
    xg, _, _ = estimate_own_rates(old + recent, 2020, GROUP)
    nineties = sum(0.97 ** age for age in range(40))  # each match back weighs less
    assert abs(xg - (0.9 * nineties + 5 * 0.2) / (nineties + 5)) < 1e-12
    goals, _, share = estimate_reference_rates(
        [_appearance(goals=0)] * 30 + [_appearance(goals=1)] * 20, GROUP
    )  # the last 20 only: a goal each
    assert abs(goals - (20 + 5 * 0.15) / 25) < 1e-12
    assert abs(share - (20 + 5 * 0.12) / (40 + 5)) < 1e-12


def test_the_goals_cells_are_poisson_with_the_tail_folded():
    cells = make_poisson_cells(0.4)
    assert abs(cells.sum() - 1) < 1e-12
    assert abs(cells[0] - math.exp(-0.4)) < 1e-12
    assert abs(cells[3] - (1 - sum(math.exp(-0.4) * 0.4 ** k / math.factorial(k)
                                   for k in range(3)))) < 1e-12


def test_a_sides_published_scorers_add_up_and_are_whole_distributions():
    from vesta.distributions import check_distribution
    from vesta.lineups.publication import TeamLineup
    from vesta.players.publication import KEYS, PlayersOfTheDay

    day = PlayersOfTheDay.__new__(PlayersOfTheDay)
    day.groups = {None: GROUP, "forward": GroupRates(0.45, 0.05, 0.15, 0.3, 0.1, 0.25)}
    day.leagues = {"PL": {"own_goal": 0.03, "assisted": 0.72, "conversion": 0.78,
                          "home_first": 0.42, "away_first": 0.33}}  # fmt: skip
    day.histories, day.known_at = {}, {}
    day.uncovered, day.held = {}, []
    day.season, day.now = 2026, "2026-10-31T05:00:00Z"
    pool = tuple((player, "forward" if player < 3 else None, 0.8, 0.9, 80.0)
                 for player in range(14))  # fmt: skip
    lineup = TeamLineup(7, [], "4-4-2", 0.4, [], pool)
    matrix = score_matrices(np.array([1.6]), np.array([0.9]), Calibration())[0]
    found = day.build_side_distributions(lineup, matrix, True, "PL")
    assert [item.quantity_key for item in found[:5]] == list(KEYS)
    kinds = {"P3.goals": "count"}
    for item in found:
        check_distribution(kinds.get(item.quantity_key, "binary"), item.values)
    goals = np.array([-math.log(1 - item.values["yes"]) for item in found
                      if item.quantity_key == "P3.scores"])  # fmt: skip
    side = compute_side_means(matrix)[0] * (1 - 0.03)
    assert abs(float(np.full(14, 0.9) @ goals) - side) < 1e-9
    assert goals[0] > goals[5]  # a forward's rate above the others'
    assert (
        day.build_side_distributions(lineup, matrix, True, "XX") == []
    )  # a league without rates publishes none


def test_a_side_waits_when_a_players_last_match_has_no_tracked_shots(monkeypatch):
    from types import SimpleNamespace

    from vesta.lineups.publication import TeamLineup
    from vesta.players import publication
    from vesta.players.publication import (
        PlayersOfTheDay,
        count_trailing_without_shots,
        find_uncovered_last_matches,
    )

    def match(match_id, kickoff):
        return SimpleNamespace(match_id=match_id, kickoff_utc=kickoff, home_team_id=7,
                               away_team_id=8)

    matches = [match(1, "2026-10-25T14:00:00Z"), match(2, "2026-11-01T14:00:00Z")]
    sheets = SimpleNamespace(sheets={
        (1, 7): {10: (True, 90, "F"), 11: (True, 90, "M")},
        (2, 7): {10: (True, 90, "F"), 12: (False, 0, "M")},  # 12 on the bench: not played
        (2, 8): {20: (False, 15, "D")}})  # fmt: skip
    assert find_uncovered_last_matches(matches, sheets, {1, 2}) == {}
    assert find_uncovered_last_matches(matches, sheets, {1}) == {10: 2, 20: 2}  # 11's last is 1
    day = PlayersOfTheDay.__new__(PlayersOfTheDay)
    day.groups = {None: GROUP}
    day.leagues = {"PL": {"own_goal": 0.03, "assisted": 0.72, "conversion": 0.78,
                          "home_first": 0.42, "away_first": 0.33}}  # fmt: skip
    day.histories, day.known_at, day.held, day.stale = {}, {}, [], set()
    day.uncovered = find_uncovered_last_matches(matches, sheets, {1})
    day.without_shots = count_trailing_without_shots(matches, {1})
    assert day.without_shots == {7: 1, 8: 1} and count_trailing_without_shots(matches, {2}) == {
        7: 0,
        8: 0,
    }
    day.season, day.now = 2026, "2026-11-02T06:00:00Z"
    matrix = score_matrices(np.array([1.6]), np.array([0.9]), Calibration())[0]
    pool = ((10, None, 0.9, 0.95, 85.0), (11, None, 0.8, 0.9, 80.0))
    monkeypatch.setattr(publication, "STALE_MATCHES", 0)  # out longer than the check allows
    assert (
        day.build_side_distributions(TeamLineup(7, [], "4-4-2", 0.4, [], pool), matrix, True, "PL")
        == []
    )
    assert day.held == [("PL", 7, [2])]  # the whole side waits, the match named
    covered = ((11, None, 0.8, 0.9, 80.0),)
    assert (
        len(
            day.build_side_distributions(
                TeamLineup(7, [], "4-4-2", 0.4, [], covered), matrix, True, "PL"
            )
        )
        == 5
    )
    assert day.stale == set()
    # while the source is out no more than the check allows: published, the side marked stale
    monkeypatch.setattr(publication, "STALE_MATCHES", 1)
    assert (
        len(
            day.build_side_distributions(
                TeamLineup(7, [], "4-4-2", 0.4, [], pool), matrix, True, "PL"
            )
        )
        == 10
    )
    assert day.stale == {7} and day.held == [("PL", 7, [2])]


def test_a_match_without_shots_is_left_out_of_his_own_rates():
    from vesta.players.model import Appearance, estimate_own_rates

    played = [Appearance("2026-10-01", 2026, 90.0, 0.6, 0, 0.2, 1, 0, 2),
              Appearance("2026-10-08", 2026, 90.0, 0.0, 0, 0.0, 0, 0, 1, covered=False)]
    assert estimate_own_rates(played, 2026, GROUP) == estimate_own_rates(played[:1], 2026, GROUP)
    assert estimate_own_rates(played, 2026, GROUP) != estimate_own_rates(
        [played[0], Appearance("2026-10-08", 2026, 90.0, 0.0, 0, 0.0, 0, 0, 1)], 2026, GROUP)
