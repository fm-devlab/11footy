"""The season verdict's pieces that need no model: the checkpoint, the final places with a
play-off between two teams of the league, the baselines' table and their probabilities."""

import numpy as np

from vesta.season.league_verdict import (
    Game,
    LeagueYear,
    _from_places,
    _score_ranked_probability,
    build_table_at,
    find_checkpoint_instant,
    rank_final_places,
    simulate_points_per_game,
)


def _year(deciders=()) -> LeagueYear:
    teams = (1, 2, 3, 4)
    games, day = [], 1
    for home in teams:
        for away in teams:
            if home != away:  # the lower id wins at home, draws away
                goals = (2, 0) if home < away else (1, 1)
                games.append(Game(len(games) + 1, home, away, *goals,
                                  f"2017-08-{day:02d}T15:00:00Z", f"2017-08-{day:02d}T17:00:00Z"))
                day += len(games) % 2 == 0
    return LeagueYear("PL", 2019, teams, tuple(games), {}, deciders)


def test_the_checkpoint_is_the_end_of_the_day_of_the_last_match_of_its_rounds():
    year = _year()
    assert find_checkpoint_instant(year.games, 4, 1) == "2017-08-01T23:59:59Z"
    assert find_checkpoint_instant(year.games, 4, 3) == "2017-08-03T23:59:59Z"


def test_the_table_at_an_instant_counts_only_the_results_known():
    points, played, place = build_table_at(_year(), "2017-08-01T23:59:59Z")
    assert played.sum() == 4 and place[0] == 1


def test_a_play_off_between_two_teams_of_the_league_settles_their_places():
    plain = rank_final_places(_year(), 1)
    assert plain == {1: 1, 2: 2, 3: 3, 4: 4}
    settled = rank_final_places(LeagueYear("PL", 2020, (1, 2, 3, 4),
                                           _year().games, {}, ((4, 3),)), 1)  # fmt: skip
    assert settled[4] == 3 and settled[3] == 4


def test_the_baselines_give_lines_that_add_up():
    found = simulate_points_per_game(
        np.array([20.0, 10.0, 5.0, 1.0]), np.full(4, 10.0), 30, 1.4, 10, 1.8, 2000, 1
    )
    assert np.isclose(found["title"].sum(), 1) and np.isclose(found["top_four"].sum(), 4)
    assert found["title"][0] > 0.9
    uniform = _from_places(np.tile(np.arange(4), (4, 1)).T, 4)
    assert np.allclose(uniform["places"], 0.25)
    assert _score_ranked_probability(np.eye(4), np.arange(4)).tolist() == [0.0] * 4


def test_a_rule_carried_forward_is_publishable_only_where_a_decision_names_it(tmp_path):
    import pytest

    from vesta.season.standings import read_tie_break_rule

    rules = tmp_path / "rules.csv"
    rules.write_text(
        "competition,first_season,last_season,criteria,criteria_two,verified,source\n"
        'PL,2026,2026,"points|goal_difference|play_off=1,17|shared",,carried_forward,"d"\n'
        'PL,2027,2027,"points|goal_difference|shared",,carried_forward,"no decision"\n'
        'SA,2026,2026,"points|lots",,no,"unread"\n', encoding="utf-8")  # fmt: skip
    carried = read_tie_break_rule("PL", 2026, rules)
    assert carried.carried_forward and carried.publishable and not carried.verified
    assert not read_tie_break_rule("SA", 2026, rules).publishable
    with pytest.raises(ValueError, match="without a decision"):
        read_tie_break_rule("PL", 2027, rules)
