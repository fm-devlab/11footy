"""The league clubs' streaks, next k matches and 30 days (T3, T12, T13) as the season job
publishes them: from the job's simulated seasons, only what the order of the matches allows."""

import numpy as np

from vesta.jobs.fixtures import Fixture
from vesta.quantities import load_quantities
from vesta.season.league import SeasonDraws
from vesta.season.league_windows import (
    KEYS,
    T13,
    CupGame,
    build_window_distributions,
    find_window_dates,
    is_publishable,
)
from vesta.season.standings import SeasonMatches
from vesta.spine.score_matrix import SIZE

NOW = "2026-11-01T08:00:00Z"
TEAMS = (1, 2, 3, 4)
VERSIONS = {key: "v" for key in KEYS}


def _fixture(match_id: int, home: int, away: int, kickoff: str) -> Fixture:
    return Fixture(match_id, "PL", "league", 2026, home, away, kickoff, False, "")


def _draws(played, fixtures, drawn: list[tuple[list[int], list[int]]]) -> SeasonDraws:
    """The simulated seasons: the played matches broadcast, then each fixture's (home goals,
    away goals) over the draws."""
    count = len(drawn[0][0])
    home_goals = np.array([[row[2]] * count for row in played] + [home for home, _ in drawn]).T
    away_goals = np.array([[row[3]] * count for row in played] + [away for _, away in drawn]).T
    column = {team: index for index, team in enumerate(TEAMS)}
    homes = [row[0] for row in played] + [f.home_team_id for f in fixtures]
    aways = [row[1] for row in played] + [f.away_team_id for f in fixtures]
    matches = SeasonMatches(np.array([column[team] for team in homes]),
                            np.array([column[team] for team in aways]), home_goals, away_goals)
    zeros = np.zeros((count, len(TEAMS)), dtype=np.int64)
    return SeasonDraws(zeros, zeros, zeros, zeros, matches)


def _by_key(published, team):
    return {(row.quantity_key, row.window_key): row for row in published if row.subject_id == team}


PLAYED = ((1, 2, 1, 0), (3, 4, 0, 2), (2, 3, 1, 1), (4, 1, 3, 0), (1, 3, 0, 1), (2, 4, 2, 0))
LISTED = [_fixture(11, 1, 4, "2026-11-02T15:00:00Z"), _fixture(12, 3, 2, "2026-11-03T15:00:00Z"),
          _fixture(13, 2, 1, "2026-11-09T15:00:00Z"), _fixture(14, 4, 3, "2026-11-10T15:00:00Z"),
          _fixture(15, 3, 1, "2026-11-16T15:00:00Z"), _fixture(16, 4, 2, "2026-11-17T15:00:00Z")]
# two draws per match: team 1 wins every match it plays in the first, loses every one in the second
DRAWN = [([2, 0], [0, 1]), ([1, 1], [1, 1]), ([0, 1], [1, 0]), ([0, 0], [0, 0]),
         ([0, 1], [1, 0]), ([1, 1], [0, 0])]


def test_the_next_matches_and_the_firsts_from_the_simulated_seasons():
    published = build_window_distributions(
        TEAMS,
        PLAYED,
        LISTED,
        _draws(PLAYED, LISTED, DRAWN),
        NOW,
        "Europe/London",
        VERSIONS,
        load_quantities(),
    )
    own = _by_key(published, 1)
    wins = own[("T12.wins.next3", "next_matches=3")].values
    assert wins["probabilities"] == [0.5, 0.0, 0.0, 0.5]
    assert own[("T12.goals.next3", "next_matches=3")].values["probabilities"] == [0.5, 0.0, 0.0,
                                                                                  0.0, 0.5]
    first_loss = dict(own[("T12.next_loss", "next_matches=10")].values)
    # three matches left: counted to the season's end, not to ten
    assert first_loss == {"1": 0.5, "2": 0.0, "3": 0.0, "none": 0.5}
    # a window running past the season's end is not published
    assert ("T12.wins.next5", "next_matches=5") not in own
    # team 1's played results: W L L; drawn W W W or L L L
    streak = own[("T3.longest_winning_streak", "")]
    assert streak.values["probabilities"] == [0.0, 0.5, 0.0, 0.5]
    assert "over=3.5" in streak.decided_lines  # at most three in a row is still possible
    three = own[("T3.loses_three_in_a_row", "")]
    assert three.values == {"yes": 0.5} and three.decided_lines == ()


def test_only_the_keys_the_gate_allows():
    published = build_window_distributions(
        TEAMS,
        PLAYED,
        LISTED,
        _draws(PLAYED, LISTED, DRAWN),
        NOW,
        "Europe/London",
        {"T12.points.next3": "v2"},
        load_quantities(),
    )
    assert {row.quantity_key for row in published} == {"T12.points.next3"}
    assert {row.model_version for row in published} == {"v2"}


def test_an_unknown_order_holds_what_depends_on_it():
    # team 1's last match is not listed yet: its next three are known, its runs are not
    fixtures = [*LISTED[:4], _fixture(15, 3, 1, "2026-11-16T15:00:00Z"),
                _fixture(-1, 4, 2, "2026-11-17T15:00:00Z")]
    fixtures[4] = _fixture(-2, 3, 1, "2026-12-31T15:00:00Z")
    published = build_window_distributions(
        TEAMS,
        PLAYED,
        fixtures,
        _draws(PLAYED, fixtures, DRAWN),
        NOW,
        "Europe/London",
        VERSIONS,
        load_quantities(),
    )
    own = _by_key(published, 1)
    assert ("T12.wins.next3", "next_matches=3") not in own  # only two listed
    assert ("T3.longest_winning_streak", "") not in own
    assert ("T12.next_loss", "next_matches=10") not in own
    # a match postponed (its kick-off passed without a result): no window for its teams
    postponed = [_fixture(11, 1, 4, "2026-10-30T15:00:00Z"), *LISTED[1:]]
    published = build_window_distributions(
        TEAMS,
        PLAYED,
        postponed,
        _draws(PLAYED, postponed, DRAWN),
        NOW,
        "Europe/London",
        VERSIONS,
        load_quantities(),
    )
    assert not any(key.startswith(("T12", "T13")) for key, _ in _by_key(published, 4))
    assert ("T12.wins.next3", "next_matches=3") in _by_key(published, 3)


def test_three_defeats_already_in_a_row_are_decided():
    played = ((2, 1, 1, 0), (1, 3, 0, 1), (4, 1, 2, 0), *PLAYED[3:])
    published = build_window_distributions(
        TEAMS,
        played,
        LISTED,
        _draws(played, LISTED, DRAWN),
        NOW,
        "Europe/London",
        VERSIONS,
        load_quantities(),
    )
    three = _by_key(published, 1)[("T3.loses_three_in_a_row", "")]
    assert three.values == {"yes": 1.0} and set(three.decided_lines) == {"yes", "no"}


def test_the_thirty_days_add_the_uefa_matches():
    matrix = np.zeros((SIZE, SIZE))
    matrix[2, 0] = 1.0  # the club at home scores two
    published = build_window_distributions(
        TEAMS, PLAYED, LISTED, _draws(PLAYED, LISTED, DRAWN), NOW, "Europe/London", {T13: "v"},
        load_quantities(), {1: [CupGame("2026-11-05T20:00:00Z", True, matrix)]},
        np.random.default_rng(1))  # fmt: skip
    own = _by_key(published, 1)
    days = own[(T13, "2026-11-01/2026-12-01")]
    # its league goals in the 30 days: 2 + 1 + 1 in the first draw, none in the second; plus the
    # two of the UEFA match in both
    league = np.array([4, 0])
    expected = np.bincount(league + 2) / 2
    assert days.values["probabilities"] == expected.tolist()


def test_published_from_five_rounds_and_the_window_in_local_dates():
    assert not is_publishable(20, 49) and is_publishable(20, 50)
    end, key = find_window_dates("2026-03-28T23:30:00Z", "Europe/London")
    assert end == "2026-04-27T23:30:00Z" and key == "2026-03-28/2026-04-28"
