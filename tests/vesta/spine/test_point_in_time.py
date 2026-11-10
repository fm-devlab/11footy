"""The times of a match and the point-in-time engine."""

import tomllib
from pathlib import Path

import pytest

from vesta.spine.point_in_time import Fact, Family, ForecastRequest, replay
from vesta.spine.timing import TimeRules, find_kickoff_bounds, find_result_known_at, forecast_time

ROOT = Path(__file__).resolve().parents[3]
RULES = TimeRules.from_config(
    tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
)


def test_a_trusted_clock_gives_one_kickoff_and_the_result_after_the_delay():
    assert find_kickoff_bounds("2026-10-24T17:30:00Z", "minute", "2026-10-24", RULES) == (
        "2026-10-24T17:30:00Z", "2026-10-24T17:30:00Z",
    )  # fmt: skip
    assert find_result_known_at("2026-10-24T17:30:00Z", "minute", "2026-10-24", RULES) == (
        "2026-10-24T19:45:00Z"
    )
    assert forecast_time("2026-10-24T17:30:00Z", "minute", "2026-10-24", RULES) == (
        "2026-10-24T17:30:00Z"
    )


def test_a_date_only_match_is_forecast_at_its_earliest_and_known_after_its_latest():
    assert forecast_time("2016-05-14T17:30:00Z", "date", "2016-05-14", RULES) == (
        "2016-05-14T09:00:00Z"
    )
    assert find_result_known_at("2016-05-14T17:30:00Z", "date", "2016-05-14", RULES) == (
        "2016-05-15T01:45:00Z"
    )


def test_a_kickoff_precision_without_a_rule_is_refused():
    with pytest.raises(ValueError, match="no timing rule"):
        forecast_time("2026-10-24T17:30:00Z", "estimated", "2026-10-24", RULES)


class GoalsSoFar(Family):
    """The goals each team has scored in the facts known so far."""

    name = "goals_so_far"
    columns = ("home_goals_so_far", "away_goals_so_far")

    def make_empty_state(self) -> dict[int, int]:
        return {}

    def learn(self, state: dict[int, int], fact: Fact) -> None:
        state[fact.home_team_id] = state.get(fact.home_team_id, 0) + fact.home_goals
        state[fact.away_team_id] = state.get(fact.away_team_id, 0) + fact.away_goals

    def build_features(self, state: dict[int, int], request: ForecastRequest) -> dict[str, float]:
        return {
            "home_goals_so_far": state.get(request.home_team_id, 0),
            "away_goals_so_far": state.get(request.away_team_id, 0),
        }


class GoalsEverywhere(GoalsSoFar):
    name = "goals_everywhere"
    scope = "all"
    columns = ("home_goals_everywhere", "away_goals_everywhere")

    def build_features(self, state: dict[int, int], request: ForecastRequest) -> dict[str, float]:
        return {
            "home_goals_everywhere": state.get(request.home_team_id, 0),
            "away_goals_everywhere": state.get(request.away_team_id, 0),
        }


def fact(match_id: int, competition: str, home: int, away: int, goals: tuple[int, int],
         known_at: str) -> Fact:  # fmt: skip
    return Fact(match_id, competition, 2026, home, away, goals[0], goals[1], known_at)


def request(match_id: int, competition: str, home: int, away: int, at: str) -> ForecastRequest:
    return ForecastRequest(match_id, competition, 2026, home, away, at)


def test_a_request_sees_the_facts_known_at_or_before_it_and_never_after():
    facts = [
        fact(1, "PL", 10, 20, (2, 1), "2026-10-24T14:15:00Z"),
        fact(2, "PL", 10, 30, (3, 0), "2026-10-24T19:45:00Z"),  # known as the next kicks off
        fact(3, "PL", 10, 40, (5, 0), "2026-10-24T22:00:00Z"),  # known after
    ]
    rows = replay([GoalsSoFar()], facts, [request(9, "PL", 10, 50, "2026-10-24T19:45:00Z")])
    assert rows[0].features == {"home_goals_so_far": 5, "away_goals_so_far": 0}
    assert rows[0].latest_known_at == {"goals_so_far": "2026-10-24T19:45:00Z"}


def test_a_competition_family_keeps_competitions_apart_and_an_all_family_does_not():
    facts = [fact(1, "UCL", 10, 20, (4, 0), "2026-10-22T21:15:00Z")]
    rows = replay([GoalsSoFar(), GoalsEverywhere()], facts,
                  [request(9, "PL", 10, 50, "2026-10-24T14:00:00Z")])  # fmt: skip
    assert rows[0].features["home_goals_so_far"] == 0
    assert rows[0].features["home_goals_everywhere"] == 4
    assert rows[0].latest_known_at == {"goals_so_far": None,
                                       "goals_everywhere": "2026-10-22T21:15:00Z"}  # fmt: skip


def test_requests_come_out_in_the_order_of_their_instants():
    requests = [request(2, "PL", 1, 2, "2026-10-25T14:00:00Z"),
                request(1, "PL", 3, 4, "2026-10-24T14:00:00Z")]  # fmt: skip
    assert [row.request.match_id for row in replay([GoalsSoFar()], [], requests)] == [1, 2]


def test_a_family_that_returns_other_columns_is_refused():
    class Wrong(GoalsSoFar):
        name = "wrong"
        columns = ("something_else",)

    with pytest.raises(ValueError, match="returned"):
        replay([Wrong()], [], [request(1, "PL", 1, 2, "2026-10-24T14:00:00Z")])


def test_two_families_with_one_name_are_refused():
    with pytest.raises(ValueError, match="share a name"):
        replay([GoalsSoFar(), GoalsSoFar()], [], [])
