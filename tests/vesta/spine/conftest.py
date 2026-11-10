"""The small world of the spine's tests: two leagues of ten clubs, seasons 2014 to 2017, built
with the world's own schema."""

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from tests.vesta.world_schema import create_world_schema

TEAMS = 10


def small_world(path: Path) -> sqlite3.Connection:
    """Two leagues of ten clubs, seasons 2014 to 2017, each a double round robin played on
    Saturdays at 15:00 UTC, goals drawn from fixed strengths; plus one awarded result and one
    cancelled match."""
    connection = sqlite3.connect(path)
    create_world_schema(connection)
    state = np.random.RandomState(20261024)
    for competition_id, code in ((1, "PL"), (2, "SA")):
        connection.execute(
            "INSERT INTO competitions VALUES (?, ?, ?, 'Testland', 'league', 'UTC', 2014, 1)",
            (competition_id, code, code),
        )
        for club in range(TEAMS):
            connection.execute(
                "INSERT INTO teams (team_id, name, kind) VALUES (?, ?, 'club')",
                (competition_id * 100 + club, f"{code} club {club}"),
            )
        strength = state.normal(0, 0.35, TEAMS)
        for season in range(2014, 2018):
            season_id = competition_id * 100 + season - 2000
            connection.execute(
                "INSERT INTO seasons VALUES (?, ?, ?, ?, NULL, NULL, ?, 'closed', 'league')",
                (season_id, competition_id, season, str(season), TEAMS),
            )
            day = date(season, 8, 9)
            for round_index, pairs in enumerate(_double_round_robin()):
                match_day = day + timedelta(weeks=round_index)
                for home, away in pairs:
                    home_goals = int(state.poisson(np.exp(0.35 + strength[home] - strength[away])))
                    away_goals = int(state.poisson(np.exp(0.05 + strength[away] - strength[home])))
                    connection.execute(
                        "INSERT INTO matches (season_id, kickoff_utc, kickoff_precision,"
                        " local_date, home_team_id, away_team_id, status, home_goals,"
                        " away_goals, source) VALUES (?, ?, 'minute', ?, ?, ?, 'played', ?, ?,"
                        " 'test')",
                        (season_id, f"{match_day.isoformat()}T15:00:00Z", match_day.isoformat(),
                         competition_id * 100 + home, competition_id * 100 + away, home_goals,
                         away_goals),  # fmt: skip
                    )
    connection.execute(
        "UPDATE matches SET status = 'awarded', home_goals_on_pitch = home_goals,"
        " away_goals_on_pitch = away_goals, home_goals = 0, away_goals = 3 WHERE match_id = 5"
    )
    connection.execute(
        "UPDATE matches SET status = 'cancelled', home_goals = NULL, away_goals = NULL"
        " WHERE match_id = 6"
    )
    connection.commit()
    return connection


def _double_round_robin() -> list[list[tuple[int, int]]]:
    """The circle method: every club meets every other once at home and once away."""
    clubs = list(range(TEAMS))
    rounds = []
    for _ in range(TEAMS - 1):
        rounds.append([(clubs[i], clubs[TEAMS - 1 - i]) for i in range(TEAMS // 2)])
        clubs = [clubs[0], clubs[-1], *clubs[1:-1]]
    return rounds + [[(away, home) for home, away in pairs] for pairs in rounds]


@pytest.fixture
def world(tmp_path: Path) -> sqlite3.Connection:
    connection = small_world(tmp_path / "football-world.db")
    yield connection
    connection.close()
