"""What the spine reads of the football world: the league matches with their score on the pitch.

The world (`docs/DATA_MODEL.md`) is opened read-only; Vesta never changes it. A match's score is
the one played on the pitch (`home_goals_on_pitch` when a ruling awarded another result,
otherwise `home_goals`): the model forecasts the play, not the rulings (decision of 23 October
2026). Matches called off, abandoned or not yet played have no result to learn from.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from vesta.spine.cups import ClubMatch
from vesta.spine.nations import NationMatch

RESULT_STATUSES = ("played", "awarded")


@dataclass(frozen=True)
class LeagueMatch:
    match_id: int
    competition: str
    season: int
    home_team_id: int
    away_team_id: int
    kickoff_utc: str
    kickoff_precision: str
    local_date: str
    home_goals: int
    away_goals: int


def open_world(path: Path) -> sqlite3.Connection:
    """The world, read-only."""
    if not path.is_file():
        raise FileNotFoundError(
            f"no football world at {path}: build it with `python -m data build`"
        )
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def read_league_matches(
    connection: sqlite3.Connection, competitions: list[str], first_season: int, last_season: int
) -> list[LeagueMatch]:
    """The finished matches of `competitions` from `first_season` to `last_season`, in the order
    of their kick-off."""
    marks = ", ".join("?" for _ in competitions)
    statuses = ", ".join("?" for _ in RESULT_STATUSES)
    rows = connection.execute(
        f"""
        SELECT m.match_id, c.code, s.season, m.home_team_id, m.away_team_id, m.kickoff_utc,
               m.kickoff_precision, m.local_date,
               COALESCE(m.home_goals_on_pitch, m.home_goals),
               COALESCE(m.away_goals_on_pitch, m.away_goals)
        FROM matches m
        JOIN seasons s ON s.season_id = m.season_id
        JOIN competitions c ON c.competition_id = s.competition_id
        WHERE c.code IN ({marks}) AND s.season BETWEEN ? AND ?
          AND m.status IN ({statuses})
          AND COALESCE(m.home_goals_on_pitch, m.home_goals) IS NOT NULL
        ORDER BY m.kickoff_utc, m.match_id
        """,
        (*competitions, first_season, last_season, *RESULT_STATUSES),
    ).fetchall()
    return [LeagueMatch(*row) for row in rows]


def read_match_players(
    connection: sqlite3.Connection, match_ids: set[int]
) -> dict[int, tuple[tuple[int, int, float, bool, bool], ...]]:
    """Per match, who played: (side 0 home or 1 away, player id, minutes, started, kept goal),
    every player who stayed on the pitch for at least a minute."""
    players: dict[int, list[tuple[int, int, float, bool, bool]]] = {}
    for match_id, side, player_id, minutes, started, goalkeeper in connection.execute(
        """
        SELECT p.match_id, p.team_id <> m.home_team_id, p.player_id, p.minutes_played,
               p.started, p.position_group = 'goalkeeper'
        FROM match_players p JOIN matches m ON m.match_id = p.match_id
        WHERE p.minutes_played > 0
        ORDER BY p.match_id, p.team_id <> m.home_team_id, p.player_id
        """
    ):
        if match_id in match_ids:
            players.setdefault(match_id, []).append(
                (int(side), int(player_id), float(minutes), bool(started), bool(goalkeeper))
            )
    return {match_id: tuple(rows) for match_id, rows in players.items()}


def read_nation_matches(connection: sqlite3.Connection) -> list[NationMatch]:
    """Every finished match between two national teams, with the kind of its competition."""
    statuses = ", ".join("?" for _ in RESULT_STATUSES)
    rows = connection.execute(
        f"""
        SELECT m.match_id, c.code, c.kind, m.local_date, m.home_team_id, m.away_team_id,
               COALESCE(m.home_goals_on_pitch, m.home_goals),
               COALESCE(m.away_goals_on_pitch, m.away_goals), m.neutral_venue
        FROM matches m
        JOIN seasons s ON s.season_id = m.season_id
        JOIN competitions c ON c.competition_id = s.competition_id
        JOIN teams home ON home.team_id = m.home_team_id
        JOIN teams away ON away.team_id = m.away_team_id
        WHERE home.kind = 'nation' AND away.kind = 'nation' AND m.status IN ({statuses})
          AND COALESCE(m.home_goals_on_pitch, m.home_goals) IS NOT NULL
        ORDER BY m.local_date, m.match_id
        """,
        RESULT_STATUSES,
    ).fetchall()
    return [NationMatch(*row[:8], bool(row[8])) for row in rows]


# The score after 90 minutes (the spine's M1-M5, 31 October 2026): the
# regular-time score where extra time was played, else the score on the pitch
NINETY_HOME = ("CASE WHEN m.extra_time_played = 1 AND m.home_goals_regular_time IS NOT NULL"
               " THEN m.home_goals_regular_time ELSE COALESCE(m.home_goals_on_pitch,"
               " m.home_goals) END")
NINETY_AWAY = NINETY_HOME.replace("home_goals", "away_goals")


def read_club_matches(
    connection: sqlite3.Connection,
    leagues: list[str],
    cups: list[str],
    first_league_season: int,
    first_cup_season: int,
    last_season: int,
) -> list[ClubMatch]:
    """The finished matches of the leagues and the cups the cups model rates, with the country
    of their venue: the score after 90 minutes, and the final score (extra time included) the
    Elo may read (`[cups] elo_on`)."""
    league_marks = ", ".join("?" for _ in leagues)
    cup_marks = ", ".join("?" for _ in cups)
    statuses = ", ".join("?" for _ in RESULT_STATUSES)
    rows = connection.execute(
        f"""
        SELECT m.match_id, c.code, c.kind, s.season, COALESCE(m.stage, ''), m.kickoff_utc,
               m.home_team_id, m.away_team_id,
               {NINETY_HOME}, {NINETY_AWAY}, m.neutral_venue, v.country,
               COALESCE(m.home_goals_on_pitch, m.home_goals),
               COALESCE(m.away_goals_on_pitch, m.away_goals), m.leg, m.tie_id
        FROM matches m
        JOIN seasons s ON s.season_id = m.season_id
        JOIN competitions c ON c.competition_id = s.competition_id
        LEFT JOIN venues v ON v.venue_id = m.venue_id
        WHERE ((c.code IN ({league_marks}) AND s.season >= ?)
               OR (c.code IN ({cup_marks}) AND s.season >= ?))
          AND s.season <= ? AND m.status IN ({statuses})
          AND COALESCE(m.home_goals_on_pitch, m.home_goals) IS NOT NULL
        ORDER BY m.kickoff_utc, m.match_id
        """,
        (*leagues, first_league_season, *cups, first_cup_season, last_season, *RESULT_STATUSES),
    ).fetchall()
    return [ClubMatch(*row[:10], bool(row[10]), row[11], *row[12:]) for row in rows]


@dataclass(frozen=True)
class HalfTimeScores:
    """The final and the half-time goals of each side, one entry per match that has both."""

    home_goals: list[int]
    away_goals: list[int]
    home_half_time: list[int]
    away_half_time: list[int]


def read_half_time_scores(
    connection: sqlite3.Connection, competitions: list[str], first_season: int, last_season: int
) -> HalfTimeScores:
    """The matches of `competitions` in the seasons whose half-time score is known (on the pitch:
    an awarded result has no half time of its own and is left out)."""
    marks = ", ".join("?" for _ in competitions)
    rows = connection.execute(
        f"""
        SELECT m.home_goals, m.away_goals, m.home_goals_half_time, m.away_goals_half_time
        FROM matches m
        JOIN seasons s ON s.season_id = m.season_id
        JOIN competitions c ON c.competition_id = s.competition_id
        WHERE c.code IN ({marks}) AND s.season BETWEEN ? AND ? AND m.status = 'played'
          AND m.home_goals IS NOT NULL AND m.home_goals_half_time IS NOT NULL
          AND m.away_goals_half_time IS NOT NULL
        """,
        (*competitions, first_season, last_season),
    ).fetchall()
    columns = [list(column) for column in zip(*rows, strict=True)] if rows else [[], [], [], []]
    return HalfTimeScores(*columns)


def read_goal_methods(
    connection: sqlite3.Connection, competitions: list[str], first_season: int, last_season: int
) -> list[str]:
    """The manner of every goal of the seasons (`vesta.spine.goals_in_time.METHODS`), read from
    the goal events and their shots, in this order of precedence: own goal, penalty, direct free
    kick, header, from outside the box, from open play inside the box. A goal without its shot
    (no place, no body part) has no manner and is left out."""
    marks = ", ".join("?" for _ in competitions)
    methods = []
    for kind, situation, body_part, zone, is_penalty in connection.execute(
        f"""
        SELECT e.kind, sh.situation, sh.body_part, sh.zone, sh.is_penalty
        FROM match_events e
        JOIN matches m ON m.match_id = e.match_id
        JOIN seasons s ON s.season_id = m.season_id
        JOIN competitions c ON c.competition_id = s.competition_id
        LEFT JOIN shots sh ON sh.shot_id = e.shot_id
        WHERE c.code IN ({marks}) AND s.season BETWEEN ? AND ?
          AND e.kind IN ('goal', 'penalty_goal', 'own_goal')
        """,
        (*competitions, first_season, last_season),
    ):
        method = classify_goal_method(kind, situation, body_part, zone, is_penalty)
        if method is not None:
            methods.append(method)
    return methods


def classify_goal_method(
    kind: str, situation: str | None, body_part: str | None, zone: str | None, is_penalty: int
) -> str | None:
    """The manner of a goal, by precedence: own goal, penalty, direct free kick, header, from
    outside the box, from open play inside the box; None for a goal without its shot."""
    if kind == "own_goal":
        return "own_goal"
    if kind == "penalty_goal" or is_penalty:
        return "penalty"
    if situation is None:
        return None
    if situation == "direct_free_kick":
        return "direct_free_kick"
    if body_part == "head":
        return "header"
    if zone == "outside_box":
        return "outside_box"
    return "open_play_inside_box"


@dataclass(frozen=True)
class MatchStory:
    """What happened in a finished match, for the check of the goals in time: the half-time and
    full-time scores and every goal in order as (reported minute, the side it counts for, its
    manner or None)."""

    match_id: int
    home_team_id: int
    away_team_id: int
    half_time: tuple[int, int]
    full_time: tuple[int, int]
    goals: tuple[tuple[int, str, str | None], ...]


def read_match_stories(
    connection: sqlite3.Connection, match_ids: set[int]
) -> dict[int, MatchStory]:
    """The story of each match whose half-time score is known and whose goal events add up to its
    final score (a match whose events miss a goal cannot be checked minute by minute)."""
    goals: dict[int, list[tuple[int, int, str, str | None]]] = {}
    for match_id, sequence, minute, kind, team, related, situation, body_part, zone, penalty in (
        connection.execute(
            """
            SELECT e.match_id, e.sequence, e.minute, e.kind, e.team_id, e.related_team_id,
                   sh.situation, sh.body_part, sh.zone, sh.is_penalty
            FROM match_events e LEFT JOIN shots sh ON sh.shot_id = e.shot_id
            WHERE e.kind IN ('goal', 'penalty_goal', 'own_goal') AND e.minute IS NOT NULL
            """
        )
    ):
        if match_id in match_ids:
            credited = related if kind == "own_goal" else team
            goals.setdefault(match_id, []).append(
                (sequence, minute, credited,
                 classify_goal_method(kind, situation, body_part, zone, penalty))  # fmt: skip
            )
    stories = {}
    for match_id, home, away, home_goals, away_goals, home_half, away_half in connection.execute(
        """
        SELECT match_id, home_team_id, away_team_id, home_goals, away_goals,
               home_goals_half_time, away_goals_half_time
        FROM matches WHERE status = 'played' AND home_goals_half_time IS NOT NULL
        """
    ):
        if match_id not in match_ids:
            continue
        ordered = sorted(goals.get(match_id, []), key=lambda goal: (goal[1], goal[0]))
        sides = tuple(
            (minute, "home" if credited == home else "away", method)
            for _, minute, credited, method in ordered
        )
        counted = (sum(side == "home" for _, side, _ in sides),
                   sum(side == "away" for _, side, _ in sides))  # fmt: skip
        if counted != (home_goals, away_goals):
            continue
        stories[match_id] = MatchStory(match_id, home, away, (home_half, away_half),
                                       (home_goals, away_goals), sides)  # fmt: skip
    return stories
