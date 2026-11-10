"""The shots as the expected-goals model reads them, from the football world.

Each shot keeps its own facts (place, situation, body part, the action before, minute, result) and
gains what its match says about the moment: the shooter's side plays at home, the score from the
shooter's side before the shot's minute, and the move (set up by a team-mate; minutes since the
side's previous shot, ten when none in the last ten; that shot saved, blocked or off the post a
minute ago or less, a rebound; the side's shots in the same minute, a scramble). Columns of numpy
arrays, one row per shot, in the order of the world's shot ids within each match.
"""

import sqlite3
from dataclasses import dataclass, fields

import numpy as np

GOAL_KINDS = ("goal", "penalty_goal", "own_goal")
STOPPED = ("saved", "blocked", "woodwork")  # a shot whose ball stayed in play near the goal
NO_PREVIOUS_SHOT_MINUTES = 10.0
# the order of a world's shots: the football facts first, the id only as the last tie-break, so
# no result hangs on the ids' values (keyed ids in the data, 9 November)
SHOT_ORDER = ("sh.match_id, sh.period, sh.minute, COALESCE(sh.added_minute, 0), sh.team_id,"
              " sh.player_id, sh.x, sh.y, sh.result, sh.situation, sh.body_part, sh.shot_id")


@dataclass(frozen=True)
class ShotTable:
    """One entry per shot in every array."""

    match_id: np.ndarray
    season: np.ndarray
    competition: np.ndarray
    x: np.ndarray
    y: np.ndarray
    minute: np.ndarray
    situation: np.ndarray
    body_part: np.ndarray
    last_action: np.ndarray
    result: np.ndarray
    home: np.ndarray  # 1 when the shooter's side plays at home
    state: np.ndarray  # goals for minus against, from the shooter's side, before the minute
    assisted: np.ndarray
    gap: np.ndarray
    after_stop: np.ndarray
    same_minute: np.ndarray

    @property
    def goal(self) -> np.ndarray:
        return (self.result == "goal").astype(int)

    def __len__(self) -> int:
        return len(self.match_id)

    def select(self, chosen: np.ndarray) -> "ShotTable":
        """The shots where `chosen` (a boolean mask or indexes) holds."""
        return ShotTable(
            **{field.name: getattr(self, field.name)[chosen] for field in fields(self)}
        )


def read_shots(
    connection: sqlite3.Connection,
    competitions: list[str],
    first_season: int,
    last_season: int,
    layers: list[str],
) -> ShotTable:
    """The shots of `competitions` from `first_season` to `last_season` with their context."""
    competition_marks = ", ".join("?" for _ in competitions)
    layer_marks = ", ".join("?" for _ in layers)
    rows = connection.execute(
        f"""
        SELECT sh.match_id, s.season, c.code, sh.x, sh.y, sh.minute, sh.situation, sh.body_part,
               COALESCE(sh.last_action, 'None'), sh.result,
               sh.team_id = m.home_team_id, sh.team_id, sh.assist_player_id IS NOT NULL
        FROM shots sh
        JOIN matches m ON m.match_id = sh.match_id
        JOIN seasons s ON s.season_id = m.season_id
        JOIN competitions c ON c.competition_id = s.competition_id
        WHERE c.code IN ({competition_marks}) AND s.season BETWEEN ? AND ?
          AND sh.layer IN ({layer_marks}) AND sh.x IS NOT NULL AND sh.y IS NOT NULL
        ORDER BY {SHOT_ORDER}
        """,
        (*competitions, first_season, last_season, *layers),
    ).fetchall()
    goals = _read_goals_by_match(connection, {row[0] for row in rows})
    columns = list(zip(*rows, strict=True)) if rows else [[]] * 13
    match_id = np.array(columns[0], dtype="int64")
    team_id = np.array(columns[11], dtype="int64")
    minute = np.array(columns[5], dtype="float64")
    result = np.array(columns[9], dtype=object)
    gap, after_stop, same_minute = _compute_move_features(match_id, team_id, minute, result)
    return ShotTable(
        match_id=match_id,
        season=np.array(columns[1], dtype="int64"),
        competition=np.array(columns[2], dtype=object),
        x=np.array(columns[3], dtype="float64"),
        y=np.array(columns[4], dtype="float64"),
        minute=minute,
        situation=np.array(columns[6], dtype=object),
        body_part=np.array(columns[7], dtype=object),
        last_action=np.array(columns[8], dtype=object),
        result=result,
        home=np.array(columns[10], dtype="int64"),
        state=_compute_score_states(match_id, team_id, minute, goals),
        assisted=np.array(columns[12], dtype="int64"),
        gap=gap,
        after_stop=after_stop,
        same_minute=same_minute,
    )


def _read_goals_by_match(
    connection: sqlite3.Connection, match_ids: set[int]
) -> dict[int, list[tuple[int, int]]]:
    """Per match, the goals as (the team they count for, minute)."""
    kinds = ", ".join("?" for _ in GOAL_KINDS)
    goals: dict[int, list[tuple[int, int]]] = {}
    for match_id, team_id, minute in connection.execute(
        f"""
        SELECT match_id, CASE WHEN kind = 'own_goal' THEN related_team_id ELSE team_id END,
               minute
        FROM match_events WHERE kind IN ({kinds})
        """,
        GOAL_KINDS,
    ):
        if match_id in match_ids:
            goals.setdefault(match_id, []).append((team_id, minute))
    return goals


def _compute_score_states(
    match_id: np.ndarray,
    team_id: np.ndarray,
    minute: np.ndarray,
    goals: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    """The score of the moment from the shooter's side: the goals of earlier minutes."""
    states = np.zeros(len(match_id), dtype="int64")
    for index, (match, team, at) in enumerate(zip(match_id, team_id, minute, strict=True)):
        for scorer, goal_minute in goals.get(int(match), ()):
            if goal_minute < at:
                states[index] += 1 if scorer == team else -1
    return states


def _compute_move_features(
    match_id: np.ndarray, team_id: np.ndarray, minute: np.ndarray, result: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gap since the side's previous shot, a rebound of a stopped shot, shots in the minute."""
    count = len(match_id)
    gap = np.full(count, NO_PREVIOUS_SHOT_MINUTES)
    after_stop = np.zeros(count, dtype="int64")
    same_minute = np.zeros(count, dtype="int64")
    previous: dict[tuple[int, int], int] = {}
    per_minute: dict[tuple[int, int, float], int] = {}
    for index in range(count):
        side = (int(match_id[index]), int(team_id[index]))
        per_minute[(*side, minute[index])] = per_minute.get((*side, minute[index]), 0) + 1
        if side in previous:
            before = previous[side]
            gap[index] = minute[index] - minute[before]
            after_stop[index] = int(result[before] in STOPPED and gap[index] <= 1)
        previous[side] = index
    for index in range(count):
        key = (int(match_id[index]), int(team_id[index]), minute[index])
        same_minute[index] = per_minute[key]
    return gap, after_stop, same_minute
