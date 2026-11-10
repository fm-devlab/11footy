"""When goals are scored within a match, read from past matches: the timing behind goals in time
(M7, M8, M10-M13), fitted on the seasons of `[timing]` in `config/vesta/spine.toml`.

The clock follows the match reports in minute slots: the first half has slots 1-45 and its added
time 46-55, the second 46-90 and its added time 91-105.

- `read_goal_history`: per finished match, each side's goals by slot, kept only when the goal
  events add up to the score (an own goal counts for the side it benefits).
- `fit_profile`: the share of a match's goals in each slot. The regular minutes come from every
  season given, as a moving mean of `smoothing_minutes` inside each half (the first two minutes
  and the last raw). The added-time slots come from the recent seasons only, since added time
  has grown, scaled to the level of every season.
- `effective_added_minutes`: per half, how many minutes of play at the regular pace carry the
  goals of its added time. Goals in time spreads that share of a half's goals into added time.

Moved from the live chain's package when live forecasting left the product (8 November 2026); the
numbers are those the released verdicts were judged on.
"""

import sqlite3
from dataclasses import dataclass, fields

import numpy as np

from vesta.spine.world import NINETY_AWAY, NINETY_HOME

FIRST_HALF_ADDED = 10  # slots 46-55
SECOND_HALF_ADDED = 15  # slots 91-105
SLOT_HALF = np.array([1] * (45 + FIRST_HALF_ADDED) + [2] * (45 + SECOND_HALF_ADDED))
SLOT_MINUTE = np.concatenate([np.arange(1, 46 + FIRST_HALF_ADDED),
                              np.arange(46, 91 + SECOND_HALF_ADDED)])  # fmt: skip
SLOT_ADDED = ((SLOT_HALF == 1) & (SLOT_MINUTE > 45)) | ((SLOT_HALF == 2) & (SLOT_MINUTE > 90))
SLOTS = len(SLOT_MINUTE)  # 115
SECOND_HALF_FIRST_SLOT = 45 + FIRST_HALF_ADDED
GOAL_KINDS = ("goal", "penalty_goal", "own_goal")
SMALLEST_RATE = 1e-9  # a floor that keeps a division by an empty rate finite


def find_slot(half: int, minute: int) -> int:
    """The slot of an event at `minute` of the reports in `half`."""
    if half == 1:
        return int(min(max(minute, 1), 45 + FIRST_HALF_ADDED)) - 1
    return SECOND_HALF_FIRST_SLOT + int(min(max(minute, 46), 90 + SECOND_HALF_ADDED)) - 46


@dataclass(frozen=True)
class GoalHistory:
    """Arrays over n matches, one row each."""

    match_id: np.ndarray
    season: np.ndarray
    competition: np.ndarray
    kickoff_utc: np.ndarray
    home_goals: np.ndarray  # (n, SLOTS) goals of the home side by slot
    away_goals: np.ndarray

    def __len__(self) -> int:
        return len(self.match_id)

    def select(self, chosen: np.ndarray) -> "GoalHistory":
        return GoalHistory(**{field.name: getattr(self, field.name)[chosen]
                              for field in fields(self)})  # fmt: skip


def read_goal_history(connection: sqlite3.Connection, competitions: list[str], first_season: int,
                      last_season: int, ninety_minutes: tuple[str, ...] = ()) -> GoalHistory:
    """The finished matches of the seasons whose goal events in the two halves add up to their
    score. A competition in `ninety_minutes` (the cups) is checked against its 90-minute score,
    so a match with a goal in extra time is kept; every other one against its final score."""
    marks = ", ".join("?" for _ in competitions)
    matches = connection.execute(
        f"""
        SELECT m.match_id, s.season, c.code, m.kickoff_utc, m.home_team_id, m.away_team_id,
               m.home_goals, m.away_goals, {NINETY_HOME}, {NINETY_AWAY}
        FROM matches m
        JOIN seasons s ON s.season_id = m.season_id
        JOIN competitions c ON c.competition_id = s.competition_id
        WHERE c.code IN ({marks}) AND s.season BETWEEN ? AND ? AND m.status = 'played'
          AND m.home_goals IS NOT NULL
        ORDER BY m.kickoff_utc, m.match_id
        """,
        (*competitions, first_season, last_season),
    ).fetchall()
    index = {row[0]: position for position, row in enumerate(matches)}
    goals = np.zeros((2, len(matches), SLOTS))
    kinds = ", ".join("?" for _ in GOAL_KINDS)
    for match_id, period, minute, added, kind, team, related in connection.execute(
            f"SELECT match_id, period, minute, added_minute, kind, team_id, related_team_id"
            f" FROM match_events WHERE kind IN ({kinds}) AND minute IS NOT NULL"
            f" AND period IN (1, 2)", GOAL_KINDS):  # fmt: skip
        row = index.get(match_id)
        if row is None:
            continue
        credited = related if kind == "own_goal" else team
        side = 0 if credited == matches[row][4] else 1
        goals[side, row, find_slot(period, minute + (added or 0))] += 1
    ninety = [row[2] in ninety_minutes for row in matches]
    adds_up = (goals[0].sum(axis=1) == [row[8] if at else row[6] for row, at in
                                        zip(matches, ninety, strict=True)]) & (
        goals[1].sum(axis=1) == [row[9] if at else row[7] for row, at in
                                 zip(matches, ninety, strict=True)]
    )  # fmt: skip
    history = GoalHistory(
        match_id=np.array([row[0] for row in matches], dtype="int64"),
        season=np.array([row[1] for row in matches], dtype="int64"),
        competition=np.array([row[2] for row in matches], dtype=object),
        kickoff_utc=np.array([row[3] for row in matches], dtype=object),
        home_goals=goals[0],
        away_goals=goals[1],
    )
    return history.select(adds_up)


@dataclass(frozen=True)
class TimingSeasons:
    """`[timing]`: the first season fitted, how many recent seasons the added time is read
    from, the moving mean of the profile's regular minutes."""

    first_season: int
    recent_seasons: int
    smoothing_minutes: int

    @classmethod
    def from_config(cls, spine: dict) -> "TimingSeasons":
        section = spine["timing"]
        return cls(int(section["first_season"]), int(section["recent_seasons"]),
                   int(section["smoothing_minutes"]))  # fmt: skip

    def find_recent_seasons(self, season: int) -> tuple[int, int]:
        """The seasons the added time is read from for `season`."""
        return max(self.first_season, season - self.recent_seasons), season - 1


def fit_profile(history: GoalHistory, recent: GoalHistory, smoothing_minutes: int) -> np.ndarray:
    """The share of a match's goals in each slot, from the goal minutes of `history` (regular
    time) and of `recent` (added time)."""
    every, late = _average_goal_rates(history), _average_goal_rates(recent)
    regular = ~SLOT_ADDED
    rate = np.zeros(SLOTS)
    for half in (1, 2):
        slots = np.flatnonzero(regular & np.equal(SLOT_HALF, half))
        smoothed = _smooth_by_moving_mean(every[slots], smoothing_minutes)
        rate[slots[2:-1]] = smoothed[2:-1]
        rate[slots[:2]], rate[slots[-1]] = every[slots[:2]], every[slots[-1]]
    level = every[regular].mean() / max(late[regular].mean(), SMALLEST_RATE)
    for half in (1, 2):
        added = np.flatnonzero(SLOT_ADDED & np.equal(SLOT_HALF, half))
        rate[added] = late[added] * level
    return rate / rate.sum()


def measure_added_minutes(profile: np.ndarray) -> tuple[float, float]:
    """Per half, its added-time slots' goals over the goals of one regular minute of that half:
    the minutes of added time at the regular pace."""
    minutes = []
    for half in (1, 2):
        regular = profile[np.equal(SLOT_HALF, half) & ~SLOT_ADDED].mean()
        added = profile[np.equal(SLOT_HALF, half) & SLOT_ADDED].sum()
        minutes.append(float(added / regular) if regular > 0 else 0.0)
    return minutes[0], minutes[1]


def _average_goal_rates(history: GoalHistory) -> np.ndarray:
    return (history.home_goals + history.away_goals).sum(axis=0) / max(len(history), 1)


def _smooth_by_moving_mean(values: np.ndarray, width: int) -> np.ndarray:
    smoothed = values.copy()
    half = width // 2
    for index in range(half, len(values) - half):
        smoothed[index] = values[index - half : index + half + 1].mean()
    return smoothed
