"""Our expected goals as a dataset gives them: `shots.xg`, read as given (S3a).

In our dataset `xg` is written by the weekly build (`vesta.jobs.dataset_xg`) with the
released model, the same chances the world's walk-forward gives; in a foreign dataset it is the
dataset's own, read as it is. On a dataset Vesta never fits the xG trees. The shots read are the
ones with `xg`, in the order of the world's reader (match, then shot id), so the scorers' rows
line up with the chances as they do on the world.
"""

import sqlite3
from dataclasses import dataclass, fields

import numpy as np

# `SHOT_ORDER` of `vesta.expected_goals.shots` on the format's columns (x_m = 105 x,
# y_m = 68 (1 - y)), so a dataset's shots come in the world's order
DATASET_SHOT_ORDER = ("sh.match_id, sh.period, sh.minute, COALESCE(sh.added_minute, 0),"
                      " sh.team_id, sh.player_id, sh.x_m, sh.y_m DESC, sh.result, sh.situation,"
                      " sh.body_part, sh.shot_id")


@dataclass(frozen=True)
class RatedShots:
    """One entry per shot with `xg`: what the champion and the scorers read beside the chance."""

    match_id: np.ndarray
    season: np.ndarray
    home: np.ndarray  # 1 when the shooter's side plays at home

    def __len__(self) -> int:
        return len(self.match_id)

    def select(self, chosen: np.ndarray) -> "RatedShots":
        return RatedShots(**{field.name: getattr(self, field.name)[chosen]
                             for field in fields(self)})  # fmt: skip


def is_dataset(connection: sqlite3.Connection) -> bool:
    """Whether the connection is a dataset in the format (a world has no `format_version`)."""
    return connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table'"
                              " AND name = 'format_version'").fetchone() is not None  # fmt: skip


def _select_rated(connection: sqlite3.Connection, columns: str, competitions: list[str],
                  first_season: int, last_season: int) -> list[tuple]:  # fmt: skip
    marks = ", ".join("?" for _ in competitions)
    return connection.execute(
        f"""
        SELECT {columns}
        FROM shots sh
        JOIN matches m ON m.match_id = sh.match_id
        JOIN seasons s ON s.season_id = m.season_id
        JOIN competitions c ON c.competition_id = s.competition_id
        WHERE c.code IN ({marks}) AND s.season BETWEEN ? AND ? AND sh.xg IS NOT NULL
        ORDER BY {DATASET_SHOT_ORDER}
        """, (*competitions, first_season, last_season)).fetchall()  # fmt: skip


def read_rated_shots(connection: sqlite3.Connection, competitions: list[str], first_season: int,
                     last_season: int) -> tuple[RatedShots, np.ndarray]:  # fmt: skip
    """The shots with `xg` of `competitions` from `first_season` to `last_season`, and their
    chances."""
    rows = _select_rated(connection, "sh.match_id, s.season, sh.team_id = m.home_team_id, sh.xg",
                         competitions, first_season, last_season)  # fmt: skip
    columns = list(zip(*rows, strict=True)) if rows else [[]] * 4
    shots = RatedShots(match_id=np.array(columns[0], dtype="int64"),
                       season=np.array(columns[1], dtype="int64"),
                       home=np.array(columns[2], dtype="int64"))  # fmt: skip
    return shots, np.array(columns[3], dtype="float64")


def read_rated_shot_players(connection: sqlite3.Connection, competitions: list[str],
                            first_season: int, last_season: int) -> list[tuple]:  # fmt: skip
    """(shooter, assister, is penalty, team) of every shot `read_rated_shots` reads, in its
    order."""
    return _select_rated(connection, "sh.player_id, sh.assist_player_id,"
                         " COALESCE(sh.is_penalty, 0), sh.team_id", competitions, first_season,
                         last_season)  # fmt: skip
