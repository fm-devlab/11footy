"""The Vesta data format (VDF 1) as code: the tables and columns Vesta may read, by block.

`docs/VESTA_DATA_FORMAT.md` is the contract; this module is the same contract for the code, so a
reader or a check can name it. Our dataset (repository 2) is the format; a foreign dataset is
normalised into it by its owner. Vesta never reads a column outside these lists, and never a
provider's name or id.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

VERSION = "VDF 1"
OUR_DATASET = "11footy"

# block -> table -> column -> SQLite type affinity
REQUIRED = "required"
BLOCKS: dict[str, dict[str, dict[str, str]]] = {
    REQUIRED: {
        "format_version": {"version": "TEXT", "dataset": "TEXT", "built_at": "TEXT"},
        "competitions": {"competition_id": "INTEGER", "code": "TEXT", "kind": "TEXT",
                         "time_zone": "TEXT"},
        "seasons": {"season_id": "INTEGER", "competition_id": "INTEGER", "season": "INTEGER",
                    "status": "TEXT"},
        "teams": {"team_id": "INTEGER", "kind": "TEXT", "name": "TEXT"},
        "matches": {"match_id": "INTEGER", "season_id": "INTEGER", "kickoff_utc": "TEXT",
                    "kickoff_precision": "TEXT", "local_date": "TEXT", "home_team_id": "INTEGER",
                    "away_team_id": "INTEGER", "neutral_venue": "INTEGER", "status": "TEXT",
                    "home_goals": "INTEGER", "away_goals": "INTEGER",
                    "home_goals_on_pitch": "INTEGER", "away_goals_on_pitch": "INTEGER"},
    },
    "B1": {"matches": {"home_goals_half_time": "INTEGER", "away_goals_half_time": "INTEGER"}},
    "B2": {
        "match_events": {"event_id": "INTEGER", "match_id": "INTEGER", "sequence": "INTEGER",
                         "period": "INTEGER", "minute": "INTEGER", "added_minute": "INTEGER",
                         "kind": "TEXT", "team_id": "INTEGER", "related_team_id": "INTEGER",
                         "player_id": "INTEGER", "related_player_id": "INTEGER",
                         "shot_id": "INTEGER", "layer": "TEXT"},
    },
    "B3": {
        "shots": {"shot_id": "INTEGER", "match_id": "INTEGER", "team_id": "INTEGER",
                  "player_id": "INTEGER", "assist_player_id": "INTEGER", "period": "INTEGER",
                  "minute": "INTEGER", "added_minute": "INTEGER", "x_m": "REAL", "y_m": "REAL",
                  "result": "TEXT", "situation": "TEXT", "body_part": "TEXT",
                  "is_penalty": "INTEGER", "zone": "TEXT", "layer": "TEXT", "xg": "REAL"},
    },
    "B4": {
        "match_team_statistics": {"match_id": "INTEGER", "team_id": "INTEGER", "layer": "TEXT",
                                  "shots": "INTEGER", "shots_on_target": "INTEGER",
                                  "corners": "INTEGER", "fouls": "INTEGER",
                                  "offsides": "INTEGER", "yellow_cards": "INTEGER",
                                  "red_cards": "INTEGER"},
    },
    "B5": {
        "match_players": {"match_id": "INTEGER", "team_id": "INTEGER", "player_id": "INTEGER",
                          "started": "INTEGER", "minutes_played": "INTEGER",
                          "position_group": "TEXT", "position_code": "TEXT",
                          "formation_slot": "INTEGER"},
        "match_team_sheets": {"match_id": "INTEGER", "team_id": "INTEGER", "formation": "TEXT"},
    },
    "B6": {
        "player_match_statistics": {"match_id": "INTEGER", "team_id": "INTEGER",
                                    "player_id": "INTEGER", "assists": "INTEGER",
                                    "shots": "INTEGER", "shots_on_target": "INTEGER",
                                    "key_passes": "INTEGER", "fouls_committed": "INTEGER",
                                    "fouls_suffered": "INTEGER", "yellow_cards": "INTEGER",
                                    "offsides": "INTEGER", "saves": "INTEGER"},
    },
    "B7": {
        "matches": {"stage": "TEXT", "group_name": "TEXT", "leg": "INTEGER", "tie_id": "INTEGER",
                    "winner_team_id": "INTEGER", "home_goals_regular_time": "INTEGER",
                    "away_goals_regular_time": "INTEGER", "extra_time_played": "INTEGER",
                    "venue_id": "INTEGER"},
        "venues": {"venue_id": "INTEGER", "country": "TEXT"},
        "match_ties": {"tie_id": "INTEGER", "season_id": "INTEGER",
                       "first_leg_match_id": "INTEGER", "second_leg_match_id": "INTEGER",
                       "winner_team_id": "INTEGER", "decided_by": "TEXT"},
    },
    "B8": {
        "standings": {"season_id": "INTEGER", "after_matchday": "INTEGER", "team_id": "INTEGER",
                      "position": "INTEGER", "points_deducted": "INTEGER"},
    },
    "B9": {
        "suspensions_by_rule": {"player_id": "INTEGER", "competition_id": "INTEGER",
                                "served_in_match_ids": "TEXT"},
    },
}  # fmt: skip

# the closed value sets (a value outside is a format error)
VALUES: dict[tuple[str, str], tuple] = {
    ("competitions", "kind"): ("league", "cup", "cup qualifying", "super cup",
                               "nations qualifying", "nations league", "nations tournament",
                               "friendly"),
    ("teams", "kind"): ("club", "nation"),
    ("matches", "kickoff_precision"): ("minute", "date"),
    ("matches", "neutral_venue"): (0, 1),
    ("matches", "status"): ("played", "awarded", "scheduled", "postponed", "suspended",
                            "cancelled", "abandoned"),
    ("matches", "extra_time_played"): (0, 1),
    ("matches", "leg"): (1, 2, None),
    ("match_events", "period"): (1, 2, 3, 4, 5),
    ("match_events", "layer"): ("summary", "play_by_play", "correction"),
    ("match_team_statistics", "layer"): ("report", "feed"),
    ("shots", "result"): ("goal", "saved", "blocked", "woodwork", "off_target"),
    ("shots", "situation"): ("open_play", "corner", "set_piece", "direct_free_kick", "penalty"),
    ("shots", "body_part"): ("right_foot", "left_foot", "head", "other", "unknown"),
    ("shots", "is_penalty"): (0, 1),
    ("shots", "zone"): ("six_yard_box", "penalty_area", "outside_box", None),
    ("shots", "layer"): ("tracked", "commentary"),
    ("match_players", "started"): (0, 1),
    ("match_players", "position_group"): ("goalkeeper", "defender", "midfielder", "forward",
                                          None),
    ("match_ties", "decided_by"): ("aggregate", "away_goals", "extra_time", "shootout", "ruling",
                                   None),
}  # fmt: skip

# the primary keys of the format's tables
KEYS: dict[str, tuple[str, ...]] = {
    "competitions": ("competition_id",), "seasons": ("season_id",), "teams": ("team_id",),
    "matches": ("match_id",), "match_events": ("event_id",), "shots": ("shot_id",),
    "match_team_statistics": ("match_id", "team_id", "layer"),
    "match_players": ("match_id", "team_id", "player_id"),
    "match_team_sheets": ("match_id", "team_id"),
    "player_match_statistics": ("match_id", "team_id", "player_id"),
    "venues": ("venue_id",), "match_ties": ("tie_id",),
    "standings": ("season_id", "after_matchday", "team_id"),
}  # fmt: skip


def merge_columns(*blocks: str) -> dict[str, dict[str, str]]:
    """The tables and columns of `blocks` together (the required block always included)."""
    merged: dict[str, dict[str, str]] = {}
    for block in (REQUIRED, *blocks):
        for table, names in BLOCKS[block].items():
            merged.setdefault(table, {}).update(names)
    return merged


def create_statements(*blocks: str) -> list[str]:
    """CREATE TABLE statements holding exactly the format's columns of `blocks`: an empty
    dataset in the format, for tests and for the reference of the check."""
    statements = []
    for table, names in merge_columns(*blocks).items():
        body = ", ".join(f"{name} {kind}" for name, kind in names.items())
        key = KEYS.get(table)
        if key:
            body += f", PRIMARY KEY ({', '.join(key)})"
        statements.append(f"CREATE TABLE {table} ({body})")
    return statements


@dataclass(frozen=True)
class Header:
    version: str
    dataset: str
    built_at: str

    @property
    def ours(self) -> bool:
        return self.dataset == OUR_DATASET


class FormatError(ValueError):
    """The dataset is not in the format."""


def open_dataset(path: Path) -> tuple[sqlite3.Connection, Header]:
    """A dataset in the format, read-only, with its header; refused without one."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"no dataset at {path}")
    connection = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT version, dataset, built_at FROM format_version"
        ).fetchall()
    except sqlite3.OperationalError as error:
        connection.close()
        raise FormatError(f"{path}: no format_version table, not a {VERSION} dataset") from error
    except sqlite3.DatabaseError as error:  # not SQLite at all, or damaged
        connection.close()
        raise FormatError(f"{path} is not an SQLite database, or it is damaged ({error})"
                          ) from error  # fmt: skip
    if len(rows) != 1 or rows[0][0] != VERSION:
        connection.close()
        raise FormatError(f"{path}: format_version must hold one row of version {VERSION!r}")
    return connection, Header(*rows[0])


# --- reading our world and a dataset through the same names (VDF S2) ---------------------------
#
# The world and the format both mark each row's `layer` (the kind of record it comes from, never
# who provided it), so one query runs on both and no reader names a provider.

STATISTICS_LAYERS = ("report", "feed")  # the order of record in the leagues (decision C3)
FEED = "feed"  # the match feed's box score: the cups' only count, the stubs, the offsides


def has_column(connection: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row[1] == column for row in connection.execute(f"PRAGMA table_info({table})"))


def build_player_rows_filter(connection: sqlite3.Connection, alias: str = "") -> str:
    """SQL condition keeping the `player_match_statistics` rows the verdicts read: the match
    feed's in the world; every row in a dataset (the format holds one count per player)."""
    if has_column(connection, "player_match_statistics", "layer"):
        return f"{alias + '.' if alias else ''}layer = '{FEED}'"
    return "1 = 1"


NOT_VALIDATED_CAVEAT = "not_validated_on_dataset"


def is_not_validated(header: Header, competition: str, released: set[str]) -> bool:
    """Whether a publication for `competition` carries `not_validated_on_dataset` (forecast log
    schema 10): always on a dataset other than ours; on ours, where no released part covers the
    competition."""
    return not header.ours or competition not in released


def make_dataset_caveats(header: Header, competition: str, released: set[str],
                         kinds: tuple[str, ...], model_version: str) -> list:  # fmt: skip
    """The caveats of one publication's `kinds` for the dataset it was made from."""
    from vesta.forecast_log.writing import Caveat

    if not is_not_validated(header, competition, released):
        return []
    return [Caveat(kind, NOT_VALIDATED_CAVEAT, model_version) for kind in kinds]
