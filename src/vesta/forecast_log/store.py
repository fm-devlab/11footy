"""Opening the forecast log: its schema, its append-only guard, its write-ahead mode.

The log cannot be rebuilt, so its path always comes from the caller (the command's argument, a
temporary file in the tests), never from a default inside the code. Vesta is the only writer; a
reader such as the desktop window opens it read-only (`open_for_reading`), and the two work at
once because the file runs in SQLite's write-ahead mode (`docs/FORECAST_LOG.md` §14).
"""

import re
import sqlite3
from pathlib import Path

SCHEMA_FILE = Path(__file__).with_name("schema.sql")

# The schema's version, kept in SQLite's user_version: 0 is an empty file.
SCHEMA_VERSION = 12

# What each version adds to the one before, applied in order to a log of an earlier version when
# the writer opens it (a column added is never a rewrite: the rows and their triggers stay).
MIGRATIONS = {
    2: [  # the season page says when last season's tie-break rule was used (28 October 2026)
        "ALTER TABLE publications ADD COLUMN tie_break_carried_forward INTEGER"
        " CHECK (tie_break_carried_forward IS NULL"
        " OR (tie_break_carried_forward IN (0, 1) AND stage = 'season_day'))",
    ],
    3: [],  # the live minutes' series as rows: the table of schema.sql, below
    4: [],  # the likeliest outcome's hits per competition season: the table of schema.sql
    5: [],  # a publication's caveats, its changes and the inputs they compare: the tables below
    6: [],  # a caveat code more (`suspensions_from_cards`): the caveats' table rebuilt, below
    7: [  # the half-time break marked on its live publication (3 November 2026)
        "ALTER TABLE publications ADD COLUMN at_break INTEGER"
        " CHECK (at_break IS NULL OR (at_break IN (0, 1) AND stage = 'live'))",
    ],
    8: [],  # a caveat code more (`shots_stale`): the caveats' table rebuilt, below
    9: [],  # the scores of every version together (model_version 'all'): rebuilt, below
    10: [],  # the caveat of a dataset no verdict covers (VDF S1): rebuilt, below
    11: [],  # the caveat of a fixture forecast before a team's previous match: rebuilt
    12: [],  # a forecast made after kick-off from the round's start: two tables rebuilt
}

# The tables a version adds whole (from schema.sql, with their indexes and triggers).
ADDED_TABLES = {3: ("live_headlines",), 4: ("track_record_hits",),
                5: ("publication_caveats", "publication_changes", "publication_inputs")}

# The tables a version rebuilds with their rows kept (a CHECK widened, which SQLite cannot alter):
# the table of schema.sql, the rows copied, the indexes and triggers made again.
REBUILT_TABLES = {6: ("publication_caveats",), 8: ("publication_caveats",),
                  9: ("track_record_scores",), 10: ("publication_caveats",),
                  11: ("publication_caveats",), 12: ("publications", "publication_caveats")}

# The only columns a row may receive after it is written, once, while still null: the date a
# model version was superseded, the date a quantity stopped being published.
CLOSING_COLUMNS = {"model_versions": "superseded_from", "published_quantities": "published_until"}

BUSY_SECONDS = 60.0


def _build_added_table(name: str) -> list[str]:
    """A table of schema.sql added by a migration: its statement, its indexes and its
    append-only triggers."""
    schema = SCHEMA_FILE.read_text(encoding="utf-8")
    table = re.search(rf"CREATE TABLE {name} \(.*?\n\);", schema, flags=re.S).group(0)
    indexes = re.findall(rf"CREATE INDEX \w+\s+ON {name} \(.*?\);", schema, flags=re.S)
    return [table, *indexes, *_build_append_only_triggers(table).split("\n")]


def _build_rebuild_steps(name: str) -> list[str]:
    """A table rebuilt as schema.sql has it now, every row kept. The renaming leaves the other
    tables' references and the views naming the table as they are (`legacy_alter_table`), so
    after the rebuild they point at the new table, not at the dropped one."""
    table, *rest = _build_added_table(name)
    return ["PRAGMA legacy_alter_table = ON", f"ALTER TABLE {name} RENAME TO {name}_before",
            f"DROP TRIGGER IF EXISTS {name}_no_delete", f"DROP TRIGGER IF EXISTS {name}_no_update",
            table, f"INSERT INTO {name} SELECT * FROM {name}_before", f"DROP TABLE {name}_before",
            *rest, "PRAGMA legacy_alter_table = OFF"]  # fmt: skip


class ForecastLogError(RuntimeError):
    """The log refused a write, or the file is not a forecast log of this schema."""


def open_log(path: Path) -> sqlite3.Connection:
    """The writer's connection: the schema created on an empty file, write-ahead mode, every
    commit flushed to disk (the log is the track record), foreign keys enforced."""
    # two of Vesta's jobs may write the log one after the other: a writer waits for another's
    # short transaction instead of failing
    connection = sqlite3.connect(path, isolation_level=None, timeout=BUSY_SECONDS)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version == 0:
        _create(connection)
    elif version < SCHEMA_VERSION:
        _back_up_before_upgrade(connection, Path(path))
        _migrate(connection, version)
    elif version != SCHEMA_VERSION:
        connection.close()
        raise ForecastLogError(
            f"{path}: schema version {version}, this code writes {SCHEMA_VERSION}"
        )
    return connection


def open_for_reading(path: Path) -> sqlite3.Connection:
    """A read-only connection, as a reader holds it: it can never write the log."""
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def _back_up_before_upgrade(connection: sqlite3.Connection, path: Path) -> Path:
    """A copy of the log as it is, beside it, before it is upgraded in place
    (`forecast-log.db.before-schema-<n>`): the log cannot be rebuilt. SQLite's backup, so the
    write-ahead file's rows are in it; an existing backup is never overwritten."""
    backup = path.with_name(f"{path.name}.before-schema-{SCHEMA_VERSION}")
    if not backup.exists():
        target = sqlite3.connect(backup)
        try:
            connection.backup(target)
        finally:
            target.close()
    return backup


def _migrate(connection: sqlite3.Connection, version: int) -> None:
    """The steps from `version` to this code's, in one transaction. A rebuilt table is dropped and
    made again while other tables point at it, so the foreign keys are off during the
    transaction (SQLite's way to change a table's definition; the pragma cannot change inside
    one) and every reference is checked before it commits."""
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute("BEGIN IMMEDIATE")
    try:
        for step in range(version + 1, SCHEMA_VERSION + 1):
            added = [
                line for name in ADDED_TABLES.get(step, ()) for line in _build_added_table(name)
            ]
            added += [
                line for name in REBUILT_TABLES.get(step, ()) for line in _build_rebuild_steps(name)
            ]
            for statement in MIGRATIONS[step] or added:
                connection.execute(statement)
        broken = connection.execute("PRAGMA foreign_key_check").fetchall()
        if broken:
            raise ForecastLogError(f"the migration would leave {len(broken)} broken references,"
                                   f" e.g. {broken[0]}")  # fmt: skip
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    except BaseException:
        connection.execute("ROLLBACK")
        connection.execute("PRAGMA foreign_keys = ON")
        raise
    connection.execute("COMMIT")
    connection.execute("PRAGMA foreign_keys = ON")


def _create(connection: sqlite3.Connection) -> None:
    schema = SCHEMA_FILE.read_text(encoding="utf-8")
    connection.executescript("BEGIN;" + schema + _build_append_only_triggers(schema))
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    connection.execute("COMMIT")


def _build_append_only_triggers(schema: str) -> str:
    """A trigger per table refusing every delete and every update, except the closing column
    written once while it is null."""
    scratch = sqlite3.connect(":memory:")
    scratch.executescript(schema)
    tables = [
        row[0]
        for row in scratch.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    ]
    triggers = []
    for table in tables:
        refuse = f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END;"
        triggers.append(f"CREATE TRIGGER {table}_no_delete BEFORE DELETE ON {table} {refuse}")
        closing = CLOSING_COLUMNS.get(table)
        if closing is None:
            triggers.append(f"CREATE TRIGGER {table}_no_update BEFORE UPDATE ON {table} {refuse}")
            continue
        columns = [row[1] for row in scratch.execute(f"PRAGMA table_info({table})")]
        changed = " OR ".join(
            f"NEW.{column} IS NOT OLD.{column}" for column in columns if column != closing
        )
        triggers.append(
            f"CREATE TRIGGER {table}_no_update BEFORE UPDATE ON {table} "
            f"WHEN OLD.{closing} IS NOT NULL OR {changed} {refuse}"
        )
    scratch.close()
    return "\n".join(triggers)
