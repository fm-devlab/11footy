"""A dataset given as several files: each opened alone, then merged once into a local working copy.

Our dataset ships as one file per league, per cup, the nations and the players, each opening alone
and sharing our global ids; it also ships whole ('all'). Vesta takes any set of them:

    paths = dataset_paths(["data/italy", "data/european-cups/champions-league.db"])
    connection, files = open_datasets(paths, cache_dir)

One file is opened read-only as it is. Several are copied into one SQLite file under `cache_dir`,
named by the hash of their sorted hashes and kept until the set changes, so the readers see one
connection exactly as with one file and keep their indexes (attached views would lose them on
joins). The rules of the merge:

- a table keyed by a match (`MATCH_TABLES`) holds each row in one file only: the same key in two
  files is an error naming both (the whole file mixed with split files falls here);
- every other table (competitions, seasons, teams, players...) may repeat a row across files,
  which must then be identical: the same key with another row is an error naming both files;
- a table's columns must be the same in every file that holds it;
- `format_version` keeps one row: the version (one for all), the dataset when every file names
  the same, and the earliest `built_at` (a set mixing builds runs at its oldest state).

A file is anyone's, so its own SQL is never run: each table is read through `PRAGMA table_info`
and created again by our own statement, every identifier quoted. A table or column whose name is
not a plain lower-case identifier, or whose type is not a plain SQLite type, refuses the set.
"""

import hashlib
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from vesta.data_format import VERSION, FormatError, Header, open_dataset

MATCH_TABLES = ("matches", "match_ties", "match_officials", "penalty_shootouts", "match_events",
                "shots", "match_players", "match_team_sheets", "match_team_statistics",
                "player_match_statistics", "player_match_event_counts")  # fmt: skip
INDEXES = ("CREATE INDEX IF NOT EXISTS vesta_matches_season ON matches (season_id)",
           "CREATE INDEX IF NOT EXISTS vesta_matches_kickoff ON matches (kickoff_utc)",
           "CREATE INDEX IF NOT EXISTS vesta_events_match ON match_events (match_id)",
           "CREATE INDEX IF NOT EXISTS vesta_shots_match ON shots (match_id)",
           "CREATE INDEX IF NOT EXISTS vesta_team_statistics_match"
           " ON match_team_statistics (match_id)",
           "CREATE INDEX IF NOT EXISTS vesta_players_match"
           " ON match_players (match_id)")  # fmt: skip


NAME = re.compile(r"[a-z_][a-z0-9_]*")  # a table's or a column's name a set may carry
TYPES = ("", "TEXT", "INTEGER", "INT", "REAL", "NUMERIC", "BLOB")  # the column types it may declare
PLAYERS_SCOPE = "players"  # the players file: every player, no match
OUTPUT_FOLDER = "vesta"  # what `vesta.run` writes beside the data: never read as data
LOG_SUFFIX = ".forecast-log.db"  # one file's log, beside it: never read as data


class MergeError(FormatError):
    """A set of files that cannot be one dataset."""


@dataclass(frozen=True)
class DatasetFile:
    path: Path
    sha256: str
    header: Header
    scope: str | None  # the competitions it holds, as its header says (None: not said)
    codes: frozenset[str] = frozenset()  # the competitions with matches in it


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def find_dataset_paths(arguments: list[str | Path]) -> list[Path]:
    """Each argument a dataset file, or a folder whose `.db` files are all taken (recursively,
    but for Vesta's own output: the folder `vesta/`, with the log and the working copies, and a
    file's log `<file>.forecast-log.db`); sorted, each once."""
    found: set[Path] = set()
    for argument in arguments:
        path = Path(argument)
        if path.is_dir():
            found.update(item.resolve() for item in path.rglob("*.db") if item.is_file()
                         and not item.name.endswith(LOG_SUFFIX)
                         and OUTPUT_FOLDER not in item.relative_to(path).parts[:-1])  # fmt: skip
        elif path.is_file():
            found.add(path.resolve())
        else:
            raise FileNotFoundError(f"no dataset file or folder at {path}")
    if not found:
        raise FileNotFoundError(f"no .db file in {', '.join(map(str, arguments))}")
    return sorted(found)


def describe(path: Path) -> DatasetFile:
    """One file opened alone: its header (refused without one) and its hash."""
    connection, header = open_dataset(path)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(format_version)")}
        scope = (connection.execute("SELECT scope FROM format_version").fetchone()[0]
                 if "scope" in columns else None)  # fmt: skip
        holds_matches = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'matches'").fetchone()
        codes = frozenset() if scope == PLAYERS_SCOPE or not holds_matches else frozenset(
            row[0] for row in connection.execute(
                "SELECT DISTINCT c.code FROM matches m JOIN seasons s ON s.season_id ="
                " m.season_id JOIN competitions c ON c.competition_id = s.competition_id"))
    finally:
        connection.close()
    return DatasetFile(Path(path), hash_file(path), header, scope, codes)


def open_datasets(paths: list[Path], cache_dir: Path,
                  files: list[DatasetFile] | None = None) -> tuple[sqlite3.Connection,
                                                                    list[DatasetFile]]:  # fmt: skip
    """The set as one read-only connection, and each file's description."""
    files = files if files is not None else [describe(path) for path in paths]
    if len(files) == 1:
        connection, _ = open_dataset(files[0].path)
        return connection, files
    key = hashlib.sha256("".join(sorted(item.sha256 for item in files)).encode()).hexdigest()
    target = Path(cache_dir) / f"merged-{key[:16]}.db"
    if not target.exists():
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial")
        if partial.exists():
            partial.unlink()
        try:
            _merge(files, partial)
        except BaseException:
            if partial.exists():
                partial.unlink()
            raise
        os.replace(partial, target)
    connection = sqlite3.connect(f"file:{target.as_posix()}?mode=ro", uri=True)
    return connection, files


def _merge(files: list[DatasetFile], target: Path) -> None:
    datasets = {item.header.dataset for item in files}
    merged = sqlite3.connect(f"file:{Path(target).as_posix()}", uri=True)  # URIs in ATTACH
    try:
        merged.execute("CREATE TABLE format_version (version TEXT, dataset TEXT, built_at TEXT)")
        merged.execute("INSERT INTO format_version VALUES (?, ?, ?)", (
            VERSION, datasets.pop() if len(datasets) == 1 else "mixed",
            min(item.header.built_at for item in files)))  # fmt: skip
        created: dict[str, tuple[list[str], Path]] = {}
        merged.commit()
        for item in files:
            merged.execute("ATTACH DATABASE ? AS part",
                           (f"file:{quote(item.path.as_posix(), safe='/:')}?mode=ro",))
            try:
                for (name,) in merged.execute(
                        "SELECT name FROM part.sqlite_master WHERE type = 'table'"
                        " AND name <> 'format_version' AND name NOT LIKE 'sqlite_%'"
                        ).fetchall():  # fmt: skip
                    columns = _read_columns(merged, name, item.path)
                    names = [column for column, _, _ in columns]
                    if name not in created:
                        merged.execute(_build_create_statement(name, columns))
                        created[name] = (names, item.path)
                    elif names != created[name][0]:
                        raise MergeError(f"{name}: other columns in {item.path} than in"
                                         f" {created[name][1]}")  # fmt: skip
                    _insert_rows_of_part(merged, name, names, item.path, created[name][1])
                merged.commit()
            finally:
                merged.rollback()  # nothing pending on success; a failed copy left out
                merged.execute("DETACH DATABASE part")
        for statement in INDEXES:
            table = statement.split(" ON ")[1].split(" ")[0]
            if table in created:
                merged.execute(statement)
        merged.commit()
    finally:
        merged.close()


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _read_columns(merged: sqlite3.Connection, table: str,
                  path: Path) -> list[tuple[str, str, int]]:  # fmt: skip
    """(name, type, place in the primary key) of each column of the attached file's `table`;
    a name or a type outside the plain ones refuses the file."""
    if not NAME.fullmatch(table):
        raise MergeError(f"{path}: the table name {table!r} is not a plain identifier")
    columns = [(row[1], str(row[2]).upper(), int(row[5]))
               for row in merged.execute(f"PRAGMA part.table_info({_quote(table)})")]
    for column, kind, _ in columns:
        if not NAME.fullmatch(column):
            raise MergeError(f"{path}: {table} has a column named {column!r}, not a plain"
                             " identifier")  # fmt: skip
        if kind not in TYPES:
            raise MergeError(f"{path}: {table}.{column} declares the type {kind!r}")
    return columns


def _build_create_statement(table: str, columns: list[tuple[str, str, int]]) -> str:
    """Our own CREATE TABLE for `columns`, with the file's primary key."""
    body = [f"{_quote(column)} {kind}".rstrip() for column, kind, _ in columns]
    key = [column for column, _, place in sorted(columns, key=lambda item: item[2]) if place]
    if key:
        body.append(f"PRIMARY KEY ({', '.join(map(_quote, key))})")
    return f"CREATE TABLE main.{_quote(table)} ({', '.join(body)})"


def _insert_rows_of_part(merged: sqlite3.Connection, name: str, columns: list[str], path: Path,
                         first: Path) -> None:  # fmt: skip
    table, listed = _quote(name), ", ".join(map(_quote, columns))
    if name in MATCH_TABLES:
        try:
            merged.execute(f"INSERT INTO main.{table} ({listed}) SELECT {listed} FROM part.{table}")
        except sqlite3.IntegrityError as error:
            raise MergeError(f"{name}: a row of {path} is already in another file of the set"
                             " (a match lives in one file; the whole file and split files"
                             " cannot be mixed)") from error  # fmt: skip
        return
    try:  # the rows not yet there; one with a key already there is a different row
        merged.execute(f"INSERT INTO main.{table} ({listed}) SELECT {listed} FROM part.{table}"
                       f" EXCEPT SELECT {listed} FROM main.{table}")  # fmt: skip
    except sqlite3.IntegrityError as error:
        raise MergeError(f"{name}: a row of {path} differs from the row of the same id in"
                         f" another file of the set (the table was first read from {first})"
                         ) from error  # fmt: skip
