"""A SQLite file the window reads and never writes: the dataset, or the forecast log beside it.

One read-only connection per file, opened at the first read: the log may not exist yet when the
window opens a dataset, and Vesta's run creates it later. SQLite's write-ahead log lets each read
see what Vesta has committed so far, so the window reads a log while Vesta writes it.
"""

import sqlite3
from pathlib import Path


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._connection: sqlite3.Connection | None = None
        self._columns: dict[str, set[str]] = {}

    def has_file(self) -> bool:
        return self.path.is_file()

    def _open(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = sqlite3.connect(
                f"file:{self.path.as_posix()}?mode=ro", uri=True, check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def read_rows(self, sql: str, *parameters: object) -> list[sqlite3.Row]:
        return self._open().execute(sql, parameters).fetchall()

    def read_first_row(self, sql: str, *parameters: object) -> sqlite3.Row | None:
        return self._open().execute(sql, parameters).fetchone()

    def read_columns(self, table: str) -> set[str]:
        """A table's columns, empty when the file has no such table. Kept once found (a table
        never loses a column); a table not there yet is looked up again, since Vesta's first run
        creates the log's tables."""
        if table not in self._columns:
            found = {row["name"] for row in self.read_rows(f"pragma table_info({table})")}
            if not found:
                return set()
            self._columns[table] = found
        return self._columns[table]

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
