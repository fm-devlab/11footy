"""The football world's tables as Vesta's tests build them: `world_schema.sql` beside this file,
so the tests run without the data package (the public Vesta repository has none).

The file is written from the world's own schema by `tests/data/vesta_world_schema.py`, which
also checks, where the data package is present, that it is still what the world gives.
"""

import sqlite3
from pathlib import Path

FIXTURE = Path(__file__).with_name("world_schema.sql")


def create_world_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(FIXTURE.read_text(encoding="utf-8"))
