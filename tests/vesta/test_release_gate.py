"""The release read from the repository on a fresh PC: the gate from the latest release plan,
registered once in an empty local log, no git; the manifest says whether the code is the
release's."""

import sqlite3

from vesta.data_format import VERSION, create_statements
from vesta.forecast_log.store import open_log
from vesta.release_gate import (
    check_manifest,
    read_release_version,
    register_release_gate,
    write_manifest,
)


def _dataset(codes):
    connection = sqlite3.connect(":memory:")
    for statement in create_statements():
        connection.execute(statement)
    connection.execute("INSERT INTO format_version VALUES (?, '11footy', 'x')", (VERSION,))
    connection.executemany("INSERT INTO competitions VALUES (?, ?, 'league', 'Europe/Rome')",
                           list(enumerate(codes, start=1)))  # fmt: skip
    return connection


def test_an_empty_log_takes_the_release_for_the_competitions_the_files_hold(tmp_path):
    log = open_log(tmp_path / "log.db")
    only_serie_a = _dataset(["SA"])
    version, pairs = register_release_gate(log, only_serie_a, "2026-11-08T00:00:00Z")
    assert version == f"vesta-{read_release_version()}"
    assert ("M1.result", 1) in pairs and {competition for _, competition in pairs} == {1}
    # a later run with one more file: its competition joins, nothing is registered twice
    more = _dataset(["SA", "PL"])
    _, later = register_release_gate(log, more, "2026-11-09T00:00:00Z")
    assert ("M1.result", 2) in later and set(pairs) <= set(later)
    assert log.execute("SELECT count(*) FROM model_versions").fetchone()[0] == 1


def test_the_manifest_says_release_or_modified(tmp_path):
    for folder in ("src/vesta", "config/vesta"):
        (tmp_path / folder).mkdir(parents=True)
    (tmp_path / "src/vesta/model.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "config/vesta/spine.toml").write_text("a = 1\n", encoding="utf-8")
    assert check_manifest(tmp_path)[0] == "modified"  # no manifest: not a released copy
    write_manifest(tmp_path)
    assert check_manifest(tmp_path)[0] == "release"
    (tmp_path / "src/vesta/model.py").write_bytes(b"x = 1\r\n")  # other line endings
    assert check_manifest(tmp_path)[0] == "release"
    (tmp_path / "src/vesta/extra.py").write_text("y = 2\n", encoding="utf-8")
    assert check_manifest(tmp_path)[0] == "modified"
