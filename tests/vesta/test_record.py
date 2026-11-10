"""The public track record (`vesta.record`): every line of every forecast of record of Vesta
2.x, nothing selected, Vesta v1 left out, calibration the only summary, the same bytes again."""

import gzip
import math
import sqlite3

from sample_log import build
from tests.vesta.test_datasets import _file

from vesta.record import CALIBRATION, COLUMNS, MARKERS, build_summary, run


def test_the_record_is_every_line_of_vesta_two_and_the_same_bytes_again(tmp_path):
    log = build(tmp_path / "forecast-log.db")
    dataset = _file(tmp_path / "data/league.db", 1, [(1, "01")])
    source = sqlite3.connect(log)
    expected = source.execute(
        "SELECT count(*) FROM probabilities pr JOIN forecasts_of_record r"
        " ON r.publication_id = pr.publication_id AND r.quantity_key = pr.quantity_key"
        " AND r.subject_id = pr.subject_id JOIN publications p"
        " ON p.publication_id = r.publication_id"
        " WHERE p.stage = 'pre_match' AND pr.model_version LIKE 'vesta-2.%'").fetchone()[0]
    source.close()
    assert expected > 0

    written = run(log, [dataset], tmp_path / "out", tmp_path / "assets", tmp_path / "cache")
    record = sqlite3.connect(tmp_path / "assets/track-record.db")
    assert written == expected
    assert record.execute("SELECT count(*) FROM forecast_lines").fetchone()[0] == expected
    versions = {row[0] for row in record.execute("SELECT model_version FROM forecast_lines")}
    assert versions and all(version.startswith("vesta-2.") for version in versions)
    counted = record.execute("SELECT count(*) FROM forecast_lines WHERE state = 'counted'")
    assert record.execute("SELECT sum(count) FROM calibration WHERE probability_kind = 'all'"
                          ).fetchone()[0] == counted.fetchone()[0]  # fmt: skip
    tables = {row[0] for row in record.execute("SELECT name FROM sqlite_master")}
    assert not {"track_record_scores", "track_record_key_scores"} & tables  # no comparison
    record.close()

    with gzip.open(tmp_path / "assets/track-record.csv.gz", "rt", encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    assert header == [name for name, _, _ in COLUMNS]
    readme = (tmp_path / "out/README.md").read_text(encoding="utf-8")
    assert all(f"| {name} |" in readme for name, _, _ in COLUMNS + CALIBRATION)

    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        "README.md", "calibration.csv", "summary.md"]
    summary = (tmp_path / "out/summary.md").read_text(encoding="utf-8")
    assert summary.startswith(MARKERS[0]) and summary.endswith(MARKERS[1] + "\n")
    calibration = (tmp_path / "out/calibration.csv").read_text(encoding="utf-8").splitlines()
    assert calibration[0] == ",".join(name for name, _, _ in CALIBRATION) and len(calibration) > 1

    def read_files():
        return {path.relative_to(tmp_path).as_posix(): path.read_bytes()
                for folder in ("out", "assets") for path in (tmp_path / folder).iterdir()}

    first = read_files()
    run(log, [dataset], tmp_path / "out", tmp_path / "assets", tmp_path / "cache")
    assert read_files() == first


def _result_rows(match_id, chances, happened, kickoff):
    """Flat record rows (as `build_rows` gives them) for one counted match-result forecast."""
    return [(match_id, "L", "League", 2026, kickoff, "A", "B", kickoff, "vesta-2.0.0", "M1.result",
             "result", "results", "", f"outcome={line}", chance, "counted",
             int(line == happened), None, None, kickoff, match_id)
            for line, chance in zip(("home", "draw", "away"), chances, strict=True)]


def test_the_summary_scores_the_result_and_shows_the_calibration():
    rows = (_result_rows(1, (0.5, 0.3, 0.2), "home", "2026-10-28T19:45:00Z")
            + _result_rows(2, (0.2, 0.3, 0.5), "draw", "2026-11-08T20:00:00Z"))
    calibration = [("all", 0.2, 0.3, 0.25, 0.0, 4), ("results", 0.2, 0.3, 0.25, 0.0, 4)]
    summary = build_summary(rows, calibration)
    assert "From 28 October 2026 to 8 November 2026" in summary
    assert "**2 matches** played, **6 probabilities**" in summary
    loss = -(math.log(0.5) + math.log(0.3)) / 2
    assert f"log loss **{loss:.3f}**" in summary and "in **50%** of the matches" in summary
    assert summary.count("| 20% to 30% |") == 1  # the 'all' rows only


def test_before_any_match_is_played_the_summary_says_so():
    assert "first numbers come with the first weekly update" in build_summary([], [])
