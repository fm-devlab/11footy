import time
from pathlib import Path

import pytest

from desk.opening import Opened
from desk.run import (
    CHECK_FAILED,
    AlreadyRunning,
    Command,
    Progress,
    Run,
    compute_command,
    parse_progress,
    read_last_minutes,
)


def python_code(code: str) -> Command:
    return Command("fake", ["-c", code])


def finish(run: Run) -> None:
    run.start()
    deadline = time.monotonic() + 60
    while run.is_running() and time.monotonic() < deadline:
        time.sleep(0.05)


def opened(paths: list[Path], log: Path, cache: Path | None) -> Opened:
    return Opened(paths, paths, paths[0], log, cache, "11footy", "2026-11-08", True, [], [])


def test_progress_lines_are_read_and_the_rest_shown():
    found = parse_progress("PROGRESS 3 10 120 Fitting the league")
    assert found == Progress(3, 10, 120, "Fitting the league")
    assert parse_progress("PROGRESS 3 10 - Writing") == Progress(3, 10, None, "Writing")
    assert parse_progress("PROGRESS three 10 - x") is None
    assert parse_progress("Fitted 3 leagues") is None


def test_a_run_reports_progress_and_keeps_its_minutes(tmp_path):
    log = tmp_path / "vesta" / "forecast-log.db"
    said, seen = [], []
    command = python_code("print('PROGRESS 1 2 30 half'); print('hello')")
    run = Run(log, command, said.append, seen.append)
    finish(run)
    assert run.succeeded
    assert seen == [Progress(1, 2, 30, "half")]
    assert "    hello" in said
    assert read_last_minutes(log) is not None
    assert not run.lock.exists()


def test_a_failed_check_or_run_says_so_and_keeps_no_minutes(tmp_path):
    log = tmp_path / "forecast-log.db"
    said = []
    refused = Run(log, python_code(f"raise SystemExit({CHECK_FAILED})"), said.append)
    finish(refused)
    assert refused.succeeded is False
    assert any(line.startswith("Not run") for line in said)
    failed = Run(log, python_code("raise SystemExit(2)"), said.append)
    finish(failed)
    assert failed.succeeded is False
    assert any(line.startswith("Failed") for line in said)
    assert read_last_minutes(log) is None


def test_one_run_per_log(tmp_path):
    log = tmp_path / "forecast-log.db"
    log.with_name("forecast-log.db.lock").write_text("1")
    with pytest.raises(AlreadyRunning, match="already running"):
        Run(log, python_code("pass"), lambda _: None).start()


def test_vesta_takes_every_file_with_the_log_and_the_joined_copy(tmp_path):
    one = compute_command(opened([tmp_path / "d.db"], tmp_path / "d.db.forecast-log.db", None))
    assert one.arguments[2:] == [str(tmp_path / "d.db"), "--log",
                                 str(tmp_path / "d.db.forecast-log.db")]  # fmt: skip
    files = [tmp_path / "a.db", tmp_path / "players.db"]
    cache = tmp_path / "vesta" / "cache"
    several = compute_command(opened(files, tmp_path / "vesta" / "forecast-log.db", cache))
    assert several.arguments[2:4] == [str(path) for path in files]
    assert several.arguments[-2:] == ["--cache", str(cache)]
