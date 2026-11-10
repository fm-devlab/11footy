"""Vesta's commands on a dataset, run on this PC as a child process, one at a time.

    python -m vesta.run <dataset file> ... --log <forecast log> [--cache <folder>]

`vesta.run` is Vesta's one command for a user's PC: it takes the dataset's files (the whole file,
or a set of split files with the copy they are joined into), exits 0 when done, 1 when the dataset
fails its format or its checks, 2 when the run fails. Its progress follows the window's contract,
one line on standard output per step:

    PROGRESS <done> <total> <seconds left, or -> <what is being done>

Every other line goes to the window's box of Vesta's lines as it is. A command that fails leaves
the log as it was, since Vesta only appends to it. One command per dataset at a time (a lock
file beside its log); Stop ends it. A successful run's minutes are kept beside the log
(`<log>.runs.json`), the estimate the window gives before the next run.
"""

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from desk.opening import Opened

REPOSITORY = Path(__file__).resolve().parents[2]
VESTA_COMMAND = "vesta.run"
CHECK_FAILED = 1  # vesta.run: the dataset fails its format or its checks


@dataclass(frozen=True)
class Command:
    name: str
    arguments: list[str]


@dataclass(frozen=True)
class Progress:
    done: int
    total: int
    seconds_left: int | None
    doing: str


class AlreadyRunning(Exception):
    """Another command runs on the dataset; the message says what to do, in plain words."""


def find_runs_file(log: Path) -> Path:
    return log.with_name(log.name + ".runs.json")


def is_command_ready() -> bool:
    """Whether this copy of the repository has Vesta's one command yet."""
    return importlib.util.find_spec(VESTA_COMMAND) is not None


def compute_command(opened: Opened) -> Command:
    arguments = ["-m", VESTA_COMMAND, *map(str, opened.paths), "--log", str(opened.log)]
    if opened.cache is not None:
        arguments += ["--cache", str(opened.cache)]
    return Command("Vesta: the forecasts", arguments)


def parse_progress(line: str) -> Progress | None:
    """A progress line read, or None for any other line (one that does not follow the contract
    is shown as it is, never dropped)."""
    parts = line.strip().split(" ", 4)
    if len(parts) < 4 or parts[0] != "PROGRESS":
        return None
    try:
        done, total = int(parts[1]), int(parts[2])
        left = None if parts[3] == "-" else int(float(parts[3]))
    except ValueError:
        return None
    return Progress(done, total, left, parts[4] if len(parts) > 4 else "")


def read_last_minutes(log: Path) -> float | None:
    """How long the last successful run took on this PC, None before the first."""
    file = find_runs_file(log)
    try:
        return float(json.loads(file.read_text(encoding="utf-8"))["minutes"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


class Run:
    """One run of Vesta's command in a background thread, writing to `log`. `say` receives every
    line and `progress` every progress line, from that thread, so the window hands them over to
    its own."""

    def __init__(
        self,
        log: Path,
        command: Command,
        say: Callable[[str], None],
        progress: Callable[[Progress], None] = lambda _: None,
    ) -> None:
        self.log = log
        self.command = command
        self.say = say
        self.progress = progress
        self.lock = log.with_name(log.name + ".lock")
        self.process: subprocess.Popen | None = None
        self.stopped = False
        self.succeeded: bool | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.lock.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise AlreadyRunning(
                "Vesta is already running on this dataset (or a run stopped without cleaning"
                f" up: delete {self.lock} if no window is running)."
            ) from error
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stopped = True
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def _run(self) -> None:
        environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        name = self.command.name
        began = time.monotonic()
        try:
            self.say(name)
            self.process = subprocess.Popen(
                [sys.executable, *self.command.arguments],
                cwd=REPOSITORY,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                # no console window on Windows; the flag is 0 elsewhere
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            for line in self.process.stdout:
                found = parse_progress(line)
                if found is None:
                    self.say("    " + line.rstrip())
                else:
                    self.progress(found)
            code = self.process.wait()
            minutes = (time.monotonic() - began) / 60
            if self.stopped:
                self.say(f"Stopped: {name}. The forecasts already written are kept.")
                self.succeeded = False
            elif code == CHECK_FAILED:
                self.say(f"Not run: {name}. The dataset failed Vesta's checks; see the lines"
                         " above.")  # fmt: skip
                self.succeeded = False
            elif code != 0:
                self.say(f"Failed: {name} (exit {code}, {minutes:.1f} min). The forecasts"
                         " already written are kept.")  # fmt: skip
                self.succeeded = False
            else:
                self.say(f"Done: {name} ({minutes:.1f} min).")
                self.succeeded = True
                find_runs_file(self.log).write_text(
                    json.dumps({"minutes": round(minutes, 1)}), encoding="utf-8"
                )
        finally:
            self.lock.unlink(missing_ok=True)
