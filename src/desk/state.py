"""What the window remembers for the person at this PC: the dataset opened last, as the files
and folders that were chosen; the light or dark look they chose; the window's size and place.

It lives in the user's own folder for applications, never in the repository, so a fresh download
or a `git pull` leaves it alone: `%LOCALAPPDATA%\\11footy\\state.json` on Windows,
`$XDG_STATE_HOME/11footy` (or `~/.local/state/11footy`) elsewhere. A file that cannot be read is
treated as no file: the window then starts as on its first day.
"""

import json
import os
from pathlib import Path


def find_state_file() -> Path:
    """Where the window keeps what it remembers, in the user's folder for applications."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_STATE_HOME")
    folder = Path(base) if base else Path.home() / ".local" / "state"
    return folder / "11footy" / "state.json"


def read_state(file: Path | None = None) -> dict:
    """Everything remembered; nothing when the file is missing or cannot be read."""
    try:
        found = json.loads((file or find_state_file()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return found if isinstance(found, dict) else {}


def keep(values: dict, file: Path | None = None) -> None:
    """Remember these values beside the others."""
    file = file or find_state_file()
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(read_state(file) | values, indent=2), encoding="utf-8")


def read_last_chosen(file: Path | None = None) -> list[Path]:
    """The files and folders chosen last, when they are all still there; none otherwise."""
    try:
        chosen = [Path(path) for path in read_state(file)["chosen"]]
    except (KeyError, TypeError):
        return []
    return chosen if chosen and all(path.exists() for path in chosen) else []


def remember(chosen: list[Path], file: Path | None = None) -> None:
    """Keep what was chosen, to open again when the window starts."""
    keep({"chosen": [str(path) for path in chosen]}, file)
