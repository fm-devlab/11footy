"""What the morning keeps from one day to the next: the past that cannot change.

The pre-match job needs, every morning, the expected goals of every shot (the champion learns
from them) and the champion's forecasts of the past league matches (the matrix calibration and
the live chain's effects are fitted on them). For the seasons before the current one these are
the same every day, so they are computed once and kept here; each morning then rates only the
current season's shots and asks the champion only for the fixtures.

A kept past is used only when both still hold, else it is computed again and replaced:
- the release: the sha256 of every file of `src/vesta` and `config/vesta` (a past kept by
  another version of the model is never reused);
- the inputs: the sha256 of the closed seasons' league matches, their players and their shots as
  read from today's world (a correction of an old season computes the past again).
The file is derived and can be deleted at any time; it is written to a new file, then moved over
the old one. It is plain numbers and text (NumPy's `.npz`, read with `allow_pickle=False`), so a
file planted in a dataset folder can at worst be refused, never run as code.
"""

import hashlib
import zipfile
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from vesta.expected_goals.shots import ShotTable

FROZEN_FOLDERS = ("src/vesta", "config/vesta")


@dataclass(frozen=True)
class KeptPast:
    release: str
    season: int  # the current season: everything kept is of the seasons before it
    inputs: str
    chances: np.ndarray  # the closed seasons' shots, in the order of the shot table
    forecasts: tuple[tuple[int, str, float, float], ...]  # match id, forecast time, intensities


def hash_release(root: Path) -> str:
    """The sha256 of the model's files: every file under `src/vesta` and `config/vesta`, by path
    and content (caches and compiled files left out)."""
    digest = hashlib.sha256()
    for folder in FROZEN_FOLDERS:
        for path in sorted((root / folder).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
            digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()


def hash_inputs(matches: list, players: dict, shots: ShotTable) -> str:
    """The sha256 of the closed seasons' inputs (the caller passes only those)."""
    digest = hashlib.sha256()
    for match in sorted(matches, key=lambda item: item.match_id):
        digest.update(repr(match).encode("utf-8"))
        digest.update(repr(players.get(match.match_id, ())).encode("utf-8"))
    for field in fields(shots):
        column = getattr(shots, field.name)
        digest.update(field.name.encode("utf-8"))
        digest.update(np.asarray(column).astype(str).tobytes() if column.dtype == object
                      else np.ascontiguousarray(column).tobytes())  # fmt: skip
    return digest.hexdigest()


def build_cache_path(folder: Path, season: int) -> Path:
    return folder / f"pre-match-past-{season}.npz"


def read_kept(folder: Path | None, release: str, season: int, inputs: str) -> KeptPast | None:
    """The kept past when it is of this release, this season and these inputs; else None."""
    if folder is None or not build_cache_path(folder, season).exists():
        return None
    try:
        with np.load(build_cache_path(folder, season), allow_pickle=False) as stored:
            kept = KeptPast(
                str(stored["release"]), int(stored["season"]), str(stored["inputs"]),
                np.asarray(stored["chances"], dtype=float),
                tuple(zip(stored["match_ids"].astype(int).tolist(),
                          stored["forecast_times"].astype(str).tolist(),
                          stored["home_intensities"].astype(float).tolist(),
                          stored["away_intensities"].astype(float).tolist(), strict=True)))
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
        return None  # not a kept past of ours: computed again
    if (kept.release, kept.season, kept.inputs) != (release, season, inputs):
        return None
    return kept


def write_kept(folder: Path | None, kept: KeptPast) -> None:
    if folder is None:
        return
    folder.mkdir(parents=True, exist_ok=True)
    path = build_cache_path(folder, kept.season)
    temporary = path.with_name(path.name + ".tmp")
    columns = list(zip(*kept.forecasts, strict=True)) or [(), (), (), ()]
    with temporary.open("wb") as handle:
        np.savez(handle, release=np.str_(kept.release), season=np.int64(kept.season),
                 inputs=np.str_(kept.inputs), chances=np.asarray(kept.chances, dtype=float),
                 match_ids=np.array(columns[0], dtype=np.int64),
                 forecast_times=np.array(columns[1], dtype=str),
                 home_intensities=np.array(columns[2], dtype=float),
                 away_intensities=np.array(columns[3], dtype=float))  # fmt: skip
    temporary.replace(path)
    for other in [*folder.glob("pre-match-past-*.npz"), *folder.glob("pre-match-past-*.pickle")]:
        # an older season's past is spent; a pickle of an earlier version is never read
        if other != path:
            other.unlink()
