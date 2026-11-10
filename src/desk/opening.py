"""Opening a dataset: one file, several files, or a folder of them; unpacked when packed, each file
checked, and a set joined into one copy the screens read.

Our release (`dataset/README.md`), refreshed after each round, ships the split files (one per
league, per European cup, `nations` and `players`, each `<name>.db.xz`) and the whole file
`11footy.db.xz`, with `SHA256SUMS`. A person opens the whole file, a folder of split files, or
the split files they want; a dataset of their own in the Vesta data format opens the same way.

The order, each step refusing with a reason a person can act on:
1. The files: a folder gives every dataset file in it, at any depth, but Vesta's own output
   folder (`vesta/`); a `.db` beside its `.db.xz` is the same file, unpacked.
2. A packed file: enough free disk, its digest checked against the `SHA256SUMS` beside it (or in
   a folder up to three above, as Vesta looks), then unpacked beside itself (to
   `<name>.unpacking`, renamed once whole). A file unpacked before and whole is used again.
3. Every dataset file: its digest checked when `SHA256SUMS` names it, its header read
   (`vesta.data_format.open_dataset`: VDF 1, or refused).
4. The set: the whole file alone or split files, never both (every match would be there twice);
   the split files with the players file, looked for beside them when it was not chosen.
5. Several files are joined by Vesta's own merge (`vesta.datasets.open_datasets`) into one copy
   under `vesta/cache/` beside them; Vesta's run reuses the same copy.
6. The coverage: each competition, the file it came from, and what Vesta can serve it with, by
   Vesta's own capability check on the set (`vesta.capabilities.competitions`); a European cup
   whose leagues are missing names them.

Nothing here writes outside the dataset's own folders, and nothing reaches the network.
"""

import csv
import hashlib
import lzma
import os
import shutil
import sqlite3
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from desk.formats import format_local_moment, parse_instant
from desk.reading import find_log
from vesta.capabilities import _read_released_competitions as released_competitions
from vesta.capabilities import (
    assess_competitions,
    assess_families,
    describe_families,
    find_present_blocks,
    has_player_names,
    validate,
)
from vesta.data_format import OUR_DATASET, FormatError, Header, open_dataset
from vesta.dataset_proof import Proof, prove
from vesta.datasets import OUTPUT_FOLDER, DatasetFile, open_datasets
from vesta.release_gate import check_manifest, find_latest_plan

REPOSITORY = Path(__file__).resolve().parents[2]
SPINE = REPOSITORY / "config" / "vesta" / "spine.toml"
COMPETITION_NAMES = REPOSITORY / "config" / "competitions.csv"
SUMS = "SHA256SUMS"
SUMS_LEVELS = 3  # how many folders above a file its SHA256SUMS may lie (Vesta's rule)
DATABASE = ".db"
PACKED = ".db.xz"
WHOLE_PART = "all"
WHOLE_FILE = OUR_DATASET  # the whole dataset's file in a release: 11footy.db.xz
PLAYERS_PART = "players"
LOG_NAME = "forecast-log.db"  # Vesta's log, beside a dataset or in its folder
# Free space wanted before unpacking, as a multiple of the packed file: SQLite packs about three
# to one with xz; the margin leaves room for the forecast log beside it.
SPACE_FACTOR = 5
STALE_AFTER = timedelta(days=7)
CHUNK = 1 << 20

Progress = Callable[[str, int, int], None]  # what is being done, bytes done, bytes in all


class Refused(Exception):
    """The dataset cannot be opened; the message says why, in words for the person at the PC."""


@dataclass
class CheckedFile:
    path: Path
    part: str  # "italy/serie-a", "players", "all"; the file's name for a dataset of another
    codes: frozenset[str]  # the competitions with matches in it
    sha256: str
    listed: bool  # its digest found in a SHA256SUMS
    proof: Proof  # whether it is ours, by Vesta's rule (`vesta.dataset_proof`)
    players: bool = False  # it holds the players' names (the players file, or the whole file)


@dataclass
class Coverage:
    """One competition of the set: where it came from and what Vesta can do with it."""

    code: str
    name: str
    part: str
    served: bool
    model: str | None
    reason: str
    validation: str | None
    missing: list[str] = field(default_factory=list)  # the leagues a European cup still needs
    families: str = ""  # Vesta's words on the families beyond result and goals


@dataclass
class Opened:
    chosen: list[Path]  # what the person picked: files or folders
    paths: list[Path]  # the dataset files, as `vesta.run` takes them
    data: Path  # what the screens read: the one file, or the joined copy
    log: Path
    cache: Path | None  # the joined copy's folder, None for one file
    name: str
    built_at: str
    ours: bool
    files: list[CheckedFile]
    coverage: list[Coverage]
    notes: list[str] = field(default_factory=list)
    digests_checked: bool = True  # false when reopened: checked when first opened


def ignore_progress(_: str, __: int, ___: int) -> None:
    """The progress of a caller that shows none."""


def find_dataset_files(chosen: list[Path]) -> list[Path]:
    """The dataset files of the chosen files and folders, each once: a folder at any depth but
    Vesta's own output folder; a `.db` beside its `.db.xz` taken once, through the packed file,
    which opens it without unpacking again."""
    found: set[Path] = set()
    for path in map(Path, chosen):
        if path.is_dir():
            found.update(
                item for item in path.rglob("*")
                if item.is_file() and is_dataset_file(item)
                and OUTPUT_FOLDER not in item.relative_to(path).parts[:-1]
            )  # fmt: skip
        elif path.is_file():
            found.add(path)
        else:
            raise Refused(f"No file or folder at {path}.")
    unpacked = {item.with_suffix("") for item in found if item.name.endswith(PACKED)}
    return sorted(item.resolve() for item in found
                  if not (item.suffix == DATABASE and item in unpacked
                          and item.with_name(item.name + ".xz") in found))  # fmt: skip


def is_dataset_file(path: Path) -> bool:
    """A dataset file by its name; never a forecast log (`<file>.forecast-log.db` beside one file
    opened alone, `forecast-log.db` in Vesta's folder), which a folder may hold too."""
    if path.name == LOG_NAME or path.name.endswith("." + LOG_NAME):
        return False
    return path.name.endswith(PACKED) or path.suffix == DATABASE


def find_sums(path: Path) -> Path | None:
    """The SHA256SUMS of a file: beside it, or in a folder up to `SUMS_LEVELS` above."""
    for folder in [path.parent, *path.parent.parents][: SUMS_LEVELS + 1]:
        if (folder / SUMS).is_file():
            return folder / SUMS
    return None


def read_digests(sums: Path | None) -> dict[str, str]:
    """The digests a SHA256SUMS gives, by file name; empty without one."""
    if sums is None:
        return {}
    found = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.strip().partition("  ")
        if digest and name:
            found[Path(name.lstrip("*")).name] = digest.lower()
    return found


def compute_sha256(path: Path, progress: Progress = ignore_progress) -> str:
    total = path.stat().st_size
    digest = hashlib.sha256()
    done = 0
    with path.open("rb") as source:
        while chunk := source.read(CHUNK):
            digest.update(chunk)
            done += len(chunk)
            progress(f"Checking {path.name}", done, total)
    return digest.hexdigest()


def check_digest(path: Path, digest: str) -> bool:
    """True when a SHA256SUMS names the file with this digest; refused when it names it with
    another (a damaged download)."""
    expected = read_digests(find_sums(path)).get(path.name)
    if expected is None:
        return False
    if digest != expected:
        raise Refused(
            f"{path.name} does not match its line in {SUMS}: the file is damaged or not the one"
            " released. Download it again."
        )
    return True


def unpack(packed: Path, progress: Progress = ignore_progress) -> Path:
    """`<name>.db.xz` unpacked beside itself as `<name>.db`."""
    target = packed.with_suffix("")
    free = shutil.disk_usage(packed.parent).free
    wanted = SPACE_FACTOR * packed.stat().st_size
    if free < wanted:
        raise Refused(
            f"Not enough free disk space beside {packed.name}: {free / 1e9:.1f} GB free, about"
            f" {wanted / 1e9:.1f} GB needed. Free some space or move the file."
        )
    building = target.with_name(target.name + ".unpacking")
    total = packed.stat().st_size
    try:
        with packed.open("rb") as raw, lzma.open(raw) as source, building.open("wb") as out:
            while chunk := source.read(CHUNK):
                out.write(chunk)
                progress(f"Unpacking {packed.name}", raw.tell(), total)
    except lzma.LZMAError as error:
        building.unlink(missing_ok=True)
        raise Refused(
            f"{packed.name} is not a whole xz file ({error}). Download it again."
        ) from error
    except BaseException:
        building.unlink(missing_ok=True)
        raise
    os.replace(building, target)
    return target


def prepare_database(path: Path, progress: Progress, check: bool) -> Path:
    """The database of a dataset file: itself, or a packed file checked and unpacked. A file
    unpacked before is used again: an unpacking is renamed into place only once whole, and its
    digest is checked next like any other."""
    if not path.name.endswith(PACKED):
        return path
    database = path.with_suffix("")
    if database.exists():
        return database
    if check:
        check_digest(path, compute_sha256(path, progress))
    return unpack(path, progress)


def read_contents(connection: sqlite3.Connection, path: Path) -> tuple[str, frozenset[str], bool]:
    """A dataset file's part, the competitions with matches in it, and whether it holds the
    players' names."""
    try:
        columns = {row[1] for row in connection.execute("pragma table_info(format_version)")}
        part = (connection.execute("select part from format_version").fetchone()[0]
                if "part" in columns else path.stem)  # fmt: skip
        if part == WHOLE_FILE:
            part = WHOLE_PART  # the whole file of a release before parts were named
        codes = frozenset(row[0] for row in connection.execute(
            "select distinct c.code from matches m join seasons s using (season_id)"
            " join competitions c using (competition_id)"))  # fmt: skip
    except sqlite3.Error:
        part, codes = path.stem, frozenset()  # a players file holds no matches
    try:
        players = connection.execute("select exists (select 1 from players)").fetchone()[0] == 1
    except sqlite3.Error:
        players = False  # a split file of matches: its players' names are in the players file
    return part, codes, players


def check_file(path: Path, progress: Progress, check: bool) -> tuple[CheckedFile, Header]:
    """One dataset file: its digest (refused on a mismatch), its header and its competitions."""
    digest = compute_sha256(path, progress)
    listed = check_digest(path, digest) if check else False
    try:
        connection, header = open_dataset(path)
    except FormatError as error:
        raise Refused(
            f"{path.name} is not a dataset in the Vesta data format: {error}. What a dataset"
            " holds: docs/VESTA_DATA_FORMAT.md, in this program's folder."
        ) from error
    try:
        part, codes, players = read_contents(connection, path)
    finally:
        connection.close()
    proof = prove(DatasetFile(path, digest, header, None, codes))
    return CheckedFile(path, part, codes, digest, listed, proof, players), header


def find_players(chosen: list[Path], files: list[Path]) -> Path | None:
    """The players file beside the chosen files, when they were not chosen with it."""
    folders = {path if path.is_dir() else path.parent for path in map(Path, chosen)}
    folders |= {path.parent for path in files}
    for folder in sorted(folders):
        for name in (PLAYERS_PART + DATABASE, PLAYERS_PART + PACKED):
            if (folder / name).is_file():
                return folder / name
    return None


def open_paths(
    chosen: list[Path],
    progress: Progress = ignore_progress,
    now: datetime | None = None,
    check: bool = True,
) -> Opened:
    """The dataset of the chosen files and folders, unpacked, checked and joined. `check` false
    skips the digests, for files this PC checked when they were first opened."""
    chosen = [Path(path) for path in chosen]
    files = find_dataset_files(chosen)
    if not files:
        raise Refused("No dataset file (.db or .db.xz) in what was chosen.")
    notes: list[str] = []
    whole = [path for path in files if path.name in (WHOLE_FILE + DATABASE, WHOLE_FILE + PACKED)]
    if whole and len(files) > 1 and any(path.is_dir() for path in chosen):
        # a release folder holds the whole file beside the split ones: the same matches, in one
        files = whole[:1]
        notes.append(f"The folder holds the whole dataset: {files[0].name} was opened alone.")
    databases = [prepare_database(path, progress, check) for path in files]
    checked, headers = [], []
    for database in databases:
        found, header = check_file(database, progress, check)
        checked.append(found)
        headers.append(header)
    parts = [item.part for item in checked]
    if WHOLE_PART in parts and len(checked) > 1:
        raise Refused(
            "The whole dataset (11footy) was chosen with split files: every match would be there"
            " twice. Open either the whole file or the split files."
        )
    has_competitions = any(item.codes for item in checked)
    if has_competitions and not any(item.players for item in checked):
        players = find_players(chosen, databases)
        if players is not None and headers[0].dataset == OUR_DATASET:
            database = prepare_database(players, progress, check)
            found, header = check_file(database, progress, check)
            checked.append(found)
            headers.append(header)
            databases.append(database)
        elif headers[0].dataset == OUR_DATASET:
            notes.append(
                "No players file: players show as numbers. Add players.db.xz from the same release."
            )
    if check and any(header.dataset == OUR_DATASET for header in headers):
        unconfirmed = [item.path.name for item in checked if not item.listed]
        if unconfirmed:
            notes.append(
                f"Could not confirm these are 11footy's files: no {SUMS} names"
                f" {', '.join(unconfirmed)}."
            )
    paths = sorted(databases)
    if len(paths) == 1:
        data, log, cache = paths[0], find_log(paths[0]), None
        connection, _ = open_dataset(paths[0])
    else:
        base = chosen[0] if chosen[0].is_dir() else chosen[0].parent
        cache = base / OUTPUT_FOLDER / "cache"
        log = base / OUTPUT_FOLDER / "forecast-log.db"
        progress(f"Joining {len(paths)} files into one", 0, 0)
        try:
            connection, _ = open_datasets(paths, cache, [
                DatasetFile(item.path, item.sha256, header, None, item.codes)
                for item, header in zip(checked, headers, strict=True)
            ])  # fmt: skip
        except FormatError as error:
            raise Refused(f"These files do not join into one dataset: {error}") from error
        data = Path(connection.execute("pragma database_list").fetchone()[2])
    try:
        coverage = describe_coverage(connection, checked, headers)
    finally:
        connection.close()
    built_at = min(header.built_at for header in headers)
    try:
        built = parse_instant(built_at)
    except ValueError:
        built = None  # a date the dataset writes in another form: no age said
    if built is not None and (now or datetime.now(UTC)) - built > STALE_AFTER:
        notes.append(
            f"This dataset was built on {built.date().isoformat()}: matches added since are not"
            " in it."
        )
    names = {header.dataset for header in headers}
    name = next(iter(names)) if len(names) == 1 else "mixed"
    return Opened(chosen, paths, data, log, cache, name, built_at, names == {OUR_DATASET},
                  checked, coverage, notes, check)  # fmt: skip


def describe_coverage(
    connection: sqlite3.Connection, files: list[CheckedFile], headers: list[Header]
) -> list[Coverage]:
    """What Vesta can serve each competition of the set with, as its run would decide: by its
    capability check, the competitions read from files proved ours counting as ours."""
    present = find_present_blocks(connection)
    errors, _ = validate(connection, present)
    if errors:
        raise Refused(
            "The dataset is not in the Vesta data format: "
            + "; ".join(f"{problem.table}: {problem.what}" for problem in errors[:5])
        )
    spine = tomllib.loads(SPINE.read_text(encoding="utf-8"))
    manifest, _ = check_manifest(REPOSITORY)
    codes = set().union(*(item.codes for item in files))
    proved = (
        {code for code in codes
         if all(item.proof.ours for item in files if code in item.codes)}
        if manifest == "release" else set()
    )  # fmt: skip
    found = assess_competitions(
        connection,
        headers[0],
        present,
        spine,
        released_competitions(find_latest_plan(REPOSITORY)),
        proved,
    )
    families = assess_families(found, present, spine, find_latest_plan(REPOSITORY),
                               min(header.built_at for header in headers),
                               has_player_names(connection), proved)  # fmt: skip
    names = read_competition_names(connection)
    built_at = min(header.built_at for header in headers)
    part_of = {code: item.part for item in files for code in item.codes}
    leagues = spine["cups"]["leagues"]
    coverage = []
    for item in found:
        if item.code not in part_of:
            continue  # a competition only named (a league of a cup's teams), with no match here
        missing = []
        if item.code in spine["cups"]["cups"]:
            missing = [names.get(code, code) for code in leagues if code not in part_of]
        coverage.append(Coverage(item.code, names.get(item.code, item.code),
                                 part_of[item.code], item.served, item.model, item.reason,
                                 item.validation, missing,
                                 describe_moment(describe_families(families, item.code),
                                                 built_at)))  # fmt: skip
    return coverage


def describe_moment(said: str, moment: str) -> str:
    """Vesta's words with the dataset's build moment as the window writes moments (Vesta writes
    `2026-11-10T12:21:18Z`)."""
    try:
        return said.replace(moment, format_local_moment(moment))
    except ValueError:
        return said  # a moment the dataset writes in another form: said as it is


def read_competition_names(connection: sqlite3.Connection) -> dict[str, str]:
    """Each competition's name by its code: the set's own, else ours (`config/competitions.csv`),
    so that a league a cup still needs is named though the set does not hold it."""
    with COMPETITION_NAMES.open(encoding="utf-8", newline="") as handle:
        names = {row["code"]: row["name"] for row in csv.DictReader(handle)}
    columns = {row[1] for row in connection.execute("pragma table_info(competitions)")}
    if "name" in columns:
        names.update(connection.execute("select code, name from competitions"))
    return names
