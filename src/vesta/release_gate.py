"""What is released, read from the repository instead of our forecast log: the user's run on a fresh
PC.

- The gate is the latest plan in `config/vesta/releases`. Its codes are resolved to the dataset's
  ids; a code the user's files do not hold is left out.
- The local log registers that release once, on its first run, in the same tables our own
  log uses (`vesta.releases.register_release_plan`), so the jobs and the window read it
  unchanged. Git is not asked.
- Whether the code is the release's is the manifest's word: `RELEASE.sha256` at the repository
  root, written when the release is made, lists the sha256 of every file under `src/vesta` and
  `config/vesta`. Equal: the release. Missing or different: a modified Vesta, whose every
  publication carries `not_validated_on_dataset`, as a foreign dataset's.
"""

import hashlib
import sqlite3
import tomllib
from pathlib import Path

from vesta.code_version import _order
from vesta.jobs.past_cache import FROZEN_FOLDERS
from vesta.quantities import load_quantities
from vesta.releases import read_plan, register_release_plan

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = "RELEASE.sha256"


def find_latest_plan(root: Path = ROOT) -> Path:
    plans = sorted((root / "config/vesta/releases").glob("*.toml"), key=_order)
    if not plans:
        raise FileNotFoundError("no release plan in config/vesta/releases")
    return plans[-1]


def _list_frozen_files(root: Path) -> list[Path]:
    return [path for folder in FROZEN_FOLDERS for path in sorted((root / folder).rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"]


def _hash_normalised_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_manifest(root: Path = ROOT) -> Path:
    """The release's manifest, written when the release is made (line endings normalised, as
    `release_identity`)."""
    lines = [
        f"{_hash_normalised_file(path)}  {path.relative_to(root).as_posix()}"
        for path in _list_frozen_files(root)
    ]
    target = root / MANIFEST
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return target


def check_manifest(root: Path = ROOT) -> tuple[str, str]:
    """("release", why) when every file is the manifest's and no file is added; else
    ("modified", why)."""
    manifest = root / MANIFEST
    if not manifest.is_file():
        return "modified", f"no {MANIFEST}: not a released copy of Vesta"
    listed = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        if name:
            listed[name] = digest
    present = {path.relative_to(root).as_posix(): path for path in _list_frozen_files(root)}
    changed = sorted(name for name, path in present.items()
                     if listed.get(name) != _hash_normalised_file(path))  # fmt: skip
    missing = sorted(set(listed) - set(present))
    if changed or missing:
        names = ", ".join((changed + missing)[:5])
        return "modified", f"{len(changed) + len(missing)} files differ from {MANIFEST}: {names}"
    return "release", f"every file as {MANIFEST} lists it"


def register_release_gate(log: sqlite3.Connection, dataset: sqlite3.Connection, published_from: str,
                          root: Path = ROOT) -> tuple[str, dict[tuple[str, int], str]]:  # fmt: skip
    """The release registered in the local log (once), and its gate: (key, competition id) ->
    model version, for the competitions the dataset holds."""
    plan = find_latest_plan(root)
    version, verdict_id, what_changed, parts = read_plan(plan, dataset, skip_missing=True)
    model_version = f"vesta-{version}"
    known = log.execute("SELECT 1 FROM model_versions WHERE model_version = ?",
                        (model_version,)).fetchone()  # fmt: skip
    if known is None:
        register_release_plan(log, root, version, verdict_id, what_changed, published_from,
                              parts, load_quantities(), check_freeze=False)  # fmt: skip
    else:  # a later run, maybe with more files: the competitions not yet registered
        _register_missing_competitions(log, version, published_from, parts)
    pairs = {(key, competition): version_of for key, competition, version_of in log.execute(
        "SELECT quantity_key, competition_id, model_version FROM published_quantities"
        " WHERE published_until IS NULL AND model_version = ?", (model_version,))}  # fmt: skip
    return model_version, pairs


def _register_missing_competitions(log, version, published_from, parts) -> None:
    from vesta.forecast_log.writing import register_published_quantity

    model_version = f"vesta-{version}"
    present = set(log.execute("SELECT quantity_key, competition_id FROM published_quantities"
                              " WHERE model_version = ?", (model_version,)).fetchall())
    definitions = load_quantities()
    for part in parts:
        own = [item for item in definitions.values() if item.family_member == part.family_member
               and (part.published_keys is None
                    or item.quantity.quantity_key in part.published_keys)]  # fmt: skip
        for item in own:
            for competition_id in part.competition_ids:
                if (item.quantity.quantity_key, competition_id) not in present:
                    register_published_quantity(
                        log, item.quantity.quantity_key, competition_id, model_version,
                        published_from, False, False)  # fmt: skip


def read_release_version(root: Path = ROOT) -> str:
    return tomllib.loads(find_latest_plan(root).read_text(encoding="utf-8"))["version"]
