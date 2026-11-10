"""The code's version without git, so Vesta runs from a ZIP download too.

`<release>+<identity>`: the latest release plan's version in `config/vesta/releases` and the first
12 hex of `vesta.jobs.past_cache.release_identity` (the sha256 of every file under `src/vesta` and
`config/vesta`), so two copies with the same files have the same version, and a changed file
changes it. Git is never asked.
"""

import tomllib
from pathlib import Path

from vesta.jobs.past_cache import hash_release


def _order(path: Path) -> tuple[int, ...]:
    return tuple(int(part) for part in path.stem.split("."))


def read_code_version(root: Path) -> str:
    releases = sorted((root / "config/vesta/releases").glob("*.toml"), key=_order)
    version = (tomllib.loads(releases[-1].read_text(encoding="utf-8"))["version"] if releases
               else "unreleased")  # fmt: skip
    return f"{version}+{hash_release(root)[:12]}"
