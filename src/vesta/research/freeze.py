"""The freeze: the published model is the code and configuration of a git tag.

A version of Vesta is published only from a tag `vesta-<version>` (for example `vesta-2.0.0`)
made after its verdict and the maintainers' approval; it changes only by a new verdict and a new
tag. The check below proves that what runs is what was frozen: the tag exists, and the model's
code and configuration at HEAD are those of the tag, with nothing uncommitted. The working tree's
line endings are fixed by `.gitattributes` (eol=lf), so a checkout on Windows does not break the
guard.
"""

from pathlib import Path

from vesta.research.repository import has_tag, list_changed_since, list_uncommitted_paths

FROZEN_PATHS = ("src/vesta", "config/vesta")


def make_tag(version: str) -> str:
    return f"vesta-{version}"


def find_freeze_problems(root: Path, version: str) -> list[str]:
    """Every reason the working tree is not the frozen version `version`; empty when it is."""
    tag = make_tag(version)
    if not has_tag(root, tag):
        return [f"the tag {tag} does not exist: version {version} was never frozen"]
    problems = [
        f"{path} differs from {tag}" for path in list_changed_since(root, tag, *FROZEN_PATHS)
    ]
    problems += [f"{path} is not committed" for path in list_uncommitted_paths(root, *FROZEN_PATHS)]
    return problems
