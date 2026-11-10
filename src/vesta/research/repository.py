"""What the discipline asks of git: the committed state of a file, the clean state of a path,
the difference between a tag and the working commit.

A verdict and a freeze are only as trustworthy as the proof that the code and the registry they
name are the ones committed, so every such question goes to git and nowhere else.
"""

import subprocess
from pathlib import Path


class RepositoryError(RuntimeError):
    """Git could not answer: not a repository, an unknown revision, git missing."""


def run_git(root: Path, *arguments: str) -> str:
    """The output of `git <arguments>` in `root`; on failure `RepositoryError` with git's words."""
    result = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, encoding="utf-8"
    )
    if result.returncode != 0:
        raise RepositoryError(f"git {' '.join(arguments)}: {result.stderr.strip()}")
    return result.stdout


def read_committed_text(root: Path, relative_path: str) -> str | None:
    """The file as committed at HEAD, or None when HEAD does not hold it."""
    try:
        return run_git(root, "show", f"HEAD:{relative_path}")
    except RepositoryError:
        return None


def list_uncommitted_paths(root: Path, *relative_paths: str) -> list[str]:
    """The files under `relative_paths` that differ from HEAD or are not tracked."""
    status = run_git(root, "status", "--porcelain", "--untracked-files=all", "--", *relative_paths)
    return [line[3:] for line in status.splitlines() if line]


def list_changed_since(root: Path, revision: str, *relative_paths: str) -> list[str]:
    """The files under `relative_paths` that differ between `revision` and HEAD."""
    output = run_git(root, "diff", "--name-only", revision, "HEAD", "--", *relative_paths)
    return output.splitlines()


def has_tag(root: Path, tag: str) -> bool:
    return bool(run_git(root, "tag", "--list", tag).strip())
