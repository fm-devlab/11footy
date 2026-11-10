"""The verdict runner end to end on the small world: every verdict runs or says why it cannot,
one report holds them all, and the closed seasons are refused."""

import tomllib
from pathlib import Path

import pytest

from vesta.verdicts import report, run_verdicts

ROOT = Path(__file__).resolve().parents[3]
SPINE = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
RESEARCH = tomllib.loads((ROOT / "config/vesta/research.toml").read_text(encoding="utf-8"))
NAMES = ["B0 (the previous baseline)", "The champion", "The UEFA cups", "The national teams",
         "Goals in time (step 3)", "Ties (step 3b)"]  # fmt: skip


def test_every_verdict_runs_or_says_why_in_one_report(world, tmp_path, monkeypatch):
    # the report names the commit; the public tree is tested outside a git checkout too
    monkeypatch.setattr("vesta.verdicts.run_git", lambda root, *arguments: "0" * 40)
    path = tmp_path / "football-world.db"
    verdicts = run_verdicts(path, (2016, 2017), SPINE, RESEARCH, say=lambda _: None)
    assert [verdict.name for verdict in verdicts] == NAMES
    b0 = verdicts[0]
    assert b0.passed is not None  # the small world's two leagues are enough for B0
    assert any(row.startswith("| all |") for row in b0.rows)
    assert "deduplicated diagnostic on the uncorrected world: 0.99213" in b0.note
    for verdict in verdicts:
        assert verdict.passed is not None or verdict.note.startswith("could not run")
    text = report(verdicts, path, (2016, 2017))
    assert text.startswith("# The verdicts")
    for name in NAMES:
        assert f"## {name}" in text
    assert "| B0 (the previous baseline) | " in text


def test_the_closed_seasons_are_never_read(world, tmp_path):
    closed = tuple(RESEARCH["seasons"]["closed"])
    with pytest.raises(ValueError, match="closed seasons"):
        run_verdicts(tmp_path / "football-world.db", closed, SPINE, RESEARCH, say=lambda _: None)
