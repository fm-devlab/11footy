"""The spine's quantities, its distributions of a match, and the registration of a frozen
version in the log."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from vesta.forecast_log.store import ForecastLogError, open_log
from vesta.forecast_log.writing import Publication, publish
from vesta.forecast_reading import read_line
from vesta.quantities import load_quantities
from vesta.releases import ReleasePart, register_release_plan
from vesta.spine.publication import build_spine_distributions
from vesta.spine.score_matrix import Calibration, score_matrices

MATCH, HOME, AWAY = 77120, 13, 14


def register_part(log, root: Path, version: str, family_member: str, verdict_id: str,
                  what_changed: str, published_from: str, competition_ids: list[int],
                  definitions, published_keys: set[str] | None = None) -> list[str]:  # fmt: skip
    """A release plan of one part; returns the keys now published."""
    part = ReleasePart(family_member, tuple(competition_ids),
                       None if published_keys is None else frozenset(published_keys))  # fmt: skip
    pairs = register_release_plan(log, root, version, verdict_id, what_changed, published_from,
                                  [part], definitions)  # fmt: skip
    return list(dict.fromkeys(key for key, _ in pairs))


MATRIX = score_matrices(np.array([1.6]), np.array([1.1]),
                        Calibration(-0.05, 0.08, 1.0, 1.0, 0.1, 1.08))[0]  # fmt: skip


def by_key(distributions):
    return {(item.quantity_key, item.subject_id): item.values for item in distributions}


def test_the_spines_quantities_are_defined_once_with_readable_lines():
    definitions = load_quantities()
    spine = {key for key, item in definitions.items() if item.family_member == "spine"}
    assert spine == {
        "M1.result", "M2.score", "M3.goals", "M4.both_score", "M5.goals.team",
        "T2.points.so_far", "T2.goals_for.so_far", "T2.goals_against.so_far",
        "T2.clean_sheets.so_far",
    }  # fmt: skip
    # the record's flags: the 1X2 moved live (from vesta-2.0.2), both score too (from vesta-2.6.0);
    # the rest of the spine did not
    assert definitions["M1.result"].live_headline and definitions["M1.result"].moves_live
    both = definitions["M4.both_score"]
    assert both.moves_live and not both.live_headline
    assert not any(definitions[key].moves_live for key in
                   ("M2.score", "M3.goals", "M5.goals.team"))  # fmt: skip
    assert "margin_at_least=home,2" in definitions["M2.score"].quantity.lines


def test_every_declared_line_reads_from_the_spines_distributions():
    definitions = load_quantities()
    for distribution in build_spine_distributions(MATRIX, MATCH, HOME, AWAY, "vesta-2.0.0"):
        quantity = definitions[distribution.quantity_key].quantity
        for line_key in quantity.lines:
            probability = read_line(quantity.distribution_kind, distribution.values, line_key)
            assert 0.0 <= probability <= 1.0, (distribution.quantity_key, line_key)


def test_the_spines_numbers_agree_because_they_come_from_one_matrix():
    values = by_key(build_spine_distributions(MATRIX, MATCH, HOME, AWAY, "vesta-2.0.0"))
    score = values[("M2.score", MATCH)]
    for outcome, line in (("home", "home_win"), ("draw", "draw"), ("away", "away_win")):
        assert read_line("ordered", values[("M1.result", MATCH)], f"outcome={outcome}") == (
            pytest.approx(read_line("joint_count", score, line), abs=1e-12)
        )
    for threshold in ("0.5", "2.5", "4.5"):
        assert read_line("count", values[("M3.goals", MATCH)], f"over={threshold}") == (
            pytest.approx(read_line("joint_count", score, f"total_over={threshold}"), abs=1e-12)
        )
        assert read_line("count", values[("M5.goals.team", HOME)], f"over={threshold}") == (
            pytest.approx(read_line("joint_count", score, f"home_over={threshold}"), abs=1e-12)
        )
    assert values[("M4.both_score", MATCH)]["yes"] == pytest.approx(
        read_line("joint_count", score, "both_score"), abs=1e-12
    )
    assert read_line("count", values[("M5.goals.team", AWAY)], "exactly=0") == pytest.approx(
        1 - read_line("joint_count", score, "away_scores"), abs=1e-12
    )


def git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)


@pytest.fixture
def frozen_repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    (root / "src/vesta").mkdir(parents=True)
    (root / "config/vesta").mkdir(parents=True)
    (root / "src/vesta/model.py").write_text("WEIGHT = 0.5\n", encoding="utf-8")
    git(root, "init", "-q")
    git(root, "config", "user.email", "vesta tests")
    git(root, "config", "user.name", "Vesta tests")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "the spine")
    return root


def test_a_version_is_registered_only_from_its_tag(tmp_path, frozen_repository):
    log = open_log(tmp_path / "forecast-log.db")
    arguments = ("2.0.0", "spine", "S1", "the spine realigned", "2026-11-01T00:00:00Z", [1, 2],
                 load_quantities())  # fmt: skip
    with pytest.raises(ForecastLogError, match="not the frozen code"):
        register_part(log, frozen_repository, *arguments)
    git(frozen_repository, "tag", "vesta-2.0.0")
    keys = register_part(log, frozen_repository, *arguments)
    assert len(keys) == 9
    assert log.execute("SELECT freeze_tag FROM model_versions").fetchone() == ("vesta-2.0.0",)
    assert log.execute("SELECT COUNT(*) FROM published_quantities").fetchone() == (18,)
    publication = Publication(
        stage="pre_match",
        trigger="first",
        published_at="2026-11-01T08:00:00Z",
        code_commit="abc1234",
        match_id=MATCH,
        scheduled_kickoff_utc="2026-11-02T15:00:00Z",
    )
    publish(log, publication, build_spine_distributions(MATRIX, MATCH, HOME, AWAY, "vesta-2.0.0"))
    assert log.execute("SELECT COUNT(*) FROM distributions").fetchone() == (6,)
    log.close()


def test_a_key_already_in_the_log_with_another_meaning_is_refused(tmp_path, frozen_repository):
    git(frozen_repository, "tag", "vesta-2.0.0")
    log = open_log(tmp_path / "forecast-log.db")
    definitions = load_quantities()
    register_part(log, frozen_repository, "2.0.0", "spine", "S1", "the spine",
                     "2026-11-01T00:00:00Z", [1], definitions)  # fmt: skip
    changed = definitions["M3.goals"]
    definitions["M3.goals"] = type(changed)(
        type(changed.quantity)(**{**changed.quantity.__dict__, "lines": ("over=2.5",)}),
        changed.family_member, changed.moves_live, changed.live_headline,
    )  # fmt: skip
    git(frozen_repository, "tag", "vesta-2.0.1")
    with pytest.raises(ForecastLogError, match="another definition"):
        register_part(log, frozen_repository, "2.0.1", "spine", "S2", "the spine again",
                         "2026-12-01T00:00:00Z", [1], definitions)  # fmt: skip
    log.close()


def test_a_release_publishes_only_the_keys_whose_verdicts_passed(tmp_path, frozen_repository):
    git(frozen_repository, "tag", "vesta-2.0.0")
    log = open_log(tmp_path / "forecast-log.db")
    definitions = load_quantities()
    passed = {"M9.result.first_half", "M7.first_goal"}
    keys = register_part(log, frozen_repository, "2.0.0", "goals in time", "GT1",
                            "goals in time", "2026-11-01T00:00:00Z", [1], definitions,
                            passed)  # fmt: skip
    assert set(keys) == passed
    published = {row[0] for row in log.execute("SELECT quantity_key FROM published_quantities")}
    assert published == passed  # the others keep their definitions, unpublished
    assert log.execute("SELECT COUNT(*) FROM quantities WHERE quantity_key ="
                       " 'M8.goal_in_band.band=16-30'").fetchone() == (1,)  # fmt: skip
    git(frozen_repository, "tag", "vesta-2.0.1")
    with pytest.raises(ForecastLogError, match="not quantities of"):
        register_part(log, frozen_repository, "2.0.1", "goals in time", "GT2", "again",
                         "2026-12-01T00:00:00Z", [1], definitions, {"M1.result"})  # fmt: skip
    log.close()


def test_a_release_plan_registers_its_version_once_with_every_part(tmp_path, frozen_repository):
    git(frozen_repository, "tag", "vesta-2.0.0")
    log = open_log(tmp_path / "forecast-log.db")
    parts = [ReleasePart("spine", (1, 2), frozenset({"M1.result", "M2.score"})),
             ReleasePart("spine", (7,), frozenset({"M1.result"})),  # a cup, by its own verdict
             ReleasePart("goals in time", (1,), frozenset({"M7.first_goal"}))]
    pairs = register_release_plan(log, frozen_repository, "2.0.0", "V1", "the first release",
                                  "2026-11-01T00:00:00Z", parts, load_quantities())  # fmt: skip
    assert sorted(pairs) == [("M1.result", 1), ("M1.result", 2), ("M1.result", 7),
                             ("M2.score", 1), ("M2.score", 2), ("M7.first_goal", 1)]
    assert log.execute("SELECT model_version, family_member FROM model_versions").fetchall() == [
        ("vesta-2.0.0", "spine, goals in time")]
    moving = log.execute("SELECT MAX(moves_live) FROM published_quantities").fetchone()[0]
    assert moving == 0  # every key is pre-match
    log.close()
    log = open_log(tmp_path / "other-log.db")
    twice = [ReleasePart("spine", (1,), frozenset({"M1.result"}))] * 2
    with pytest.raises(ForecastLogError, match="published twice"):
        register_release_plan(log, frozen_repository, "2.0.0", "V1", "again",
                              "2026-11-01T00:00:00Z", twice, load_quantities())  # fmt: skip
    log.close()


def test_a_new_release_supersedes_the_previous_one_with_all_its_keys(tmp_path, frozen_repository):
    from vesta.jobs.pre_match import read_published_versions

    git(frozen_repository, "tag", "vesta-2.0.0")
    git(frozen_repository, "tag", "vesta-2.0.1")
    log = open_log(tmp_path / "forecast-log.db")
    first = [ReleasePart("spine", (1, 2), frozenset({"M1.result", "M2.score"}))]
    register_release_plan(log, frozen_repository, "2.0.0", "V1", "the spine",
                          "2026-10-28T14:23:00Z", first, load_quantities())  # fmt: skip
    second = [ReleasePart("spine", (1,), frozenset({"M1.result"})),
              ReleasePart("goals in time", (1,), frozenset({"M7.first_goal"}))]
    register_release_plan(log, frozen_repository, "2.0.1", "V2", "the passing keys",
                          "2026-10-29T06:00:00Z", second, load_quantities())  # fmt: skip
    assert read_published_versions(log) == {
        ("M1.result", 1): "vesta-2.0.1",
        ("M7.first_goal", 1): "vesta-2.0.1",
    }  # M2.score and league 2 stop
    assert log.execute("SELECT model_version, superseded_from FROM model_versions"
                       " ORDER BY 1").fetchall() == [("vesta-2.0.0", "2026-10-29T06:00:00Z"),
                                                     ("vesta-2.0.1", None)]  # fmt: skip
    assert log.execute("SELECT COUNT(*) FROM published_quantities WHERE model_version ="
                       " 'vesta-2.0.0' AND published_until = '2026-10-29T06:00:00Z'"
                       ).fetchone() == (4,)  # fmt: skip
    log.close()
