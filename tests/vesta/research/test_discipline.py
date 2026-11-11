"""The rules of the discipline: registry, leakage, seasons, verdicts, freeze."""

import random
import subprocess
import tomllib
from pathlib import Path

import pytest

from vesta.research.freeze import find_freeze_problems
from vesta.research.hypotheses import (
    REGISTRY_PATH,
    RegistryError,
    parse_registry,
    read_committed_hypothesis,
)
from vesta.research.leakage import (
    LeakageAlarm,
    LeakageError,
    check_known_by,
    enforce,
    find_alarms,
)
from vesta.research.seasons import (
    ClosedSeasonError,
    refuse_closed,
)
from vesta.research.verdicts import PairedScore, VerdictRules, judge, write_verdict

ROOT = Path(__file__).resolve().parents[3]
ENTRY = """
[[hypothesis]]
id = "H1"
family_member = "spine"
title = "the refitted champion"
change = "the old champion on the new world"
catalogue_ids = ["M1", "M2"]
distribution_kind = "ordered"
expected_gain = 0.0
registered_on = "2026-10-23"
"""
PROVENANCE = {"code_commit": "abc1234", "world_build": "f00d", "command": "python -m vesta ..."}


def research_config() -> dict:
    return tomllib.loads((ROOT / "config/vesta/research.toml").read_text(encoding="utf-8"))


def rules(**changes) -> VerdictRules:
    base = VerdictRules.from_config(research_config())
    return VerdictRules(**{**base.__dict__, "bootstrap_repetitions": 300, **changes})


def git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "vesta tests")
    git(tmp_path, "config", "user.name", "Vesta tests")
    (tmp_path / "config/vesta").mkdir(parents=True)
    (tmp_path / "src/vesta").mkdir(parents=True)
    (tmp_path / REGISTRY_PATH).write_text(ENTRY, encoding="utf-8")
    (tmp_path / "src/vesta/model.py").write_text("WEIGHT = 0.5\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "register H1")
    return tmp_path


def test_the_project_configuration_and_registry_parse():
    config = research_config()
    registry = (ROOT / REGISTRY_PATH).read_text(encoding="utf-8")
    entries = parse_registry(registry, config["verdict"]["grid_max"])
    assert [entry.id for entry in entries][:3] == ["H1", "H2", "H3"]
    assert config["alarm"]["ordered"]["max_accuracy"] == 0.56


def test_an_entry_takes_the_first_metric_of_its_kind():
    (entry,) = parse_registry(ENTRY, grid_max=6)
    assert entry.metric == "ranked_probability_score"
    assert entry.catalogue_ids == ("M1", "M2")


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ('metric = "brier"\n', None),
        ('metric = "accuracy"\n', "not a metric"),
        ("grid = [{a = 1}, {a = 2}, {a = 3}]\n", "limit is 2"),
        ('colour = "red"\n', "unknown fields"),
    ],
)
def test_registry_rules(extra, message):
    text = ENTRY + extra
    if message is None:
        assert parse_registry(text, grid_max=2)[0].metric == "brier"
    else:
        with pytest.raises(RegistryError, match=message):
            parse_registry(text, grid_max=2)


def test_registry_refuses_a_duplicate_id_and_a_missing_field():
    with pytest.raises(RegistryError, match="twice"):
        parse_registry(ENTRY + ENTRY, grid_max=6)
    with pytest.raises(RegistryError, match="lacks"):
        parse_registry(ENTRY.replace('title = "the refitted champion"\n', ""), grid_max=6)


def test_a_committed_entry_is_found(repository):
    assert read_committed_hypothesis(repository, "H1", grid_max=6).title == "the refitted champion"


def test_an_entry_written_after_the_commit_is_refused(repository):
    registry = repository / REGISTRY_PATH
    registry.write_text(ENTRY + ENTRY.replace('"H1"', '"H2"'), encoding="utf-8")
    with pytest.raises(RegistryError, match="uncommitted"):
        read_committed_hypothesis(repository, "H2", grid_max=6)
    git(repository, "commit", "-q", "-am", "register H2")
    assert read_committed_hypothesis(repository, "H2", grid_max=6).id == "H2"
    with pytest.raises(RegistryError, match="not in the committed"):
        read_committed_hypothesis(repository, "H3", grid_max=6)


def test_inputs_known_after_the_forecast_stop_the_run():
    check_known_by(
        [
            ("result of match 1", "2026-10-22T21:00:00Z"),
            ("result of match 2", "2026-10-23T08:00:00Z"),  # known at the forecast's instant
        ],
        "2026-10-23T08:00:00Z",
    )
    with pytest.raises(LeakageError, match="1 inputs"):
        check_known_by(
            [("result of match 1", "2026-10-22T21:00:00Z"), ("lineup", "2026-10-23T08:00:01Z")],
            "2026-10-23T08:00:00Z",
        )


def test_alarm_trips_on_gain_and_accuracy():
    alarm_rules = research_config()["alarm"]
    assert find_alarms("ordered", alarm_rules, gain=0.004, accuracy=0.53) == []
    findings = find_alarms("ordered", alarm_rules, gain=0.03, accuracy=0.58)
    assert [finding.condition for finding in findings] == [
        "gain over the published model",
        "accuracy",
    ]


def test_a_kind_without_an_alarm_rule_cannot_be_judged():
    with pytest.raises(LeakageAlarm, match="no alarm rule"):
        find_alarms("time_to_event", research_config()["alarm"], gain=0.0)


def test_an_alarm_stops_until_a_record_explains_it(tmp_path):
    findings = find_alarms("ordered", research_config()["alarm"], gain=0.05)
    enforce([], None, tmp_path)
    with pytest.raises(LeakageAlarm, match="stops"):
        enforce(findings, None, tmp_path)
    with pytest.raises(LeakageAlarm, match="does not exist"):
        enforce(findings, "docs/research/explained.md", tmp_path)
    (tmp_path / "docs/research").mkdir(parents=True)
    (tmp_path / "docs/research/explained.md").write_text("why", encoding="utf-8")
    enforce(findings, "docs/research/explained.md", tmp_path)




def test_closed_seasons_are_refused():
    refuse_closed([2022, 2023], [2024, 2025])
    with pytest.raises(ClosedSeasonError, match="2024"):
        refuse_closed([2023, 2024], [2024, 2025])


def paired_scores(improvement: float, seed: int = 1) -> list[PairedScore]:
    generator = random.Random(seed)
    pairs = []
    for index in range(1500):
        season = 2019 + index % 5
        published = generator.uniform(0.1, 0.4)
        candidate = published - improvement + generator.gauss(0.0, 0.01)
        pairs.append(PairedScore(f"m{index}", season, f"SA|{index // 12}", published, candidate))
    return pairs


def calibrated_lines() -> list[tuple[float, bool]]:
    return [(0.3, index % 10 < 3) for index in range(100)]


def test_a_clear_improvement_is_kept(tmp_path):
    (hypothesis,) = parse_registry(ENTRY, grid_max=6)
    lines = calibrated_lines()
    pairs = paired_scores(0.01)
    verdict = judge(hypothesis, pairs, lines, lines, rules(), [], PROVENANCE, tmp_path)
    assert verdict.kept
    assert verdict.interval_high < 0 and verdict.seasons_improved == 5


def test_no_improvement_is_discarded(tmp_path):
    (hypothesis,) = parse_registry(ENTRY, grid_max=6)
    lines = calibrated_lines()
    verdict = judge(hypothesis, paired_scores(0.0), lines, lines, rules(), [], PROVENANCE, tmp_path)
    assert not verdict.kept
    assert not verdict.conditions["interval_below_zero"]


def test_a_worse_calibration_discards_a_better_score(tmp_path):
    (hypothesis,) = parse_registry(ENTRY, grid_max=6)
    badly = [(0.9, index % 10 < 3) for index in range(100)]
    verdict = judge(
        hypothesis, paired_scores(0.01), calibrated_lines(), badly, rules(), [], PROVENANCE,
        tmp_path,
    )  # fmt: skip
    assert not verdict.kept and not verdict.conditions["calibration_not_worse"]


def test_a_verdict_on_a_closed_season_is_refused(tmp_path):
    (hypothesis,) = parse_registry(ENTRY, grid_max=6)
    pairs = [PairedScore("m1", 2024, "SA|1", 0.3, 0.2)]
    lines = calibrated_lines()
    with pytest.raises(ClosedSeasonError):
        judge(hypothesis, pairs, lines, lines, rules(), [], PROVENANCE, tmp_path)


def test_a_verdict_is_reproduced_and_written_once(tmp_path):
    (hypothesis,) = parse_registry(ENTRY, grid_max=6)
    lines = calibrated_lines()
    first = judge(hypothesis, paired_scores(0.01), lines, lines, rules(), [], PROVENANCE, tmp_path)
    again = judge(hypothesis, paired_scores(0.01), lines, lines, rules(), [], PROVENANCE, tmp_path)
    assert first == again
    path = write_verdict(first, tmp_path)
    assert path.read_text(encoding="utf-8").startswith("# Verdict of H1: kept")
    with pytest.raises(FileExistsError):
        write_verdict(first, tmp_path)


def test_freeze_holds_only_on_the_tagged_code(repository):
    assert find_freeze_problems(repository, "2.0.0") == [
        "the tag vesta-2.0.0 does not exist: version 2.0.0 was never frozen"
    ]
    git(repository, "tag", "vesta-2.0.0")
    assert find_freeze_problems(repository, "2.0.0") == []
    (repository / "src/vesta/model.py").write_text("WEIGHT = 0.6\n", encoding="utf-8")
    assert find_freeze_problems(repository, "2.0.0") == ["src/vesta/model.py is not committed"]
    git(repository, "commit", "-q", "-am", "change the weight")
    assert find_freeze_problems(repository, "2.0.0") == [
        "src/vesta/model.py differs from vesta-2.0.0"
    ]
