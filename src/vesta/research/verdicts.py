"""The verdict of a hypothesis: kept or discarded, with its numbers.

The rule is generalised from the result's log loss to the
deciding metric of any kind of distribution (`vesta.research.scores`). The candidate and the
published model forecast the same forecasts of the development seasons; per forecast the
difference of the deciding metric (candidate minus published, lower is better) is taken, and the
candidate is kept only when all three conditions hold:
- the whole interval of the mean difference, by block bootstrap, lies below zero;
- the candidate improves the mean in at least `min_seasons_improved` seasons;
- its expected calibration error is not worse than the published model's beyond the tolerance.
The leakage alarm runs first: a verdict is never written over an alarm not understood.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from vesta.research.bootstrap import bootstrap_mean
from vesta.research.calibration import measure_calibration_error
from vesta.research.hypotheses import Hypothesis
from vesta.research.leakage import AlarmFinding, enforce
from vesta.research.seasons import refuse_closed

VERDICTS_DIRECTORY = "docs/research/verdicts"


@dataclass(frozen=True)
class PairedScore:
    """One forecast scored under the published model and under the candidate."""

    forecast_key: str
    season: int
    block: str
    published: float
    candidate: float


@dataclass(frozen=True)
class VerdictRules:
    bootstrap_repetitions: int
    seed: int
    coverage: float
    min_seasons_improved: int
    calibration_tolerance: float
    closed_seasons: tuple[int, ...]

    @classmethod
    def from_config(cls, research: dict) -> "VerdictRules":
        """The rules of `config/vesta/research.toml` ([verdict] and [seasons])."""
        verdict = research["verdict"]
        return cls(
            bootstrap_repetitions=int(verdict["bootstrap_repetitions"]),
            seed=int(verdict["seed"]),
            coverage=float(verdict["coverage"]),
            min_seasons_improved=int(verdict["min_seasons_improved"]),
            calibration_tolerance=float(verdict["calibration_tolerance"]),
            closed_seasons=tuple(research["seasons"]["closed"]),
        )


@dataclass(frozen=True)
class Verdict:
    hypothesis_id: str
    metric: str
    forecasts: int
    published_mean: float
    candidate_mean: float
    difference_mean: float
    interval_low: float
    interval_high: float
    difference_by_season: dict[str, float]
    seasons_improved: int
    calibration_error_published: float
    calibration_error_candidate: float
    conditions: dict[str, bool]
    kept: bool
    code_commit: str
    world_build: str
    command: str


def judge(
    hypothesis: Hypothesis,
    pairs: list[PairedScore],
    published_lines: list[tuple[float, bool]],
    candidate_lines: list[tuple[float, bool]],
    rules: VerdictRules,
    alarm_findings: list[AlarmFinding],
    provenance: dict[str, str],
    root: Path,
) -> Verdict:
    """The verdict of `hypothesis` on its paired scores and calibration lines.

    `provenance` names `code_commit`, `world_build` and the `command` that reproduces it.
    """
    enforce(alarm_findings, hypothesis.alarm_cleared_by, root)
    if not pairs:
        raise ValueError(f"hypothesis {hypothesis.id}: no forecast to judge")
    keys = [pair.forecast_key for pair in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"hypothesis {hypothesis.id}: a forecast is scored twice")
    refuse_closed({pair.season for pair in pairs}, rules.closed_seasons)
    differences = [pair.candidate - pair.published for pair in pairs]
    mean, low, high = bootstrap_mean(
        differences,
        [pair.block for pair in pairs],
        rules.bootstrap_repetitions,
        rules.seed,
        rules.coverage,
    )
    by_season = _average_by_season(pairs)
    improved = sum(1 for value in by_season.values() if value < 0.0)
    published_error = measure_calibration_error(published_lines)
    candidate_error = measure_calibration_error(candidate_lines)
    conditions = {
        "interval_below_zero": high < 0.0,
        "enough_seasons_improved": improved >= rules.min_seasons_improved,
        "calibration_not_worse": candidate_error <= published_error + rules.calibration_tolerance,
    }
    return Verdict(
        hypothesis_id=hypothesis.id,
        metric=hypothesis.metric,
        forecasts=len(pairs),
        published_mean=math.fsum(pair.published for pair in pairs) / len(pairs),
        candidate_mean=math.fsum(pair.candidate for pair in pairs) / len(pairs),
        difference_mean=mean,
        interval_low=low,
        interval_high=high,
        difference_by_season=by_season,
        seasons_improved=improved,
        calibration_error_published=published_error,
        calibration_error_candidate=candidate_error,
        conditions=conditions,
        kept=all(conditions.values()),
        code_commit=provenance["code_commit"],
        world_build=provenance["world_build"],
        command=provenance["command"],
    )


def write_verdict(verdict: Verdict, root: Path) -> Path:
    """Write the verdict's record, `docs/research/verdicts/<id>.md`: the numbers for a person and
    the same numbers as JSON for the code. A verdict is written once; a second refuses."""
    path = root / VERDICTS_DIRECTORY / f"{verdict.hypothesis_id}.md"
    if path.exists():
        raise FileExistsError(f"the verdict of {verdict.hypothesis_id} exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_verdict(verdict), encoding="utf-8", newline="\n")
    return path


def render_verdict(verdict: Verdict) -> str:
    """The Markdown record of a verdict."""
    word = "kept" if verdict.kept else "discarded"
    lines = [
        f"# Verdict of {verdict.hypothesis_id}: {word}",
        "",
        f"Deciding metric: `{verdict.metric}` on {verdict.forecasts} forecasts (lower is better).",
        "",
        "| Number | Value |",
        "|---|---|",
        f"| published model, mean | {verdict.published_mean:.5f} |",
        f"| candidate, mean | {verdict.candidate_mean:.5f} |",
        f"| difference, mean | {verdict.difference_mean:+.5f} |",
        f"| interval of the difference | {verdict.interval_low:+.5f} to "
        f"{verdict.interval_high:+.5f} |",
        f"| seasons improved | {verdict.seasons_improved} of "
        f"{len(verdict.difference_by_season)} |",
        f"| calibration error, published / candidate | {verdict.calibration_error_published:.4f}"
        f" / {verdict.calibration_error_candidate:.4f} |",
        "",
        "Conditions: " + _describe_conditions(verdict.conditions) + ".",
        "",
        f"Reproduced by `{verdict.command}` at commit {verdict.code_commit} on the world build "
        f"{verdict.world_build}.",
        "",
        "```json",
        json.dumps(asdict(verdict), indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def _describe_conditions(conditions: dict[str, bool]) -> str:
    return ", ".join(f"{name} {'yes' if held else 'no'}" for name, held in conditions.items())


def _average_by_season(pairs: list[PairedScore]) -> dict[str, float]:
    sums: dict[int, list[float]] = {}
    for pair in pairs:
        sums.setdefault(pair.season, []).append(pair.candidate - pair.published)
    return {str(season): math.fsum(values) / len(values) for season, values in sorted(sums.items())}
