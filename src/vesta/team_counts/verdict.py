"""The verdict of team counts (corners, cards), declared before the run in
its plan (4ae876c), its alarm in `[alarm.count]` and
`[alarm.joint_count]` (7c32c2d).

    python -m vesta.team_counts.verdict <football-world.db> [--stage development|blind]
                                        [--out <report.md>]

Every league match of the stage's seasons with the provider's counts is forecast walk-forward
(`vesta.team_counts.model`) and each key is scored on its own against the reference: the
competition's frequencies of the seasons before, from the first training season, each outcome
counted half a time more, folded at the same sizes as the model's. The deciding metric is the
log score of what happened. A key passes when its mean difference to the reference has a block
bootstrap interval entirely below zero, it improved in enough seasons, and its calibration error
is no worse than the reference's by more than the tolerance; the alarm stops a key whose gain is
larger than its kind's limit.
"""

import argparse
import hashlib
import math
import sys
import time
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vesta.determinism import run_on_one_thread
from vesta.forecast_reading import read_line
from vesta.quantities import load_quantities
from vesta.research.bootstrap import assign_block, bootstrap_mean
from vesta.research.calibration import measure_calibration_error
from vesta.research.repository import run_git
from vesta.research.scores import LOG_LOSS, RANKED_PROBABILITY, score
from vesta.spine.goals_in_time_verdict import _has_line_happened
from vesta.spine.timing import TimeRules
from vesta.spine.world import open_world
from vesta.team_counts.model import (
    KEYS,
    Calibration,
    FitParameters,
    count_distributions,
    fit_calibration,
    read_count_matches,
    walk_forward,
)

ROOT = Path(__file__).resolve().parents[3]
SMOOTHING = 0.5
CALIBRATION_TOLERANCE = 0.005


@dataclass(frozen=True)
class KeyVerdict:
    key: str
    kind: str
    forecasts: int
    model: float
    reference: float
    difference: float
    low: float
    high: float
    seasons_improved: int
    seasons: int
    ranked_model: float | None
    ranked_reference: float | None
    calibration_model: float
    calibration_reference: float
    passed: bool
    alarm: bool


def fold_outcomes(count: str, home: int, away: int, size: int) -> dict[tuple[str, str], object]:
    """What happened to each key, by (key, side), folded as the distributions are."""
    total_key, team_key, sides_key = KEYS[count]
    top = size - 1
    return {(total_key, "match"): min(home + away, 2 * top),
            (team_key, "home"): min(home, top), (team_key, "away"): min(away, top),
            (sides_key, "match"): (min(home, top), min(away, top))}  # fmt: skip


def smooth_frequencies(kind: str, seen: list, size: int, total: bool = False) -> object:
    """The smoothed frequencies of the outcomes, as long as the model's distribution: `size`
    cells for a side, 2 size - 1 for a match's total."""
    if kind == "joint_count":
        counts = np.full((size, size), SMOOTHING)
        for home, away in seen:
            counts[home, away] += 1
        return {"matrix": (counts / counts.sum()).tolist(), "tail": 0.0}
    counts = np.bincount(seen, minlength=2 * size - 1 if total else size) + SMOOTHING
    return {"probabilities": (counts / counts.sum()).tolist(), "tail_from": len(counts),
            "tail": 0.0}  # fmt: skip


@run_on_one_thread
def run(world_path: Path, stage: str, candidate: int = 1,
        say=print) -> dict[str, list[KeyVerdict]]:  # fmt: skip
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    research = tomllib.loads((ROOT / "config/vesta/research.toml").read_text(encoding="utf-8"))
    config = tomllib.loads((ROOT / "config/vesta/team_counts.toml").read_text(encoding="utf-8"))
    seasons = set(research["seasons"]["development" if stage == "development" else "closed"])
    needed = 3 if stage == "development" else int(research["blind"]["min_seasons_improved"])
    repetitions = int(research["verdict"]["bootstrap_repetitions"])
    seed = int(research["verdict"]["seed"])
    first_training = int(spine["walk_forward"]["first_training_season"])
    leagues = list(spine["walk_forward"]["competitions"])
    definitions = load_quantities()
    results = {}
    for count in KEYS:
        parameters = FitParameters.from_config(config, count)
        size = parameters.size
        with open_world(world_path) as world:
            matches = read_count_matches(world, leagues, first_training - 2, max(seasons), count,
                                         TimeRules.from_config(spine))  # fmt: skip
        say(f"{count}: {len(matches)} matches read,"
            f" {sum(m.home_count is not None for m in matches)} with counts")
        history: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for match in matches:
            if match.home_count is not None:
                history[(match.competition, match.season)].append(
                    fold_outcomes(count, match.home_count, match.away_count, size))
        rows: dict[str, list] = defaultdict(list)
        lines: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
        references: dict[tuple[str, int], dict] = {}
        wanted = seasons | ({season - back for season in seasons for back in (1, 2)}
                            if candidate == 2 else set())  # fmt: skip
        forecasts = walk_forward(matches, parameters, wanted)
        calibrations = {season: Calibration() for season in seasons}
        if candidate == 2:  # fitted on the two seasons before each one judged, never on it
            for season in sorted(seasons):
                calibrations[season] = fit_calibration(
                    [item for item in forecasts if item[0].season in (season - 2, season - 1)],
                    size)
                say(f"{count} {season}: {calibrations[season]}")
        for match, means, size_nb in forecasts:
            if match.season not in seasons:
                continue
            calibration = calibrations[match.season]
            means = calibration.predict_means(means)
            key_of = (match.competition, match.season)
            if key_of not in references:
                training = [happened for (league, season), seen in history.items()
                            if league == match.competition and first_training <= season
                            < match.season for happened in seen]  # fmt: skip
                references[key_of] = (
                    {
                        (key, side): smooth_frequencies(
                            definitions[key].quantity.distribution_kind,
                            [happened[(key, side)] for happened in training],
                            size,
                            key == KEYS[count][0],
                        )
                        for (key, side) in training[0]
                    }
                    if training
                    else {}
                )
            reference = references[key_of]
            if not reference:
                continue
            happened = fold_outcomes(count, match.home_count, match.away_count, size)
            block = assign_block(match.competition, match.forecast_time)
            for item in count_distributions(count, match.match_id, match.home_team_id,
                                            match.away_team_id, means, size_nb, size, "check",
                                            calibration.tilt):
                side = ("home" if item.subject_id == match.home_team_id and
                        item.quantity_key == KEYS[count][1] else
                        "away" if item.quantity_key == KEYS[count][1] else "match")  # fmt: skip
                kind = definitions[item.quantity_key].quantity.distribution_kind
                outcome, against = happened[(item.quantity_key, side)], reference[
                    (item.quantity_key, side)]
                own, naive = score(kind, item.values, outcome), score(kind, against, outcome)
                rows[item.quantity_key].append((own[LOG_LOSS], naive[LOG_LOSS],
                                                own.get(RANKED_PROBABILITY),
                                                naive.get(RANKED_PROBABILITY), match.season,
                                                block))  # fmt: skip
                model_lines, reference_lines = lines[item.quantity_key]
                for line in definitions[item.quantity_key].quantity.lines:
                    hit = _has_line_happened(kind, item.values, line, outcome)
                    model_lines.append((read_line(kind, item.values, line), hit))
                    reference_lines.append((read_line(kind, against, line), hit))
        alarm = research["alarm"]
        results[count] = [
            _judge_key(key, definitions[key].quantity.distribution_kind, rows[key], lines[key],
                       repetitions, seed, needed,
                       float(alarm[definitions[key].quantity.distribution_kind]["max_gain"]))
            for key in KEYS[count] if rows[key]]  # fmt: skip
        say(f"{count}: judged")
    return results


def _judge_key(key, kind, rows, key_lines, repetitions, seed, needed, max_gain) -> KeyVerdict:
    differences = [own - naive for own, naive, *_ in rows]
    mean, low, high = bootstrap_mean(differences, [row[5] for row in rows], repetitions, seed)
    by_season: dict[int, float] = defaultdict(float)
    for difference, row in zip(differences, rows, strict=True):
        by_season[row[4]] += difference
    improved = sum(1 for total in by_season.values() if total < 0)
    ranked = [row for row in rows if row[2] is not None]
    calibration_model = measure_calibration_error(key_lines[0])
    calibration_reference = measure_calibration_error(key_lines[1])
    alarm = -mean > max_gain
    passed = (high < 0 and improved >= needed and not alarm
              and calibration_model <= calibration_reference + CALIBRATION_TOLERANCE)  # fmt: skip
    return KeyVerdict(
        key, kind, len(rows), math.fsum(row[0] for row in rows) / len(rows),
        math.fsum(row[1] for row in rows) / len(rows), mean, low, high, improved, len(by_season),
        math.fsum(row[2] for row in ranked) / len(ranked) if ranked else None,
        math.fsum(row[3] for row in ranked) / len(ranked) if ranked else None,
        calibration_model, calibration_reference, passed, alarm)  # fmt: skip


def report(results: dict, stage: str, world_path: Path, world_sha: str, commit: str,
           seconds: float, candidate: int = 1) -> str:  # fmt: skip
    lines = [f"# Team counts: the {stage} verdicts (candidate {candidate})", "",
             f"World: `{world_path}` (sha256 {world_sha}). Code: `{commit}`. Written by"
             f" `python -m vesta.team_counts.verdict`; the rule is"
             f" the plan declared before the run (4ae876c), the alarms"
             f" `[alarm.count]` and `[alarm.joint_count]` (7c32c2d). {seconds:.0f} seconds.", "",
             "Log score of what happened, model against the competition's frequencies of the"
             " seasons before (lower is better); the interval is the block bootstrap of the"
             " difference.", "",
             "| key | n | model | reference | difference [95%] | seasons improved |"
             " ranked probability model / reference | calibration model / reference | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for count in results:
        for item in results[count]:
            ranked = ("-" if item.ranked_model is None else
                      f"{item.ranked_model:.4f} / {item.ranked_reference:.4f}")  # fmt: skip
            verdict = "ALARM" if item.alarm else "pass" if item.passed else "fail"
            lines.append(f"| {item.key} | {item.forecasts} | {item.model:.4f} |"
                         f" {item.reference:.4f} | {item.difference:+.4f} [{item.low:+.4f},"
                         f" {item.high:+.4f}] | {item.seasons_improved}/{item.seasons} |"
                         f" {ranked} | {item.calibration_model:.4f} /"
                         f" {item.calibration_reference:.4f} | {verdict} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("world", type=Path)
    parser.add_argument("--stage", choices=("development", "blind"), default="development")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--candidate", type=int, choices=(1, 2), default=1)
    options = parser.parse_args(arguments)
    started = time.monotonic()
    results = run(options.world, options.stage, options.candidate)
    digest = hashlib.sha256()
    with options.world.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 24), b""):
            digest.update(chunk)
    text = report(results, options.stage, options.world, digest.hexdigest(),
                  run_git(ROOT, "rev-parse", "HEAD").strip(), time.monotonic() - started,
                  options.candidate)
    out = options.out or ROOT / f"docs/research/2026-10-29-team-counts-{options.stage}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
