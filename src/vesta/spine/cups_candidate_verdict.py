"""The cups' spine candidate "cups v3": its verdict against the released cups model.

    python -m vesta.spine.cups_candidate_verdict <football-world.db> --stage development
    python -m vesta.spine.cups_candidate_verdict <football-world.db> --stage blind
        --configuration <name>

The plan, declared before any run on 1 November 2026. Three
nested configurations of `vesta.spine.cups` (V: the venue by phase; V+L: and the second leg's
state; V+L+K: and the cups' calibration with spread and dispersion), each fitted walk-forward as
the champion (2.6.0's cups model, every switch off), each judged against it by the family rule on
the 90-minute result's log loss: the paired block bootstrap interval below zero, the seasons
improved (3 of 5 in development, 2 of 2 blind), the expected calibration error no worse than the
champion's by more than `[verdict] calibration_tolerance`; a gain over `[alarm.ordered] max_gain`
stops the configuration. The choice among those that pass: the lowest mean log loss, a larger
configuration only when lower than the smaller by at least `CHOICE_MARGIN`. Reported beside, not
deciding: the score's log loss and the log loss by stage.

The blind reads 2024-2025 a second time for the cups, for this candidate only and once (1
November 2026): it judges the configuration the development chose, and a
second blind run on the same commit is refused. Nothing is published and no tag is made.
"""

import argparse
import math
import sys
import time
import tomllib
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from vesta.research.bootstrap import assign_block, bootstrap_mean
from vesta.research.calibration import measure_calibration_error
from vesta.research.repository import run_git
from vesta.spine.cups import ClubMatch, CupsParameters, forecast_cups
from vesta.spine.match_forecast import RESULTS
from vesta.spine.score_matrix import score_matrices
from vesta.spine.world import open_world, read_club_matches
from vesta.verdicts import Rules, _passes, hash_world

ROOT = Path(__file__).resolve().parents[3]
CONFIGURATIONS = {
    "V": {"venue_by_phase": True},
    "V+L": {"venue_by_phase": True, "second_leg_state": True},
    "V+L+K": {"venue_by_phase": True, "second_leg_state": True, "calibration_spread": True},
}
CHOICE_MARGIN = 0.0005  # the plan: a larger configuration must be lower by at least this much
SMALLEST = 1e-15
REPORT = "docs/research/2026-11-01-cups-v3-{stage}.md"


@dataclass(frozen=True)
class Judged:
    name: str
    matches: int
    model: float
    champion: float
    mean: float
    low: float
    high: float
    improved: int
    calibration_model: float
    calibration_champion: float
    score_model: float
    score_champion: float
    alarm: bool
    passed: bool
    by_stage: dict[str, tuple[int, float, float]]


def classify_stage(match: ClubMatch) -> str:
    """The stage the plan's table reads: qualifying, first leg, second leg, single match (a tie's
    one match or neutral ground), league or group phase."""
    if match.qualifying:
        return "qualifying"
    if match.leg == 1:
        return "first leg"
    if match.leg == 2:
        return "second leg"
    if match.tie_id is not None or match.neutral:
        return "single match"
    return "league or group phase"


def _score_loss(forecast, match: ClubMatch) -> float:
    matrix = score_matrices(np.array([forecast.home_intensity]),
                            np.array([forecast.away_intensity]),
                            forecast.calibration)[0]  # fmt: skip
    top = matrix.shape[0] - 1
    cell = matrix[min(match.home_goals, top), min(match.away_goals, top)]
    return -math.log(max(float(cell), SMALLEST))


def judge(name: str, candidate: list, champion: list, matches: dict[int, ClubMatch],
          seasons: tuple[int, ...], rules: Rules, max_gain: float) -> Judged:  # fmt: skip
    """One configuration against the champion, match by match (the same matches)."""
    theirs = {forecast.match_id: forecast for forecast in champion if forecast.season in seasons}
    rows = [(forecast, theirs[forecast.match_id]) for forecast in candidate
            if forecast.season in seasons and forecast.match_id in theirs]  # fmt: skip
    if len(rows) != len(theirs):
        raise ValueError(f"{name}: {len(rows)} matches against the champion's {len(theirs)}")
    happened = [RESULTS.index(mine.result) for mine, _ in rows]
    model = [-math.log(max(mine.probabilities[index], SMALLEST))
             for (mine, _), index in zip(rows, happened, strict=True)]  # fmt: skip
    base = [-math.log(max(other.probabilities[index], SMALLEST))
            for (_, other), index in zip(rows, happened, strict=True)]  # fmt: skip
    differences = [model_score - base_score
                   for model_score, base_score in zip(model, base, strict=True)]
    blocks = [assign_block(mine.competition, mine.forecast_time) for mine, _ in rows]
    mean, low, high = bootstrap_mean(differences, blocks, rules.repetitions, rules.seed)
    improved = sum(math.fsum(d for (mine, _), d in zip(rows, differences, strict=True)
                             if mine.season == season) < 0 for season in seasons)  # fmt: skip
    calibration_model = measure_calibration_error(
        [(mine.probabilities[i], index == i)
         for (mine, _), index in zip(rows, happened, strict=True) for i in range(3)])  # fmt: skip
    calibration_champion = measure_calibration_error(
        [(other.probabilities[i], index == i)
         for (_, other), index in zip(rows, happened, strict=True) for i in range(3)])  # fmt: skip
    score_model = math.fsum(_score_loss(mine, matches[mine.match_id]) for mine, _ in rows)
    score_champion = math.fsum(_score_loss(other, matches[other.match_id]) for _, other in rows)
    stages: dict[str, list] = defaultdict(list)
    for (mine, _), model_score, base_score in zip(rows, model, base, strict=True):
        stages[classify_stage(matches[mine.match_id])].append((model_score, base_score))
    by_stage = {stage: (len(pairs), math.fsum(model_score for model_score, _ in pairs) / len(pairs),
                        math.fsum(base_score for _, base_score in pairs) / len(pairs))
                for stage, pairs in sorted(stages.items())}  # fmt: skip
    alarm = -mean > max_gain
    passed = not alarm and _passes(high, improved, calibration_model, calibration_champion, rules)
    return Judged(name, len(rows), math.fsum(model) / len(rows), math.fsum(base) / len(rows),
                  mean, low, high, improved, calibration_model, calibration_champion,
                  score_model / len(rows), score_champion / len(rows), alarm, passed,
                  by_stage)  # fmt: skip


def choose(judged: list[Judged]) -> Judged | None:
    """The plan's choice: among the configurations that pass, in nesting order, a larger one only
    when its mean log loss is lower than the chosen smaller one's by at least `CHOICE_MARGIN`."""
    chosen = None
    for item in judged:
        if not item.passed:
            continue
        if chosen is None or item.model <= chosen.model - CHOICE_MARGIN:
            chosen = item
    return chosen


def run(world: Path, stage: str, configuration: str | None, say=print) -> str:
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    research = tomllib.loads((ROOT / "config/vesta/research.toml").read_text(encoding="utf-8"))
    rules = Rules.from_config(research)
    if stage == "blind":
        seasons = tuple(research["seasons"]["closed"])
        rules = replace(rules, min_seasons_improved=int(research["blind"]["min_seasons_improved"]))
        names = [configuration]
    else:
        seasons = tuple(research["seasons"]["development"])
        names = list(CONFIGURATIONS)
    max_gain = float(research["alarm"]["ordered"]["max_gain"])
    commit = run_git(ROOT, "rev-parse", "HEAD").strip()
    started = time.monotonic()
    champion_parameters = CupsParameters.from_config(spine)
    with open_world(world) as connection:
        club = read_club_matches(connection, list(champion_parameters.leagues),
                                 list(champion_parameters.cups),
                                 champion_parameters.first_league_season,
                                 champion_parameters.first_cup_season, max(seasons))  # fmt: skip
    first = int(spine["walk_forward"]["first_training_season"])
    span = tuple(range(max(first, champion_parameters.first_cup_season + 1), max(seasons) + 1))
    matches = {match.match_id: match for match in club}
    say("the champion...")
    champion = forecast_cups(club, champion_parameters, span)
    judged = []
    for name in names:
        say(f"{name}...")
        parameters = replace(champion_parameters, **CONFIGURATIONS[name])
        judged.append(judge(name, forecast_cups(club, parameters, span), champion, matches,
                            seasons, rules, max_gain))  # fmt: skip
    chosen = choose(judged) if stage == "development" else None
    lines = [f"# The cups' spine candidate (cups v3): the {stage} verdict, "
             f"{datetime.now(UTC):%Y-%m-%d %H:%M} UTC", "",
             f"World: `{world}` (sha256 {hash_world(world)}). Code: `{commit}`. Seasons: "
             f"{', '.join(map(str, seasons))}. Written by `python -m"
             f" vesta.spine.cups_candidate_verdict --stage {stage}`; the plan"
             f" declared before the run."
             " Each configuration against the champion (2.6.0's cups model) on the 90-minute"
             " result; nothing published, no tag.", ""]  # fmt: skip
    if stage == "blind":
        lines += ["The second reading of 2024-2025 for the cups, for this candidate only and once"
                  " (1 November 2026); the forward guard (2026-27 cup matches"
                  " from 2 November, read once on the first world after 31 January 2027) binds.",
                  ""]  # fmt: skip
    lines += ["| configuration | matches | log loss | champion | difference [interval] |"
              " seasons improved | calibration / champion's | score log loss / champion's |"
              " alarm | verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
    for item in judged:
        lines.append(
            f"| {item.name} | {item.matches} | {item.model:.5f} | {item.champion:.5f} |"
            f" {item.mean:+.5f} [{item.low:+.5f}, {item.high:+.5f}] |"
            f" {item.improved}/{len(seasons)} | {item.calibration_model:.4f} /"
            f" {item.calibration_champion:.4f} | {item.score_model:.4f} /"
            f" {item.score_champion:.4f} | {'STOP' if item.alarm else 'no'} |"
            f" {'pass' if item.passed else 'fail'} |")  # fmt: skip
    lines += ["", "By stage (log loss of the result, configuration / champion):", "",
              "| configuration | " + " | ".join(judged[0].by_stage) + " |",
              "|---|" + "---|" * len(judged[0].by_stage)]  # fmt: skip
    for item in judged:
        lines.append(f"| {item.name} | " + " | ".join(
            f"{model:.4f} / {base:.4f} (n {count})"
            for count, model, base in item.by_stage.values()) + " |")  # fmt: skip
    lines.append("")
    if stage == "development":
        lines.append(f"**The choice by the plan's rule: {chosen.name}.**" if chosen else
                     "**No configuration passes: the champion stays.**")  # fmt: skip
    else:
        lines.append(f"**{judged[0].name}: {'PASSED' if judged[0].passed else 'NOT PASSED'}"
                     " blind.**")  # fmt: skip
    lines += ["", f"{time.monotonic() - started:.0f} seconds"]
    return "\n".join(lines) + "\n"


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("world", type=Path)
    parser.add_argument("--stage", choices=("development", "blind"), required=True)
    parser.add_argument("--configuration", choices=tuple(CONFIGURATIONS))
    options = parser.parse_args(arguments)
    if options.stage == "blind" and options.configuration is None:
        parser.error("the blind judges the configuration the development chose: --configuration")
    out = ROOT / REPORT.format(stage=options.stage)
    if options.stage == "blind" and out.exists():
        print(f"refused: the blind was run already ({out}); 2024-2025 are not read again")
        return 3
    text = run(options.world, options.stage, options.configuration)
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
