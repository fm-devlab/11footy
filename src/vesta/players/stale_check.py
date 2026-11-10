"""The check of the scorers on stale shots, declared before its run in its
plan (4 November 2026).

    python -m vesta.players.stale_check <football-world.db> [--out <md>]

The scorers' verdict harness on seasons 2025 and 2026, once, with the shots of each side's last k
league matches left out of its players' own rates, for k = 0 to 4 (`vesta.players.verdict`). A k
passes when, for every published key, the model on stale shots beats the reference in each
season, loses at most a quarter of the full model's margin over the reference, and its
calibration is no worse than the reference's by more than the verdict's tolerance. N is the
largest k with every k from 1 to N passing.
"""

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

from vesta.players.verdict import CALIBRATION_TOLERANCE, KINDS, run
from vesta.research.calibration import measure_calibration_error

ROOT = Path(__file__).resolve().parents[3]
SEASONS = [2025, 2026]
DROPS = (0, 1, 2, 3, 4)
SHARE_OF_MARGIN = 0.25


def judge(by_drop: dict) -> tuple[list[str], int]:
    """The table's rows and N."""
    full_scored, _ = by_drop[0]
    rows, passing = [], {}
    for drop in DROPS:
        scored, lines = by_drop[drop]
        passing[drop] = True
        for key in KINDS:
            items, full = scored[key], full_scored[key]
            mean = math.fsum(row[0] for row in items) / len(items)
            full_mean = math.fsum(row[0] for row in full) / len(full)
            margin = abs(math.fsum(row[0] - row[1] for row in full) / len(full))
            by_season: dict[int, float] = defaultdict(float)
            for own, naive, season, _ in items:
                by_season[season] += own - naive
            calibration = measure_calibration_error(lines[key][0])
            reference = measure_calibration_error(lines[key][1])
            seasons_ok = all(by_season[season] < 0 for season in SEASONS)
            loss_ok = mean - full_mean <= SHARE_OF_MARGIN * margin
            calibration_ok = calibration <= reference + CALIBRATION_TOLERANCE
            ok = drop == 0 or (seasons_ok and loss_ok and calibration_ok)
            passing[drop] = passing[drop] and ok
            rows.append(f"| {drop} | {key} | {len(items)} | {mean:.5f} | {mean - full_mean:+.5f} |"
                        f" {SHARE_OF_MARGIN * margin:.5f} | "
                        + " / ".join(f"{by_season[season]:+.1f}" for season in SEASONS)
                        + f" | {calibration:.4f} / {reference:.4f} |"
                        f" {'—' if drop == 0 else 'pass' if ok else 'fail'} |")  # fmt: skip
    safe_drop = 0
    for drop in DROPS[1:]:
        if not passing[drop]:
            break
        safe_drop = drop
    return rows, safe_drop


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("world", type=Path)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "docs/research/2026-11-04-scorers-stale-shots-check.md")
    options = parser.parse_args(arguments)
    started = time.monotonic()
    by_drop: dict = {}
    run(options.world, "blind", seasons=SEASONS, drops=DROPS, by_drop=by_drop)
    rows, safe_drop = judge(by_drop)
    text = "\n".join([
        "# The scorers on stale shots: the check", "",
        f"World: `{options.world}`. Seasons {SEASONS[0]} and {SEASONS[1]}. Written by"
        f" `python -m vesta.players.stale_check`; the rule is"
        f" the plan declared before the run."
        f" {time.monotonic() - started:.0f} seconds.", "",
        "k: the shots of the side's last k league matches left out. Mean log score of the model;"
        " its change from full data (k = 0) against the allowed change (a quarter of the full"
        " model's margin over the reference); the summed difference to the reference per season;"
        " calibration, model on stale shots / reference.", "",
        "| k | key | n | model | change | allowed | vs reference 2025 / 2026 |"
        " calibration | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
        *rows, "",
        f"**N = {safe_drop}**"
        + (" (k = 1 failed: the scorers keep holding)" if safe_drop == 0 else ""), ""])
    options.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
