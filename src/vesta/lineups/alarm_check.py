"""The leakage check of the lineups' alarms (L2.formation.team, L9.position, development of
29 October 2026): the verdict run again with each team's history ending one match earlier, and the
gains of both runs by league and by season.

    python -m vesta.lineups.alarm_check <football-world.db> [--out <report.md>]

A gain that comes from the match itself (its sheet read by mistake) cannot survive a history that
stops one match before; a gain from the recent sheets shrinks a little and stays.
"""

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

from vesta.lineups.verdict import ROOT, run

KEYS = ("L2.formation.team", "L9.position", "L1.starts", "L6.minutes")


def average_gains(scored: dict, key: str) -> tuple[dict[str, float], dict[int, float]]:
    by_league, by_season = defaultdict(list), defaultdict(list)
    for own, naive, season, block in scored[key]:
        by_league[block.split("|")[0]].append(naive - own)
        by_season[season].append(naive - own)
    return ({league: math.fsum(found) / len(found) for league, found in sorted(by_league.items())},
            {season: math.fsum(found) / len(found)
             for season, found in sorted(by_season.items())})  # fmt: skip


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("world", type=Path)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "docs/research/2026-10-29-lineups-alarm-check-runs.md")
    options = parser.parse_args(arguments)
    lines = ["# Lineups: the alarm check's runs (development seasons)", "",
             "The gain is the reference's mean log score minus the model's (higher: the model"
             " better). Lag 0: the verdict as run; lag 1: each team's history ends one team match"
             " earlier, for the model and the reference alike.", ""]
    for lag in (0, 1):
        scored: dict = {}
        results, summary = run(options.world, "development", lag=lag, scored_out=scored)
        lines += [f"## Lag {lag}", "", "| key | model | reference | gain | by league | by season |",
                  "|---|---|---|---|---|---|"]
        for key in KEYS:
            item = results[key]
            league, season = average_gains(scored, key)
            lines.append(f"| {key} | {item.model:.4f} | {item.reference:.4f} |"
                         f" {-item.difference:.4f} |"
                         f" {', '.join(f'{k} {v:.3f}' for k, v in league.items())} |"
                         f" {', '.join(f'{k} {v:.3f}' for k, v in season.items())} |")
        lines += ["", f"Starters guessed: {summary}", ""]
    text = "\n".join(lines) + "\n"
    options.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
