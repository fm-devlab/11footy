"""The hindsight gain of family A's counts, the measure of their alarm
(family A's plan, §8 and §11), on 2017 and 2018, seasons no verdict
reads.

    python -m vesta.team_counts.hindsight <football-world.db> [--references all,2,3]
                                          [--out <report.md>]

The candidate fitted on the very season it scores: in the leagues one fit of `make` and `allow`
per league on all the season's matches (no window, no decay), in the cups the pooled fit on all
the season's league and cup matches; then the slopes and the tilt fitted in-sample on the same
forecasts. Its gain over the reference (the competition's frequencies of the seasons before, the
verdict's) is the most a forecast made before the kick-off could reach; the proposed limit of a key
is that gain, never below the kind's floor, and a joint key's at least the sum of its two sides'.
`--references` measures it against the competition's last seasons only too (R2, R3).
"""

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

from vesta.determinism import run_on_one_thread
from vesta.quantities import load_quantities
from vesta.research.repository import run_git
from vesta.spine.world import open_world
from vesta.team_counts import cups
from vesta.team_counts.counts_verdict import (
    SCOPES,
    _load_toml,
    build_history,
    hash_file,
    judge,
    read_scope,
)
from vesta.team_counts.model import (
    FAMILY_A_KEYS,
    FitParameters,
    fit_calibration,
    fit_week,
    seconds_of,
)

SEASONS = (2017, 2018)
FLOORS = {"count": 0.10, "joint_count": 0.15}


def forecast_leagues_in_hindsight(matches: list, parameters: FitParameters, season: int) -> list:
    """Each league's matches of `season` forecast by one fit on all of them."""
    whole = FitParameters(10**6, 1e15, parameters.ridge, 1, parameters.size)
    by_league = defaultdict(list)
    for match in matches:
        if match.season == season and match.home_count is not None:
            by_league[match.competition].append(match)
    forecasts = []
    for rows in by_league.values():
        week = max(seconds_of(match.known_at) for match in rows) + 1
        fit = fit_week(rows, week, whole)
        forecasts += [
            (match, fit.predict_means(match.home_team_id, match.away_team_id), fit.size_nb)
            for match in rows
        ]
    return forecasts


@run_on_one_thread
def run(world_path: Path, say=print, windows: tuple[int | None, ...] = (None,),
        ) -> dict[tuple[str, str, int | None], dict[str, tuple[int, float]]]:  # fmt: skip
    """Per (scope, count, reference window): per key, (forecasts, hindsight gain over the
    reference); a window None reads every season before."""
    spine, config = (
        _load_toml("config/vesta/spine.toml"),
        _load_toml("config/vesta/team_counts.toml"),
    )
    first_training = int(spine["walk_forward"]["first_training_season"])
    definitions = load_quantities()
    found = {}
    for scope in SCOPES:
        for count in FAMILY_A_KEYS:
            parameters = FitParameters.from_config(config, count)
            with open_world(world_path) as world:
                matches = read_scope(world, scope, count, spine, first_training, max(SEASONS))
            history = build_history(matches, scope, count, parameters.size)
            rows: dict[int | None, dict[str, list]] = {window: defaultdict(list)
                                                       for window in windows}  # fmt: skip
            for season in SEASONS:
                forecasts = (
                    forecast_leagues_in_hindsight(matches, parameters, season)
                    if scope == "leagues"
                    else cups.forecast_in_hindsight(matches, parameters, season)
                )
                calibration = fit_calibration(forecasts, parameters.size)  # in-sample
                for window in windows:
                    judged = judge(count, scope, forecasts, {season: calibration}, history,
                                   first_training, parameters.size, definitions,
                                   window=window)  # fmt: skip
                    for key, scored in judged.rows.items():
                        rows[window][key] += scored
            for window in windows:
                found[(scope, count, window)] = {
                    key: (len(scored), math.fsum(row[1] - row[0] for row in scored) / len(scored))
                    for key, scored in rows[window].items()}  # fmt: skip
                say(f"{scope} {count} (reference {window or 'all'}):"
                    f" {found[(scope, count, window)]}")
    return found


def propose_limits(gains: dict[str, tuple[int, float]], count: str) -> dict[str, float]:
    total_key, team_key, sides_key = FAMILY_A_KEYS[count]
    proposed = {total_key: max(FLOORS["count"], gains[total_key][1]),
                team_key: max(FLOORS["count"], gains[team_key][1])}  # fmt: skip
    if sides_key is not None:
        proposed[sides_key] = max(FLOORS["joint_count"], gains[sides_key][1],
                                  2 * proposed[team_key])  # fmt: skip
    return proposed


def report(found: dict, world_path: Path, world_sha: str, commit: str, seconds: float) -> str:
    lines = ["# Team counts, family A: the hindsight gains, the measure of the alarm", "",
             f"World: `{world_path}` (sha256 {world_sha}). Code: `{commit}`. Written by"
             f" `python -m vesta.team_counts.hindsight`; the method is"
             f" family A's plan, §8 and §11. Seasons 2017 and 2018"
             f" (no verdict reads them). {seconds:.0f} seconds.", "",
             "Hindsight gain: the reference's mean log score minus the in-sample candidate's."
             " Proposed limit: the gain, at least the floor (0.10 count, 0.15 joint), a joint"
             " key's at least twice its team key's limit.", "",
             "Reference: all, every season before from the first training season (the"
             " plan's); R2 and R3, the competition's last two or three seasons.", "",
             "| scope | key | reference | forecasts | hindsight gain | floor | proposed limit |",
             "|---|---|---|---|---|---|---|"]  # fmt: skip
    definitions = load_quantities()
    for (scope, count, window), gains in found.items():
        proposed = propose_limits(gains, count)
        for key, (forecasts, gain) in gains.items():
            kind = definitions[key].quantity.distribution_kind
            lines.append(f"| {scope} | {key} | {f'R{window}' if window else 'all'} | {forecasts} |"
                         f" {gain:+.4f} | {FLOORS[kind]:.2f} | {proposed[key]:.4f} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("world", type=Path)
    parser.add_argument("--references", default="all",
                        help='comma-separated: "all" the seasons before, or a number of seasons')
    parser.add_argument("--out", type=Path, default=None)
    options = parser.parse_args(arguments)
    windows = tuple(None if item == "all" else int(item)
                    for item in options.references.split(","))  # fmt: skip
    started = time.monotonic()
    found = run(options.world, windows=windows)
    root = Path(__file__).resolve().parents[3]
    text = report(found, options.world, hash_file(options.world),
                  run_git(root, "rev-parse", "HEAD").strip(), time.monotonic() - started)
    out = options.out or root / "docs/research/2026-10-31-team-counts-hindsight.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
