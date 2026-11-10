"""The verdict of the player thresholds, declared before the run in
its plan (5f4c62c), its alarm in `[alarm.players]`
(`max_gain_thresholds` for the counts, `max_gain_binary` for the booking).

    python -m vesta.players.thresholds_verdict <football-world.db> [--stage development|blind]
        [--out <md>]

Every player who played a league match of the stage's seasons, among the players the lineups knew
before it, is forecast walk-forward (`vesta.players.thresholds`): his rates, his minutes and chance
to play from the lineups' forecast (candidate 1, fitted on the five seasons before), the match's
terms from the champion and team counts' weekly fits. Each key is scored on its own against the
reference, his own count per match over his last 20 matches played. The deciding metric is the
log score of what happened. On the condition of 31 October, each key's gain is
also split into starters and substitutes (the minutes always the lineups' forecast), beside a
diagnostic that takes the minutes as played: never a forecast, never published.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import math
import sys
import time
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np

from vesta.determinism import run_on_one_thread
from vesta.lineups.model import collect_targets, read_world, scale_to_eleven
from vesta.lineups.verdict import TRAINING_SEASONS, Priors, estimate_minutes
from vesta.players.model import compute_side_means
from vesta.players.thresholds import (
    COUNTS,
    KEYS,
    MatchTerms,
    applies_to_player,
    build_histories,
    count_cells,
    estimate_own_rates,
    estimate_player_means,
    estimate_reference_rates,
    fit_sizes,
    group_rates,
    iterate_training_rows,
    make_poisson_cells,
    read_counts,
    split_into_minute_bins,
)
from vesta.players.verdict import _judge_key
from vesta.research.bootstrap import assign_block, bootstrap_mean
from vesta.research.repository import run_git
from vesta.spine.champion import forecast_world_with_champion
from vesta.spine.timing import TimeRules
from vesta.spine.world import open_world, read_league_matches
from vesta.team_counts.model import FitParameters, read_count_matches

ROOT = Path(__file__).resolve().parents[3]
SCALE_SEASONS = 2  # candidate 2: the seasons before the one judged its scale is fitted on
EPSILON = 1e-12


def _score_log_loss(probability: float) -> float:
    return -math.log(max(probability, EPSILON))


@run_on_one_thread
def run(world_path: Path, stage: str, say=print, keys: tuple[str, ...] | None = None,
        candidate: int = 1, scales_for: int | None = None) -> tuple[dict, dict]:  # fmt: skip
    """With `keys`, only those keys are scored: the others' outcomes are never read (a key's
    closed seasons stay unread until it passes development). Candidate 2 scales each key's
    expected counts by what happened over what candidate 1 expected in the two seasons before
    the one judged (`SCALE_SEASONS`), forecast for that only, never scored. With `scales_for`, a
    season to publish, nothing is scored: candidate 2's scales of that season are fitted on the
    two seasons before it and returned (`config/vesta/thresholds-scales.toml`)."""
    judged = set(keys or KEYS)
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    research = tomllib.loads((ROOT / "config/vesta/research.toml").read_text(encoding="utf-8"))
    counts_config = tomllib.loads((ROOT / "config/vesta/team_counts.toml").read_text(
        encoding="utf-8"))  # fmt: skip
    seasons = sorted(research["seasons"]["development" if stage == "development" else "closed"])
    needed = 3 if stage == "development" else int(research["blind"]["min_seasons_improved"])
    repetitions = int(research["verdict"]["bootstrap_repetitions"])
    seed = int(research["verdict"]["seed"])
    alarm = research["alarm"]["players"]
    leagues = list(spine["walk_forward"]["competitions"])
    rules = TimeRules.from_config(spine)
    forecast_seasons = list(range(seasons[0] - (SCALE_SEASONS if candidate == 2 else 0),
                                  seasons[-1] + 1))  # fmt: skip
    if scales_for is not None:
        seasons = [scales_for]
        forecast_seasons = list(range(scales_for - SCALE_SEASONS, scales_for))
    first = forecast_seasons[0] - TRAINING_SEASONS
    forecasts = forecast_world_with_champion(world_path, tuple(range(forecast_seasons[0] - 1,
                                                                     seasons[-1] + 1)))  # fmt: skip
    say(f"champion: {len(forecasts)} matches forecast")
    fit = counts_config["fit"]
    parameters = FitParameters(int(fit["window_days"]), float(fit["half_life_days"]),
                               float(fit["ridge"]), int(fit["minimum_matches"]), 11)  # fmt: skip
    with open_world(world_path) as connection:
        matches = read_league_matches(connection, leagues, first - 1, seasons[-1])
        by_id = {match.match_id: match for match in matches}
        world = read_world(connection, first - 1, seasons[-1], rules)
        counted = read_counts(connection, world.sheets, set(by_id))
        team_counts = {count: read_count_matches(connection, leagues, forecast_seasons[0] - 3,
                                                 seasons[-1], count, rules)
                       for count in ("fouls", "cards")}  # fmt: skip
        yellow, red = connection.execute(
            "SELECT SUM(t.yellow_cards), SUM(t.red_cards) FROM match_team_statistics t JOIN"
            " matches m ON m.match_id = t.match_id JOIN seasons s ON s.season_id = m.season_id"
            " JOIN competitions c ON c.competition_id = s.competition_id WHERE t.layer = 'feed'"
            f" AND c.code IN ({', '.join('?' * len(leagues))}) AND s.season BETWEEN ? AND ?",
            (*leagues, first, seasons[0] - 1)).fetchone()  # fmt: skip
    yellow_share = yellow / max(yellow + red, 1)
    say(f"world read: {len(matches)} league matches, {len(counted)} player counts")
    terms = MatchTerms()
    ordered = sorted((by_id[item.match_id].kickoff_utc, item.match_id, item) for item in forecasts
                     if item.match_id in by_id)  # fmt: skip
    terms.set_attack_factors(
        [
            (
                match_id,
                by_id[match_id].home_team_id,
                by_id[match_id].away_team_id,
                *compute_side_means(item.matrix),
            )
            for _, match_id, item in ordered
        ]
    )
    for count, found in team_counts.items():
        terms.fit_count_factors(count, found, parameters, set(forecast_seasons))
    say("match terms ready")
    histories, groups_of = build_histories(world, by_id, counted)
    known_at = {player: [item.known_at for item in items] for player, items in histories.items()}
    by_season_rows = {}
    for season in range(first, seasons[-1] + 1):
        by_season_rows[season] = collect_targets(world, set(leagues), {season})
    forecast_of = {item.match_id: item for item in forecasts}
    scored: dict[str, list] = defaultdict(list)
    lines: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    coherence = defaultdict(lambda: np.zeros(2))  # key -> (expected, happened)
    cards = np.zeros(2)  # the booked players expected, the side's yellows expected
    dispersions = {}
    unscored = pending = 0
    fit_sums: dict[int, np.ndarray] = {}  # season -> per key (candidate 1 expected, happened)
    scales: dict[int, dict[str, float]] = {}
    for season in forecast_seasons:
        judging = season in seasons
        before = [fit_sums[year] for year in range(season - SCALE_SEASONS, season)
                  if year in fit_sums]  # fmt: skip
        if candidate == 2 and judging:
            totals = sum(before, np.zeros((len(KEYS), 2)))
            scales[season] = {key: float(totals[index, 1] / totals[index, 0])
                              if totals[index, 0] > 0 else 1.0
                              for index, key in enumerate(KEYS)}  # fmt: skip
        scale = np.array([scales.get(season, {}).get(key, 1.0) for key in KEYS])
        fit_sums[season] = np.zeros((len(KEYS), 2))
        training = range(season - TRAINING_SEASONS, season)
        lineup_priors = Priors([row for year in training for row in by_season_rows[year][0]],
                               [team for year in training for team in by_season_rows[year][1]])
        rates = group_rates([(groups_of[(player, index)], item)
                             for player, items in histories.items()
                             for index, item in enumerate(items) if item.season in training])
        sizes = fit_sizes(iterate_training_rows(histories, groups_of, set(training), rates), rates)
        dispersions[season] = sizes
        grouped = defaultdict(list)
        for row in by_season_rows[season][0]:
            grouped[(row.match_id, row.team_id)].append(row)
        played_rows = 0
        for (match_id, team_id), pool in grouped.items():
            if match_id not in forecast_of or match_id not in by_id:
                continue
            done, missing = _score_side(pool, match_id, team_id, terms, lineup_priors, rates,
                                        sizes, histories, known_at, counted, scored, lines, judged,
                                        coherence, cards, yellow_share, scale,
                                        fit_sums[season], judging)  # fmt: skip
            played_rows += done
            pending += missing
        sheet_played = sum(1 for match in matches if match.season == season
                           for side in (match.home_team_id, match.away_team_id)
                           for started, minutes, _ in world.sheets.get((match.match_id, side),
                                                                       {}).values()
                           if started or minutes > 0)  # fmt: skip
        if not judging:
            say(f"{season}: forecast for the scale only, not scored")
            continue
        unscored += sheet_played - played_rows
        say(f"{season}: {played_rows} players scored of {sheet_played} who played")
    if scales_for is not None:
        totals = sum((fit_sums[year] for year in forecast_seasons), np.zeros((len(KEYS), 2)))
        return {"scales": {key: float(totals[index, 1] / totals[index, 0])
                           for index, key in enumerate(KEYS)
                           if key in judged and totals[index, 0] > 0}}, {}  # fmt: skip
    results, splits = {}, {}
    for key, threshold in KEYS.items():
        if key not in judged:
            continue
        limit = float(alarm["max_gain_binary" if threshold.kind == "binary"
                            else "max_gain_thresholds"])  # fmt: skip
        rows = scored[key]
        results[key] = _judge_key(key, threshold.kind, [row[:4] for row in rows], lines[key],
                                  repetitions, seed, needed, limit)  # fmt: skip
        splits[key] = {}
        for name, chosen in (("starters", [row for row in rows if row[4]]),
                             ("substitutes", [row for row in rows if not row[4]])):
            if chosen:
                splits[key][name] = (len(chosen), *bootstrap_mean(
                    [row[0] - row[1] for row in chosen], [row[3] for row in chosen],
                    repetitions, seed))  # fmt: skip
        splits[key]["as_played"] = math.fsum(row[5] - row[1] for row in rows) / len(rows)
    summary = {"coherence": {key: tuple(value) for key, value in coherence.items()},
               "cards": tuple(cards), "unscored": unscored, "pending": pending,
               "dispersions": dispersions, "yellow_share": yellow_share,
               "candidate": candidate, "scales": scales}  # fmt: skip
    return {"results": results, "splits": splits}, summary


def _score_side(pool, match_id, team_id, terms, lineup_priors, rates, sizes, histories, known_at,
                counted, scored, lines, judged, coherence, cards, yellow_share, scale, fit_sums,
                judging) -> tuple[int, int]:  # fmt: skip
    """Scores the side's players who played (when `judging`); returns how many, and how many
    counts were pending (the source holds none). Every season adds candidate 1's expected
    counts and what happened, of the players who played, to `fit_sums` (candidate 2's scale)."""
    design = np.column_stack([np.ones(len(pool)), [row.features for row in pool]])
    starts = scale_to_eleven(
        design @ lineup_priors.weights, np.array([row.goalkeeper for row in pool])
    )
    count = missing = 0
    keys = list(KEYS)
    side = np.zeros((len(KEYS), 2))  # each key: expected over the pool, happened
    complete = np.ones(len(KEYS), bool)  # every player who played has the count
    for row, start in zip(pool, starts, strict=True):
        comes_on, minutes = estimate_minutes(lineup_priors, row, float(start))
        plays = float(start) + (1 - float(start)) * comes_on
        weights, bin_minutes = split_into_minute_bins(minutes)
        history = histories.get(row.player_id, [])
        cut = bisect.bisect_right(known_at.get(row.player_id, []), row.forecast_time)
        known = history[:cut]
        group_rates_of = rates.get(row.group, rates[None])
        per_ninety, ratio = estimate_own_rates(known, row.season, group_rates_of)
        means = estimate_player_means(per_ninety, ratio, terms, match_id, team_id, bin_minutes)
        raw = means @ weights  # candidate 1's expected count of each key if he plays
        means = means * scale[:, None]
        expected = means @ weights
        did = counted.get((match_id, row.player_id))
        played = row.started or row.minutes > 0
        if played and did is not None:
            for index, (key, threshold) in enumerate(KEYS.items()):
                value = did[0][COUNTS.index(threshold.count)]
                if (key in judged and applies_to_player(threshold.who, row.goalkeeper)
                        and not math.isnan(value)):  # fmt: skip
                    fit_sums[index] += (raw[index], value)
        if not judging:
            continue
        for index, (key, threshold) in enumerate(KEYS.items()):
            if key not in judged or not applies_to_player(threshold.who, row.goalkeeper):
                continue
            value = did[0][COUNTS.index(threshold.count)] if did else math.nan
            side[index, 0] += plays * expected[index]
            if played:
                if math.isnan(value):
                    complete[index] = False
                else:
                    side[index, 1] += value
        if not played:
            continue  # if he plays: void otherwise
        count += 1
        reference = estimate_reference_rates(known, group_rates_of)
        block = assign_block(row.competition, row.forecast_time)
        for index, (key, threshold) in enumerate(KEYS.items()):
            if key not in judged or not applies_to_player(threshold.who, row.goalkeeper):
                continue
            position = COUNTS.index(threshold.count)
            value = did[0][position] if did else math.nan
            if math.isnan(value):
                missing += 1
                continue
            size = sizes.get((threshold.count, row.group), sizes.get((threshold.count, None),
                                                                     math.inf))  # fmt: skip
            own = count_cells(means[index], weights, size, threshold.cells)
            per_match = float(means[index][-1])  # the 90-minute bin's mean
            as_played = count_cells(np.array([per_match * min(row.minutes or 90, 90) / 90]),
                                    np.array([1.0]), size, threshold.cells)  # fmt: skip
            against = make_poisson_cells(float(reference[position]), threshold.cells)
            cell = min(int(value), threshold.cells - 1)
            scored[key].append(
                (
                    _score_log_loss(own[cell]),
                    _score_log_loss(against[cell]),
                    row.season,
                    block,
                    row.started,
                    _score_log_loss(as_played[cell]),
                )
            )
            for at_least in threshold.lines:
                lines[key][0].append((float(own[at_least:].sum()), value >= at_least))
                lines[key][1].append((float(against[at_least:].sum()), value >= at_least))
    if not judging:
        return count, missing
    for index, key in enumerate(keys):
        if complete[index] and side[index, 1] + side[index, 0] > 0:
            coherence[key] += side[index]
    side_cards = terms.means.get((match_id, team_id), {}).get("cards")
    if side_cards is not None:
        cards += (side[keys.index("P9.booked"), 0], side_cards * yellow_share)
    return count, missing


def report(found, summary, stage, world_path, world_sha, commit, seconds) -> str:
    results, splits = found["results"], found["splits"]
    lines = [f"# Player thresholds: the {stage} verdicts (candidate {summary['candidate']})", "",
             f"Keys judged: {', '.join(results)}.", "",
             f"World: `{world_path}` (sha256 {world_sha}). Code: `{commit}`. Written by"
             f" `python -m vesta.players.thresholds_verdict`; the rule is"
             f" the plan declared before the run (5f4c62c), the alarm"
             f" `[alarm.players]` (`max_gain_thresholds` 0.10 for the counts, `max_gain_binary`"
             f" 0.03 for P9.booked, approved on 31 October 2026)."
             f" {seconds:.0f} seconds.", "",
             "Log score of what happened, model against the reference (the player's own count per"
             " match over his last 20 matches played, shrunk to his group's, Poisson; lower is"
             " better), on every player who played; the interval is the block bootstrap of the"
             " difference. A count is scored on its cells (the last one that many or more), its"
             " calibration on the lines listed.", "",
             "| key | n | model | reference | difference [95%] | seasons improved |"
             " calibration model / reference | verdict |",
             "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for item in results.values():
        verdict = "ALARM" if item.alarm else "pass" if item.passed else "fail"
        lines.append(f"| {item.key} | {item.forecasts} | {item.model:.4f} | {item.reference:.4f}"
                     f" | {item.difference:+.4f} [{item.low:+.4f}, {item.high:+.4f}] |"
                     f" {item.seasons_improved}/{item.seasons} | {item.calibration_model:.4f} /"
                     f" {item.calibration_reference:.4f} | {verdict} |")  # fmt: skip
    lines += ["", "## The gain split (the condition of 31 October)", "",
              "The model's minutes are always the lineups' forecast (his chance to start and his"
              " minutes if he plays, known before the match); who started is used only to split"
              " the rows after scoring. The last column is a diagnostic, never a forecast: the same"
              " rates with the minutes as played. A gain that came from known minutes would sit"
              " there, not in the model's columns.", "",
              "| key | starters: n, difference [95%] | substitutes: n, difference [95%] |"
              " minutes as played (diagnostic) |", "|---|---|---|---|"]  # fmt: skip
    for key, split in splits.items():
        cells = []
        for name in ("starters", "substitutes"):
            if name in split:
                players, mean, low, high = split[name]
                cells.append(f"{players}, {mean:+.4f} [{low:+.4f}, {high:+.4f}]")
            else:
                cells.append("none")
        lines.append(f"| {key} | {cells[0]} | {cells[1]} | {split['as_played']:+.4f} |")
    lines += ["", "## Coherence (tests, not verdicts)", "",
              "Each key's expected count summed over the players who played (each weighted by his"
              " chance to play), against what they did, over the stage's seasons (the plan's bound:"
              " 5 %).", "", "| key | expected | happened | gap |", "|---|---|---|---|"]  # fmt: skip
    for key, (expected, happened) in summary["coherence"].items():
        gap = expected / happened - 1 if happened else float("nan")
        lines.append(f"| {key} | {expected:.0f} | {happened:.0f} | {gap:+.1%} |")
    booked, side = summary["cards"]
    lines += ["", f"The booked players expected (every player of the pools, weighted by the chance"
              f" to play) against the sides' expected cards of team counts' fits times the yellows'"
              f" share ({summary['yellow_share']:.3f}): {booked:.0f} against {side:.0f}"
              f" ({(booked / side - 1) if side else float('nan'):+.1%}; the plan's bound: 5 %).",
              "", "## The dispersions (negative binomial size per count and group, inf = Poisson)",
              ""]  # fmt: skip
    for season, sizes in summary["dispersions"].items():
        text = ", ".join(f"{count}/{group or 'unknown'} {size:g}"
                         for (count, group), size in sorted(sizes.items(),
                                                            key=lambda item: str(item[0])))
        lines.append(f"- {season}: {text}")
    if summary["scales"]:
        lines += ["", "## Candidate 2's scales (what happened over candidate 1's expected"
                  " counts, the two seasons before)", ""]
        for season, found_scales in summary["scales"].items():
            text = ", ".join(f"{key} {found_scales[key]:.4f}" for key in results)
            lines.append(f"- {season}: {text}")
    lines += ["", f"Players who played but were outside the pool the lineups knew before the match"
              f" (not scored): {summary['unscored']}. Counts the source does not hold (pending, not"
              f" scored): {summary['pending']}."]  # fmt: skip
    return "\n".join(lines) + "\n"


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("world", type=Path)
    parser.add_argument("--stage", choices=("development", "blind"), default="development")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--keys", nargs="+", choices=list(KEYS), default=None)
    parser.add_argument("--candidate", type=int, choices=(1, 2), default=1)
    parser.add_argument("--scales-for", type=int, default=None,
                        help="fit candidate 2's scales of a season to publish; scores nothing")
    options = parser.parse_args(arguments)
    started = time.monotonic()
    if options.scales_for is not None:
        found, _ = run(options.world, options.stage, keys=options.keys, candidate=2,
                       scales_for=options.scales_for)  # fmt: skip
        print(f"[{options.scales_for}]")
        for key, value in found["scales"].items():
            print(f'"{key}" = {value:.6f}')
        return 0
    found, summary = run(options.world, options.stage, keys=options.keys,
                         candidate=options.candidate)  # fmt: skip
    digest = hashlib.sha256()
    with options.world.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 24), b""):
            digest.update(chunk)
    text = report(found, summary, options.stage, options.world, digest.hexdigest(),
                  run_git(ROOT, "rev-parse", "HEAD").strip(), time.monotonic() - started)
    suffix = "" if options.candidate == 1 else f"-candidate-{options.candidate}"
    out = options.out or ROOT / (f"docs/research/2026-10-31-player-thresholds-{options.stage}"
                                 f"{suffix}.md")  # fmt: skip
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
