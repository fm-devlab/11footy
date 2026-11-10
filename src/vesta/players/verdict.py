"""The verdict of the scorers, declared before the run in its plan
(8cec816), its alarm in `[alarm.players]`.

    python -m vesta.players.verdict <football-world.db> [--stage development|blind] [--out <md>]

Every player who played a league match of the stage's seasons, among the players the lineups
knew before it, is forecast walk-forward: his side's goals from the champion's score matrix, his
share of them from his rates (`vesta.players.model`), his minutes and chance to play from the
lineups (candidate 1, fitted on the five seasons before). Each key is scored on its own against
the reference, his own rate over his last 20 matches played. The deciding metric is the log score
of what happened; a key passes when its mean difference to the reference has a block bootstrap
interval entirely below zero, it improved in enough seasons, and its calibration error is no
worse than the reference's by more than the tolerance; the alarm stops a key whose gain is larger
than its kind's limit.
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
from vesta.players.model import (
    GroupRates,
    build_histories,
    estimate_league_rates,
    estimate_reference_rates,
    find_first_scoring_side,
    group_rates,
    make_poisson_cells,
    read_happened,
    read_penalty_conversion,
    read_shot_players,
    share_out_side_goals,
    sum_shots_by_player,
    weigh_player,
)
from vesta.research.bootstrap import assign_block, bootstrap_mean
from vesta.research.calibration import measure_calibration_error
from vesta.research.repository import run_git
from vesta.spine.champion import forecast_world_with_champion
from vesta.spine.timing import TimeRules
from vesta.spine.world import open_world, read_league_matches
from vesta.team_counts.verdict import KeyVerdict

ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_TOLERANCE = 0.005
EPSILON = 1e-12
KINDS = {"P3.scores": "binary", "P3.goals": "count", "P3.first_scorer": "binary",
         "P5.assists": "binary", "P5.scores_or_assists": "binary"}  # fmt: skip
GOAL_CELLS = 4  # 0, 1, 2, 3 or more


def _score_log_loss(probability: float) -> float:
    return -math.log(max(probability, EPSILON))


@run_on_one_thread
def run(world_path: Path, stage: str, say=print, scored_out: dict | None = None,
        seasons: list[int] | None = None, drops: tuple[int, ...] = (0,),
        by_drop: dict | None = None) -> tuple[dict[str, KeyVerdict], dict]:  # fmt: skip
    """The verdict of the stage. `seasons` and `drops` serve the check of the scorers on stale
    shots (4 November 2026): for each k of `drops`, the
    shots of the side's last k league matches are left out of its players' own rates, and the
    scored rows of each k go to `by_drop` (the verdict itself is of the first k)."""
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    research = tomllib.loads((ROOT / "config/vesta/research.toml").read_text(encoding="utf-8"))
    xg = tomllib.loads((ROOT / "config/vesta/expected_goals.toml").read_text(encoding="utf-8"))
    seasons = sorted(seasons or research["seasons"]["development" if stage == "development"
                                                    else "closed"])
    needed = 3 if stage == "development" else int(research["blind"]["min_seasons_improved"])
    repetitions = int(research["verdict"]["bootstrap_repetitions"])
    seed = int(research["verdict"]["seed"])
    alarm = research["alarm"]["players"]
    leagues = list(spine["walk_forward"]["competitions"])
    first = seasons[0] - TRAINING_SEASONS
    rated: dict = {}
    forecasts = {item.match_id: item for item in
                 forecast_world_with_champion(world_path, tuple(seasons), rated)}  # fmt: skip
    say(f"champion: {len(forecasts)} matches forecast")
    shot_section = xg["shots"]
    with open_world(world_path) as connection:
        matches = read_league_matches(connection, leagues, first - 1, seasons[-1])
        players = read_shot_players(connection, shot_section["competitions"],
                                    shot_section["first_season"], seasons[-1],
                                    shot_section["layers"])  # fmt: skip
        shots = rated["shots"]
        if len(players) != len(shots):
            raise RuntimeError(f"the shots' players ({len(players)}) do not line up with the"
                               f" shots rated ({len(shots)})")  # fmt: skip
        sums = sum_shots_by_player(shots.match_id, rated["chances"], players)
        conversion = read_penalty_conversion(connection, leagues, seasons[0])
        by_id = {match.match_id: match for match in matches}
        happened, own_goals = read_happened(connection, set(by_id))
        world = read_world(connection, first - 1, seasons[-1], TimeRules.from_config(spine))
    say(f"world read: {len(matches)} league matches, {len(happened)} players with a goal or"
        f" an assist")  # fmt: skip
    histories, groups_of = build_histories(world, by_id, sums, happened)
    known_at = {player: [item.known_at for item in items] for player, items in histories.items()}
    firsts = {
        match_id: find_first_scoring_side(by_id[match_id], happened, world) for match_id in by_id
    }
    team_known: dict[int, list[str]] = defaultdict(list)
    for played in world.team_matches.values():
        for item in played:
            if item.match_id in by_id:
                for team in (by_id[item.match_id].home_team_id, by_id[item.match_id].away_team_id):
                    team_known[team].append(item.known_at)
    team_known = {team: sorted(set(items)) for team, items in team_known.items()}
    assists_by_match: dict[int, int] = defaultdict(int)
    for (match_id, _), item in happened.items():
        assists_by_match[match_id] += item.assists
    scored_k = {k: defaultdict(list) for k in drops}
    lines_k = {k: defaultdict(lambda: ([], [])) for k in drops}
    scored, lines = scored_k[drops[0]], lines_k[drops[0]]
    coherence: list[float] = []
    unscored = 0
    by_season_rows = {}
    for season in range(first, seasons[-1] + 1):
        rows, teams = collect_targets(world, set(leagues), {season})
        by_season_rows[season] = (rows, teams)
        say(f"{season}: {len(rows)} players")
    for season in seasons:
        training = range(season - TRAINING_SEASONS, season)
        lineup_priors = Priors([row for year in training for row in by_season_rows[year][0]],
                               [team for year in training for team in by_season_rows[year][1]])
        before = [match for match in matches if match.season in training]
        by_league = estimate_league_rates(before, assists_by_match, own_goals, firsts, conversion)
        rates = group_rates([(groups_of[(player, index)], item)
                             for player, items in histories.items()
                             for index, item in enumerate(items)
                             if item.season in training])  # fmt: skip
        played_rows = 0
        grouped = defaultdict(list)
        for row in by_season_rows[season][0]:
            grouped[(row.match_id, row.team_id)].append(row)
        for (match_id, team_id), pool in grouped.items():
            forecast = forecasts.get(match_id)
            match = by_id.get(match_id)
            if forecast is None or match is None:
                continue
            home = team_id == match.home_team_id
            scored_rows, gap = _score_side(pool, home, forecast.matrix, lineup_priors, rates,
                                           by_league[match.competition], histories,
                                           known_at, happened, scored_k, lines_k,
                                           team_known)  # fmt: skip
            played_rows += scored_rows
            coherence.append(gap)
        sheet_played = sum(1 for match in matches if match.season == season
                           for side in (match.home_team_id, match.away_team_id)
                           for started, minutes, _ in world.sheets.get((match.match_id, side),
                                                                       {}).values()
                           if started or minutes > 0)  # fmt: skip
        unscored += sheet_played - played_rows
        say(f"{season}: {played_rows} players scored of {sheet_played} who played")
    results = {}
    for key, kind in KINDS.items():
        limit = float(alarm["max_gain_count" if kind == "count" else "max_gain_binary"])
        results[key] = _judge_key(key, kind, scored[key], lines[key], repetitions, seed, needed,
                                  limit)  # fmt: skip
    summary = {"coherence_max": max(coherence) if coherence else 0.0,
               "sides": len(coherence), "unscored": unscored}  # fmt: skip
    if scored_out is not None:
        scored_out.update(scored)
    if by_drop is not None:
        by_drop.update({k: (dict(scored_k[k]), dict(lines_k[k])) for k in drops})
    return results, summary


def drop_stale_appearances(
    known: list, team_known: list[str], forecast_time: str, drop: int
) -> list:
    """His appearances less those from his side's k-th last league match before the forecast
    onward: the outage of the scorers' stale-shots check."""
    if drop == 0:
        return known
    cut = bisect.bisect_right(team_known, forecast_time)
    if cut < drop:
        return []
    cutoff = team_known[cut - drop]
    return [item for item in known if item.known_at < cutoff]


def _score_side(pool, home: bool, matrix, lineup_priors: Priors, rates: dict[str | None,
                GroupRates], league: dict, histories, known_at, happened, scored_k, lines_k,
                team_known) -> tuple[int, float]:  # fmt: skip
    """Scores the side's players who played, for each k of `scored_k` (the shots of the side's
    last k league matches left out); returns how many, and the coherence gap of the first k (the
    pool's expected goals weighted by the chance to play against the side's, relative)."""
    first = None
    for drop in scored_k:
        count, gap, found = _predict_side(pool, home, matrix, lineup_priors, rates, league,
                                          histories, known_at, team_known, drop)  # fmt: skip
        _score_rows(pool, found, league, home, happened, scored_k[drop], lines_k[drop])
        first = first or (count, gap)
    return first


def _predict_side(pool, home, matrix, lineup_priors, rates, league, histories, known_at, team_known,
                  drop):  # fmt: skip
    design = np.column_stack([np.ones(len(pool)), [row.features for row in pool]])
    starts = scale_to_eleven(
        design @ lineup_priors.weights, np.array([row.goalkeeper for row in pool])
    )
    plays = np.zeros(len(pool))
    raw_goals = np.zeros(len(pool))
    raw_assists = np.zeros(len(pool))
    reference = []
    for index, (row, start) in enumerate(zip(pool, starts, strict=True)):
        comes_on, minutes = estimate_minutes(lineup_priors, row, float(start))
        plays[index] = float(start) + (1 - float(start)) * comes_on
        on = minutes[1:]
        expected_minutes = float(np.arange(1, len(minutes)) @ on / max(on.sum(), EPSILON))
        history = histories.get(row.player_id, [])
        cut = bisect.bisect_right(known_at.get(row.player_id, []), row.forecast_time)
        known = history[:cut]
        group = rates.get(row.group, rates[None])
        rated = drop_stale_appearances(
            known, team_known.get(row.team_id, []), row.forecast_time, drop
        )
        raw_goals[index], raw_assists[index] = weigh_player(rated, row.season, group,
                                                           league["conversion"], expected_minutes)
        reference.append(estimate_reference_rates(known, group))
    side = share_out_side_goals(raw_goals, raw_assists, plays, matrix, home, league)
    played = sum(1 for row in pool if row.started or row.minutes > 0)
    return played, side.gap, (side, reference)


def _score_rows(pool, found, league, home, happened, scored, lines) -> None:
    side, reference = found
    first_frequency = league["home_first" if home else "away_first"]
    count = 0
    for index, row in enumerate(pool):
        if not (row.started or row.minutes > 0):
            continue  # if he plays: void otherwise
        count += 1
        did = happened.get((row.match_id, row.player_id))
        goals, assists, first = (did.goals, did.assists, did.first) if did else (0, 0, False)
        block = assign_block(row.competition, row.forecast_time)
        mean, assist_mean = float(side.goals[index]), float(side.assists[index])
        own_first = float(side.first[index])
        goals_rate, assists_rate, share = reference[index]
        reference_first = min(share, 1.0) * first_frequency
        for key, own, against, hit in (
                ("P3.scores", 1 - math.exp(-mean), 1 - math.exp(-goals_rate), goals > 0),
                ("P3.first_scorer", own_first, reference_first, first),
                ("P5.assists", 1 - math.exp(-assist_mean), 1 - math.exp(-assists_rate),
                 assists > 0),
                ("P5.scores_or_assists", 1 - math.exp(-mean - assist_mean),
                 1 - math.exp(-goals_rate - assists_rate), goals > 0 or assists > 0)):  # fmt: skip
            scored[key].append(
                (
                    _score_log_loss(own if hit else 1 - own),
                    _score_log_loss(against if hit else 1 - against),
                    row.season,
                    block,
                )
            )
            lines[key][0].append((own, hit))
            lines[key][1].append((against, hit))
        own_cells, against_cells = (
            make_poisson_cells(mean, GOAL_CELLS),
            make_poisson_cells(goals_rate, GOAL_CELLS),
        )
        cell = min(goals, GOAL_CELLS - 1)
        scored["P3.goals"].append(
            (
                _score_log_loss(own_cells[cell]),
                _score_log_loss(against_cells[cell]),
                row.season,
                block,
            )
        )
        for at_least in (1, 2, 3):
            lines["P3.goals"][0].append((float(own_cells[at_least:].sum()), goals >= at_least))
            lines["P3.goals"][1].append((float(against_cells[at_least:].sum()),
                                         goals >= at_least))  # fmt: skip


def _judge_key(key, kind, rows, key_lines, repetitions, seed, needed, max_gain) -> KeyVerdict:
    differences = [own - naive for own, naive, _, _ in rows]
    mean, low, high = bootstrap_mean(differences, [row[3] for row in rows], repetitions, seed)
    by_season: dict[int, float] = defaultdict(float)
    for difference, row in zip(differences, rows, strict=True):
        by_season[row[2]] += difference
    improved = sum(1 for total in by_season.values() if total < 0)
    calibration_model = measure_calibration_error(key_lines[0])
    calibration_reference = measure_calibration_error(key_lines[1])
    alarm = -mean > max_gain
    passed = (high < 0 and improved >= needed and not alarm
              and calibration_model <= calibration_reference + CALIBRATION_TOLERANCE)  # fmt: skip
    return KeyVerdict(key, kind, len(rows), math.fsum(row[0] for row in rows) / len(rows),
                      math.fsum(row[1] for row in rows) / len(rows), mean, low, high, improved,
                      len(by_season), None, None, calibration_model, calibration_reference,
                      passed, alarm)  # fmt: skip


def report(results, summary, stage, world_path, world_sha, commit, seconds) -> str:
    lines = [f"# Scorers: the {stage} verdicts (candidate 1)", "",
             f"World: `{world_path}` (sha256 {world_sha}). Code: `{commit}`. Written by"
             f" `python -m vesta.players.verdict`; the rule is"
             f" the plan declared before the run (8cec816), the alarm `[alarm.players]`."
             f" {seconds:.0f} seconds.", "",
             "Log score of what happened, model against the reference (the player's own goals or"
             " assists per match over his last 20 matches played, shrunk to his group's; lower is"
             " better), on every player who played; the interval is the block bootstrap of the"
             " difference. P3.goals is scored on four cells (0, 1, 2, 3 or more), its calibration"
             " on the lines at least 1, 2 and 3.", "",
             "| key | n | model | reference | difference [95%] | seasons improved |"
             " calibration model / reference | verdict |",
             "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for item in results.values():
        verdict = "ALARM" if item.alarm else "pass" if item.passed else "fail"
        lines.append(f"| {item.key} | {item.forecasts} | {item.model:.4f} | {item.reference:.4f}"
                     f" | {item.difference:+.4f} [{item.low:+.4f}, {item.high:+.4f}] |"
                     f" {item.seasons_improved}/{item.seasons} | {item.calibration_model:.4f} /"
                     f" {item.calibration_reference:.4f} | {verdict} |")  # fmt: skip
    lines += ["", f"Coherence (a test, not a verdict): over {summary['sides']} sides, the largest"
              f" gap between the pool's expected goals weighted by the chance to play and the"
              f" side's own expected goals less the opponent's own goals is"
              f" {summary['coherence_max']:.2e} (the plan's bound: 1%).", "",
              f"Players who played but were outside the pool the lineups knew before the match"
              f" (not scored): {summary['unscored']}."]  # fmt: skip
    return "\n".join(lines) + "\n"


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("world", type=Path)
    parser.add_argument("--stage", choices=("development", "blind"), default="development")
    parser.add_argument("--out", type=Path, default=None)
    options = parser.parse_args(arguments)
    started = time.monotonic()
    results, summary = run(options.world, options.stage)
    digest = hashlib.sha256()
    with options.world.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 24), b""):
            digest.update(chunk)
    text = report(results, summary, options.stage, options.world, digest.hexdigest(),
                  run_git(ROOT, "rev-parse", "HEAD").strip(), time.monotonic() - started)
    out = options.out or ROOT / f"docs/research/2026-10-30-scorers-{options.stage}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
