"""The verdict of the lineups, declared before the run in its plan
(3bf2e5d), its alarm in `[alarm.lineups]`.

    python -m vesta.lineups.verdict <football-world.db> [--stage development|blind] [--out <md>]

Every league match of the stage's seasons with a sheet is forecast walk-forward
(`vesta.lineups.model`, the weights and priors of a season from the five seasons before) and each
key is scored on its own against its reference, from the same five seasons: what the player or
team did in its last match. The deciding metric is the log score of what happened; a key passes
when its mean difference to the reference has a block bootstrap interval entirely below zero, it
improved in enough seasons, and its calibration error is no worse than the reference's by more
than the tolerance; the alarm stops a key whose gain is larger than its kind's limit.
"""

import argparse
import hashlib
import math
import sys
import time
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np

from vesta.determinism import run_on_one_thread
from vesta.lineups.model import (
    FEATURES,
    MINUTE_CELLS,
    OTHER,
    build_histogram,
    collect_targets,
    fit_logistic,
    read_world,
    scale_to_eleven,
    shrink_to_prior,
)
from vesta.research.bootstrap import assign_block, bootstrap_mean
from vesta.research.calibration import measure_calibration_error
from vesta.research.repository import run_git
from vesta.spine.timing import TimeRules
from vesta.spine.world import open_world
from vesta.team_counts.verdict import KeyVerdict

ROOT = Path(__file__).resolve().parents[3]
TRAINING_SEASONS = 5
CALIBRATION_TOLERANCE = 0.005
SMOOTHING = 0.5
BINS = ((0, 0), (1, 44), (45, 59), (60, 89), (90, 90))  # minutes, the last 90 or more
KINDS = {"L1.starts": "binary", "L6.plays": "binary", "L6.minutes": "categorical",
         "L2.formation.team": "categorical", "L9.position": "categorical"}  # fmt: skip
# the blind seasons are read only for the keys that passed in development (L6.plays did not)
KEYS_JUDGED = set(KINDS)
# candidate 2 (declared in the plan after eea9251): L6.plays' chance of coming on by its own
# logistic regression; only L6.plays is judged with it
CANDIDATE = [1]
STRENGTH = {"came_on": 3.0, "minutes": 3.0, "position": 1.0, "formation": 2.0}
EPSILON = 1e-12


def _find_minute_bin(minutes: int) -> int:
    return next(index for index, (low, high) in enumerate(BINS)
                if low <= min(minutes, 90) <= high)  # fmt: skip


def _score_log_loss(probability: float) -> float:
    return -math.log(max(probability, EPSILON))


class Priors:
    """What a season's forecasts lean on, from its five seasons before: the weights of the
    chance to start, the groups' priors, and the references."""

    def __init__(self, rows, teams):
        design = np.array([row.features for row in rows])
        self.weights = fit_logistic(design, np.array([row.started for row in rows], float))
        if CANDIDATE[0] == 2:
            benched = [row for row in rows if not row.started]
            self.on_weights = fit_logistic(
                np.array([_build_coming_on_features(row) for row in benched]),
                np.array([row.minutes > 0 for row in benched], float),
            )
        groups = defaultdict(lambda: [0.0, 0.0])
        starts, benches = defaultdict(lambda: np.zeros(MINUTE_CELLS)), defaultdict(
            lambda: np.zeros(MINUTE_CELLS))
        positions = defaultdict(lambda: defaultdict(float))
        state_start = defaultdict(lambda: np.full(2, SMOOTHING))
        state_play = defaultdict(lambda: np.full(2, SMOOTHING))
        state_bins = defaultdict(lambda: np.full(len(BINS), SMOOTHING))
        for row in rows:
            played = row.started or row.minutes > 0
            if not row.started:
                groups[row.group][0] += row.minutes > 0
                groups[row.group][1] += 1
                if row.minutes > 0:
                    benches[row.group][min(row.minutes, MINUTE_CELLS - 1)] += 1
            else:
                starts[row.group][min(row.minutes, MINUTE_CELLS - 1)] += 1
                if row.position:
                    positions[row.group][row.position] += 1
            key = (row.competition, row.state)
            state_start[key][int(row.started)] += 1
            state_play[key][int(played)] += 1
            state_bins[key][_find_minute_bin(row.minutes)] += 1
        self.came_on = {group: (on + 1.0) / (count + 2.0) for group, (on, count) in groups.items()}
        self.start_minutes = {group: (cells + 0.01) / (cells + 0.01).sum()
                              for group, cells in starts.items()}  # fmt: skip
        self.bench_minutes = {group: (cells + 0.01) / (cells + 0.01).sum()
                              for group, cells in benches.items()}  # fmt: skip
        codes = sorted({code for found in positions.values() for code in found})
        self.position_outcomes = codes + [OTHER]
        self.positions = {group: self._smooth_frequencies(found, self.position_outcomes)
                          for group, found in positions.items()}  # fmt: skip
        self.state_start = {key: cells / cells.sum() for key, cells in state_start.items()}
        self.state_play = {key: cells / cells.sum() for key, cells in state_play.items()}
        self.state_bins = {key: cells / cells.sum() for key, cells in state_bins.items()}
        formations = defaultdict(lambda: defaultdict(float))
        for team in teams:
            if team.formation:
                formations[team.competition][team.formation] += 1
        outcomes = sorted({item for found in formations.values() for item in found})
        self.formation_outcomes = outcomes + [OTHER]
        self.formations = {league: self._smooth_frequencies(found, self.formation_outcomes)
                           for league, found in formations.items()}  # fmt: skip
        self.formation_weight = self._fit_last_weight(
            [(team.last_formation, team.formation, self.formations.get(team.competition))
             for team in teams if team.formation], self.formation_outcomes)
        self.position_weight = self._fit_last_weight(
            [(row.last_position, row.position, self.positions.get(row.group))
             for row in rows if row.started and row.position], self.position_outcomes)

    @staticmethod
    def _smooth_frequencies(found: dict[str, float], outcomes: list[str]) -> dict[str, float]:
        values = np.array([found.get(item, 0.0) + SMOOTHING for item in outcomes])
        return dict(zip(outcomes, (values / values.sum()).tolist(), strict=True))

    @staticmethod
    def _fit_last_weight(cases, outcomes) -> float:
        """The weight on the last outcome that maximises the likelihood (a grid of 0.01)."""
        best, chosen = -math.inf, 0.0
        for weight in np.arange(0.0, 1.0, 0.01):
            total = 0.0
            for last, happened, frequencies in cases:
                truth = happened if happened in outcomes else OTHER
                probability = (1 - weight) * (frequencies or {}).get(truth, 1.0 / len(outcomes))
                if last is not None and (last if last in outcomes else OTHER) == truth:
                    probability += weight
                total += math.log(max(probability, EPSILON))
            if total > best:
                best, chosen = total, float(weight)
        return chosen


def _blend_last_outcome(last, frequencies, weight, outcomes) -> dict[str, float]:
    found = {item: (1 - weight) * (frequencies or {}).get(item, 1.0 / len(outcomes))
             for item in outcomes}  # fmt: skip
    if last is not None:
        found[last if last in outcomes else OTHER] += weight
    return found


def _shrink_categorical(own: dict[str, float], prior: dict[str, float] | None, strength: float,
                        outcomes: list[str]) -> dict[str, float]:  # fmt: skip
    values = np.array([sum(weight for item, weight in own.items()
                           if (item if item in outcomes else OTHER) == outcome)
                       + strength * (prior or {}).get(outcome, 1.0 / len(outcomes))
                       for outcome in outcomes])  # fmt: skip
    return dict(zip(outcomes, (values / values.sum()).tolist(), strict=True))


@run_on_one_thread
def run(world_path: Path, stage: str, say=print, lag: int = 0, scored_out: dict | None = None,
        candidate: int = 1) -> tuple[dict[str, KeyVerdict], dict]:  # fmt: skip
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    research = tomllib.loads((ROOT / "config/vesta/research.toml").read_text(encoding="utf-8"))
    seasons = sorted(research["seasons"]["development" if stage == "development" else "closed"])
    needed = 3 if stage == "development" else int(research["blind"]["min_seasons_improved"])
    repetitions = int(research["verdict"]["bootstrap_repetitions"])
    seed = int(research["verdict"]["seed"])
    alarm = research["alarm"]["lineups"]
    KEYS_JUDGED.clear()
    CANDIDATE[0] = candidate
    if candidate == 2:
        KEYS_JUDGED.add("L6.plays")
    else:
        KEYS_JUDGED.update(key for key in KINDS if stage == "development" or key != "L6.plays")
    leagues = set(spine["walk_forward"]["competitions"])
    first = seasons[0] - TRAINING_SEASONS
    with open_world(world_path) as connection:
        world = read_world(connection, first - 1, seasons[-1], TimeRules.from_config(spine))
    say(f"world read: {len(world.team_matches)} teams")
    by_season_rows, by_season_teams = {}, {}
    for season in range(first, seasons[-1] + 1):
        rows, teams = collect_targets(world, leagues, {season}, lag)
        by_season_rows[season], by_season_teams[season] = rows, teams
        say(f"{season}: {len(rows)} players, {len(teams)} teams")
    scored: dict[str, list] = defaultdict(list)
    lines: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    guessed = {"model": [], "last eleven": []}
    for season in seasons:
        training = range(season - TRAINING_SEASONS, season)
        priors = Priors([row for year in training for row in by_season_rows[year]],
                        [team for year in training for team in by_season_teams[year]])
        weights = np.round(priors.weights, 3).tolist()
        say(f"{season}: fitted {dict(zip(('intercept', *FEATURES), weights, strict=True))}")
        _score_season(priors, by_season_rows[season], by_season_teams[season], scored, lines,
                      guessed)  # fmt: skip
    results = {}
    for key, kind in KINDS.items():
        if key not in scored:
            continue
        limit = float(alarm["max_gain_binary" if kind == "binary" else "max_gain_categorical"])
        results[key] = _judge_key(key, kind, scored[key], lines[key], repetitions, seed, needed,
                                  limit)  # fmt: skip
    summary = {name: (float(np.mean(found)), len(found)) for name, found in guessed.items()}
    if scored_out is not None:
        scored_out.update(scored)
    return results, summary


def _score_season(priors: Priors, rows, teams, scored, lines, guessed) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.match_id, row.team_id)].append(row)
    for players in grouped.values():
        design = np.column_stack([np.ones(len(players)), [row.features for row in players]])
        chances = scale_to_eleven(
            design @ priors.weights, np.array([row.goalkeeper for row in players])
        )
        actual = {row.player_id for row in players if row.started}
        goalkeepers = [(c, row.player_id) for c, row in zip(chances, players, strict=True)
                       if row.goalkeeper]  # fmt: skip
        others = sorted(((c, row.player_id) for c, row in zip(chances, players, strict=True)
                         if not row.goalkeeper), reverse=True)  # fmt: skip
        picked = {player for _, player in sorted(goalkeepers, reverse=True)[:1] + others[:10]}
        guessed["model"].append(len(picked & actual))
        guessed["last eleven"].append(sum(1 for row in players
                                          if row.state == "started" and row.started))  # fmt: skip
        for chance, row in zip(chances, players, strict=True):
            _score_player(priors, row, float(chance), scored, lines)
    for team in teams:
        if not team.formation:
            continue
        outcomes = priors.formation_outcomes
        frequencies = priors.formations.get(team.competition)
        own = _shrink_categorical(
            dict(team.formations), frequencies, STRENGTH["formation"], outcomes
        )
        against = _blend_last_outcome(team.last_formation, frequencies,
                                         priors.formation_weight, outcomes)  # fmt: skip
        _score_categorical("L2.formation.team", team.formation, own, against, team.season,
                           assign_block(team.competition, team.forecast_time), scored, lines)


def _build_coming_on_features(row) -> tuple[float, ...]:
    """Candidate 2's features of coming on: the chance to start's, his share of bench
    appearances that ended on the pitch, and whether he has none."""
    came_on, not_started = row.came_on_rate
    return (*row.features, came_on / not_started if not_started else 0.0,
            float(not_started == 0))


def estimate_minutes(priors: Priors, row, start: float) -> tuple[float, np.ndarray]:
    """Candidate 1's chance to come on from the bench and the distribution of his minutes (cells
    0 to 89, then 90 or more), given his chance to start."""
    on = row.came_on_rate
    prior_on = priors.came_on.get(row.group, 0.3)
    comes_on = (on[0] + STRENGTH["came_on"] * prior_on) / (on[1] + STRENGTH["came_on"])
    flat = np.full(MINUTE_CELLS, 1.0 / MINUTE_CELLS)
    as_starter = shrink_to_prior(build_histogram(row.start_minutes, MINUTE_CELLS),
                                 priors.start_minutes.get(row.group, flat), STRENGTH["minutes"])
    as_substitute = shrink_to_prior(build_histogram(row.bench_minutes, MINUTE_CELLS),
                                    priors.bench_minutes.get(row.group, flat), STRENGTH["minutes"])
    minutes = start * as_starter + (1 - start) * comes_on * as_substitute
    minutes[0] += (1 - start) * (1 - comes_on)
    return comes_on, minutes


def _score_player(priors: Priors, row, start: float, scored, lines) -> None:
    block = assign_block(row.competition, row.forecast_time)
    comes_on, minutes = estimate_minutes(priors, row, start)
    if CANDIDATE[0] == 2:
        log_odds = priors.on_weights[0] + float(
            np.dot(priors.on_weights[1:], _build_coming_on_features(row))
        )
        plays = start + (1 - start) / (1.0 + math.exp(-log_odds))
    else:
        plays = start + (1 - start) * comes_on
    played = row.started or row.minutes > 0
    key = (row.competition, row.state)
    reference_start = priors.state_start.get(key, np.array([0.5, 0.5]))[1]
    reference_play = priors.state_play.get(key, np.array([0.5, 0.5]))[1]
    for name, own, against, hit in (("L1.starts", start, reference_start, row.started),
                                    ("L6.plays", plays, reference_play, played)):  # fmt: skip
        if name not in KEYS_JUDGED:
            continue
        scored[name].append(
            (
                _score_log_loss(own if hit else 1 - own),
                _score_log_loss(against if hit else 1 - against),
                row.season,
                block,
            )
        )
        lines[name][0].append((own, hit))
        lines[name][1].append((float(against), hit))
    bins = np.array([minutes[low:high + 1].sum() for low, high in BINS])
    against_bins = priors.state_bins.get(key, np.full(len(BINS), 1.0 / len(BINS)))
    truth = _find_minute_bin(row.minutes)
    if "L6.minutes" not in KEYS_JUDGED:
        return
    scored["L6.minutes"].append(
        (_score_log_loss(bins[truth]), _score_log_loss(against_bins[truth]), row.season, block)
    )
    for at_least in (1, 45, 60, 90):
        hit = row.minutes >= at_least
        lines["L6.minutes"][0].append((float(minutes[at_least:].sum()), hit))
        index = next(i for i, (low, _) in enumerate(BINS) if low >= at_least)
        lines["L6.minutes"][1].append((float(against_bins[index:].sum()), hit))
    if row.started and row.position:
        outcomes = priors.position_outcomes
        frequencies = priors.positions.get(row.group)
        own = _shrink_categorical(dict(row.positions), frequencies, STRENGTH["position"], outcomes)
        against = _blend_last_outcome(row.last_position, frequencies,
                                      priors.position_weight, outcomes)  # fmt: skip
        _score_categorical("L9.position", row.position, own, against, row.season, block,
                           scored, lines)


def _score_categorical(key, happened, own, against, season, block, scored, lines) -> None:
    if key not in KEYS_JUDGED:
        return
    truth = happened if happened in own else OTHER
    scored[key].append(
        (_score_log_loss(own[truth]), _score_log_loss(against[truth]), season, block)
    )
    for outcome in own:
        lines[key][0].append((own[outcome], outcome == truth))
        lines[key][1].append((against[outcome], outcome == truth))


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


def report(results, summary, stage, world_path, world_sha, commit, seconds,
           candidate: int = 1) -> str:  # fmt: skip
    lines = [f"# Lineups: the {stage} verdicts (candidate {candidate})", "",
             f"World: `{world_path}` (sha256 {world_sha}). Code: `{commit}`. Written by"
             f" `python -m vesta.lineups.verdict`; the rule is"
             f" the plan declared before the run (3bf2e5d), the alarm `[alarm.lineups]`."
             f" {seconds:.0f} seconds.", "",
             "Log score of what happened, model against the reference (the last match's state,"
             " from the five seasons before; lower is better); the interval is the block"
             " bootstrap of the difference. L6.minutes is scored on five bins (0, 1-44, 45-59,"
             " 60-89, 90+), its calibration on the lines at least 1, 45, 60 and 90.", "",
             "| key | n | model | reference | difference [95%] | seasons improved |"
             " calibration model / reference | verdict |",
             "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for item in results.values():
        verdict = "ALARM" if item.alarm else "pass" if item.passed else "fail"
        lines.append(f"| {item.key} | {item.forecasts} | {item.model:.4f} | {item.reference:.4f}"
                     f" | {item.difference:+.4f} [{item.low:+.4f}, {item.high:+.4f}] |"
                     f" {item.seasons_improved}/{item.seasons} | {item.calibration_model:.4f} /"
                     f" {item.calibration_reference:.4f} | {verdict} |")  # fmt: skip
    lines += ["", "Starters guessed of 11 (reported, not deciding): the likeliest goalkeeper and"
              " ten likeliest others against the last match's eleven, over the players Vesta"
              " knew before the match.", "", "| eleven | mean | team matches |", "|---|---|---|"]
    for name, (mean, count) in summary.items():
        lines.append(f"| {name} | {mean:.2f} | {count} |")
    return "\n".join(lines) + "\n"


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("world", type=Path)
    parser.add_argument("--stage", choices=("development", "blind"), default="development")
    parser.add_argument("--candidate", type=int, choices=(1, 2), default=1)
    parser.add_argument("--out", type=Path, default=None)
    options = parser.parse_args(arguments)
    started = time.monotonic()
    results, summary = run(options.world, options.stage, candidate=options.candidate)
    digest = hashlib.sha256()
    with options.world.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 24), b""):
            digest.update(chunk)
    text = report(results, summary, options.stage, options.world, digest.hexdigest(),
                  run_git(ROOT, "rev-parse", "HEAD").strip(), time.monotonic() - started,
                  options.candidate)
    suffix = "" if options.candidate == 1 else f"-candidate-{options.candidate}"
    out = options.out or ROOT / f"docs/research/2026-10-29-lineups-{options.stage}{suffix}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
