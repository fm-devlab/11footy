"""The verdict of family A's team counts, declared before the run in
its plan: shots, shots on target, fouls and offsides in the
five leagues and the cups, corners and cards in the cups.

    python -m vesta.team_counts.counts_verdict <football-world.db> --scope leagues|cups
        [--stage development|blind] [--counts shots,fouls,...] [--references all,2,3]
        [--out <report.md>]

Every match of the stage's seasons with both sides' counts is forecast walk-forward: the leagues
by `vesta.team_counts.model` (2.2.0's model), the cups by the pooled fit of
`vesta.team_counts.cups`; then the slopes and the tilt fitted on the forecasts of the two seasons
before the season judged. Each key is scored on its own against the competition's frequencies of
the seasons before (plan §5): a count key's smoothed frequencies; a joint key's, the product of the
two sides' smoothed frequencies. The rule is `config/vesta/research.toml` and `_verdict` of
`vesta.team_counts.verdict`.

`--references` judges the same forecasts against other references too: "all" the seasons before from
the first training season (the plan's), or a number, the competition's last that many seasons
(R2, R3); against a recent reference the calibration test is the absolute one of
`research.toml` `[verdict]` for frequency references.
"""

import argparse
import hashlib
import math
import sys
import time
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vesta.determinism import run_on_one_thread
from vesta.forecast_reading import read_line
from vesta.quantities import load_quantities
from vesta.research.bootstrap import assign_block
from vesta.research.calibration import bucket_lines, compute_chance_band, compute_joint_coverage
from vesta.research.repository import run_git
from vesta.research.scores import LOG_LOSS, RANKED_PROBABILITY, score
from vesta.spine.champion import forecast_world_with_champion
from vesta.spine.cups import CupsParameters, forecast_cups
from vesta.spine.goals_in_time_verdict import _has_line_happened
from vesta.spine.timing import TimeRules
from vesta.spine.world import open_world, read_club_matches
from vesta.team_counts import cups
from vesta.team_counts.model import (
    ALL_KEYS,
    FAMILY_A_KEYS,
    KEYS,
    Calibration,
    FitParameters,
    count_distributions,
    fit_calibration,
    fit_candidate_two,
    read_count_matches,
    seconds_of,
    walk_forward,
)
from vesta.team_counts.verdict import SMOOTHING, KeyVerdict, _judge_key, smooth_frequencies

ROOT = Path(__file__).resolve().parents[3]
SCOPES = ("leagues", "cups")
COUNTS = ("shots", "shots_on_target", "fouls", "offsides", "corners", "cards")
# a cup with fewer counted matches in the seasons before is judged against all the cups' (§5)
MINIMUM_REFERENCE_MATCHES = 200
# a group with fewer matches behind a season's corrections is called thin beside its fit
THIN_INPUTS = 150
# the shots and on target dominance (plan §7): above this share of violations both M20 keys fail
MAXIMUM_VIOLATIONS = 0.01


def _load_toml(name: str) -> dict:
    return tomllib.loads((ROOT / name).read_text(encoding="utf-8"))


def fold_outcomes(count: str, home: int, away: int, size: int) -> dict[tuple[str, str], object]:
    """What happened to each key of `count`, by (key, side), folded as the distributions are."""
    total_key, team_key, sides_key = ALL_KEYS[count]
    top = size - 1
    found = {(total_key, "match"): min(home + away, 2 * top),
             (team_key, "home"): min(home, top), (team_key, "away"): min(away, top)}  # fmt: skip
    if sides_key is not None:
        found[(sides_key, "match")] = (min(home, top), min(away, top))
    return found


def build_product_reference(seen: list[tuple[int, int]], size: int) -> dict:
    """The joint reference of plan §5: the outer product of the home side's and the away side's
    smoothed frequencies."""
    home = np.bincount([pair[0] for pair in seen], minlength=size) + SMOOTHING
    away = np.bincount([pair[1] for pair in seen], minlength=size) + SMOOTHING
    return {"matrix": np.outer(home / home.sum(), away / away.sum()).tolist(), "tail": 0.0}


def build_reference(count: str, seen: list[dict], size: int, definitions: dict,
                    product: bool) -> dict:  # fmt: skip
    """Every key's reference from the outcomes `seen`."""
    found = {}
    for key, side in seen[0]:
        kind = definitions[key].quantity.distribution_kind
        happened = [item[(key, side)] for item in seen]
        if kind == "joint_count" and product:
            found[(key, side)] = build_product_reference(happened, size)
        else:
            found[(key, side)] = smooth_frequencies(kind, happened, size, key == ALL_KEYS[count][0])
    return found


def uses_product(count: str, scope: str) -> bool:
    """The product reference for every key of this plan; the leagues' corners and cards keep
    2.2.0's smoothed joint table (they are not judged again here)."""
    return count in FAMILY_A_KEYS or scope == "cups"


@dataclass
class Judged:
    """The scored rows of one count and scope: per key, (own loss, reference loss, own ranked,
    reference ranked, season, block, qualifying); the lines' calibration pairs; each side's
    distribution by (match, side) for the coherence of shots and on target."""

    rows: dict[str, list]
    lines: dict[str, tuple[list, list]]
    sides: dict[tuple[int, str], np.ndarray]


class LeakageError(RuntimeError):
    """An input read for a match was known after the match's forecast time."""


def check_known_before(match, latest: int, what: str) -> None:
    """The run's assertion (the condition of 31 October): every input read for a match was
    known before its forecast time; `latest` is the latest instant (seconds) one was known at."""
    if latest >= seconds_of(match.forecast_time):
        raise LeakageError(f"match {match.match_id} ({match.competition} {match.season},"
                           f" forecast at {match.forecast_time}): {what} known at"
                           f" {latest}, not before the forecast")  # fmt: skip


def judge(count: str, scope: str, forecasts: list, calibrations: dict[int, Calibration],
          history: dict[tuple[str, int], list[tuple[int, dict]]], first_training: int,
          size: int, definitions: dict, calibration_known: dict[int, int] | None = None,
          window: int | None = None) -> Judged:  # fmt: skip
    """Score every forecast of a season in `calibrations` against its reference. A reference
    reads the outcomes of the seasons before known before its competition's season's first
    forecast, from `first_training`, or of the last `window` seasons only; `calibration_known`
    (absent in hindsight, in-sample by design) is the latest instant an input of each season's
    calibration was known at, asserted before every match."""
    product = uses_product(count, scope)
    starts: dict[tuple[str, int], int] = {}
    for match, _, _ in forecasts:
        key_of = (match.competition, match.season)
        starts[key_of] = min(starts.get(key_of, 2**62), seconds_of(match.forecast_time))
    rows: dict[str, list] = defaultdict(list)
    lines: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    sides: dict[tuple[int, str], np.ndarray] = {}
    references: dict[tuple[str, int], dict] = {}
    _, team_key, _ = ALL_KEYS[count]
    for match, means, size_nb in forecasts:
        if match.season not in calibrations:
            continue
        if calibration_known is not None:
            check_known_before(match, calibration_known[match.season], "the calibration's outcomes")
        calibration = calibrations[match.season]
        means = calibration.predict_means(means)
        key_of = (match.competition, match.season)
        if key_of not in references:
            first = (first_training if window is None
                     else max(first_training, match.season - window))  # fmt: skip
            earlier = [(code, item) for (code, season), seen in history.items()
                       if first <= season < match.season for item in seen
                       if item[0] < starts[key_of]]  # fmt: skip
            training = [item for code, item in earlier if code == match.competition]
            if scope == "cups" and len(training) < MINIMUM_REFERENCE_MATCHES:
                training = [item for _, item in earlier]
            references[key_of] = ((max(known for known, _ in training),
                                   build_reference(count, [happened for _, happened in training],
                                                   size, definitions, product))
                                  if training else (0, {}))  # fmt: skip
        latest, reference = references[key_of]
        check_known_before(match, latest, "the reference's outcomes")
        if not reference:
            continue
        happened = fold_outcomes(count, match.home_count, match.away_count, size)
        block = assign_block(match.competition, match.forecast_time)
        qualifying = scope == "cups" and match.qualifying
        for item in count_distributions(count, match.match_id, match.home_team_id,
                                        match.away_team_id, means, size_nb, size, "check",
                                        calibration.tilt):  # fmt: skip
            side = ("home" if item.quantity_key == team_key
                    and item.subject_id == match.home_team_id else
                    "away" if item.quantity_key == team_key else "match")  # fmt: skip
            if item.quantity_key == team_key:
                sides[(match.match_id, side)] = np.array(item.values["probabilities"])
            kind = definitions[item.quantity_key].quantity.distribution_kind
            outcome, against = happened[(item.quantity_key, side)], reference[
                (item.quantity_key, side)]
            own, naive = score(kind, item.values, outcome), score(kind, against, outcome)
            rows[item.quantity_key].append((own[LOG_LOSS], naive[LOG_LOSS],
                                            own.get(RANKED_PROBABILITY),
                                            naive.get(RANKED_PROBABILITY), match.season, block,
                                            qualifying))  # fmt: skip
            model_lines, reference_lines = lines[item.quantity_key]
            for line in definitions[item.quantity_key].quantity.lines:
                hit = _has_line_happened(kind, item.values, line, outcome)
                model_lines.append((read_line(kind, item.values, line), hit))
                reference_lines.append((read_line(kind, against, line), hit))
    return Judged(dict(rows), dict(lines), sides)


def read_scope(world, scope: str, count: str, spine: dict, first: int, last: int) -> list:
    if scope == "leagues":
        return read_count_matches(world, list(spine["walk_forward"]["competitions"]), first,
                                  last, count, TimeRules.from_config(spine))  # fmt: skip
    return cups.read_cup_count_matches(world, spine, first, last, count)


def build_history(matches: list, scope: str, count: str, size: int) -> dict:
    """Per (competition, season): each match's (instant its counts were known, outcomes)."""
    history: dict[tuple[str, int], list[tuple[int, dict]]] = defaultdict(list)
    for match in matches:
        if match.home_count is None or (scope == "cups" and not match.is_cup):
            continue
        history[(match.competition, match.season)].append(
            (seconds_of(match.known_at), fold_outcomes(count, match.home_count, match.away_count,
                                                       size)))  # fmt: skip
    return history


def forecast_scope(matches: list, scope: str, parameters: FitParameters, seasons: set[int],
                   audit=None) -> list:  # fmt: skip
    if scope == "leagues":
        return walk_forward(matches, parameters, seasons, audit)
    return cups.walk_forward(matches, parameters, seasons, audit)


def choose_size_group(scope: str, match) -> str:
    """Candidate 2's size is fitted per league, and once for the cups (plan §13)."""
    return match.competition if scope == "leagues" else "cups"


def fit_corrections(forecasts: list, seasons: set[int], size: int, candidate: int = 1,
                    scope: str = "leagues") -> tuple[dict, dict[int, int], dict]:  # fmt: skip
    """Each judged season's corrections, fitted on the forecasts of the two seasons before
    whose outcomes were known before the season's first forecast (a cup season can start before
    the one before it ends: CL 2019 ended in August 2020, after the 2020 qualifiers began); the
    latest instant one of them was known at; and candidate 2's sizes per season and group."""
    calibrations, latest, sizes = {}, {}, {}
    inputs_of: dict[int, dict[str, int]] = {}
    for season in sorted(seasons):
        start = min(seconds_of(item[0].forecast_time) for item in forecasts
                    if item[0].season == season)  # fmt: skip
        inputs = [item for item in forecasts if item[0].season in (season - 2, season - 1)
                  and seconds_of(item[0].known_at) < start]  # fmt: skip
        inputs_of[season] = dict(Counter(item[0].group if scope == "cups" else
                                         item[0].competition for item in inputs))  # fmt: skip
        if candidate == 2:
            calibrations[season], sizes[season] = fit_candidate_two(
                inputs, size, lambda match: choose_size_group(scope, match))
        else:
            calibrations[season] = fit_calibration(inputs, size)
        latest[season] = max(seconds_of(item[0].known_at) for item in inputs)
    return calibrations, latest, sizes, inputs_of


def _format_fitted_line(count: str, season: int, calibration: Calibration, sizes: dict | None,
                        inputs: dict[str, int]) -> str:
    """A report line: the corrections fitted for `season` and the matches they read."""
    def format_pair(values: tuple[float, float]) -> str:
        return f"({values[0]:+.3f}, {values[1]:.3f})"

    read = ", ".join(f"{name} {n}" for name, n in sorted(inputs.items()))
    thin = [name for name, n in inputs.items() if n < THIN_INPUTS]
    text = (
        f"| {count} | {season} | {format_pair(calibration.home)} |"
        f" {format_pair(calibration.away)} | {format_pair(calibration.total)} | "
    )
    text += (", ".join(f"{name} {value:.1f}" for name, value in sorted(sizes.items()))
             if sizes else "the weekly fits'")
    text += f" | {calibration.tilt:+.3f} | {sum(inputs.values())} ({read})"
    return text + (f" thin: {', '.join(sorted(thin))} |" if thin else " |")


def count_dominance_violations(shots: dict, on_target: dict) -> tuple[int, int]:
    """(violations, sides compared): a side's on-target law must lie below its shots' law, its
    mean below and `P(on target > k) <= P(shots > k)` for every k (plan §7)."""
    violations = compared = 0
    for key, target in on_target.items():
        if key not in shots:
            continue
        compared += 1
        attempts = shots[key]
        cells = min(len(target), len(attempts))
        mean_target = float(np.arange(len(target)) @ target)
        mean_shots = float(np.arange(len(attempts)) @ attempts)
        above_target = 1.0 - np.cumsum(target)[:cells - 1]
        above_shots = 1.0 - np.cumsum(attempts)[:cells - 1]
        if mean_target > mean_shots or np.any(above_target > above_shots + 1e-9):
            violations += 1
    return violations, compared


def correlate_with_spine(world_path: Path, scope: str, seasons: set[int], spine: dict,
                         sides: dict[str, dict]) -> dict[str, tuple[int, float]]:  # fmt: skip
    """Information (plan §7): per count, the correlation of a side's mean with the spine's
    expected goals of the side for the same matches (the champion in the leagues, the cups' model
    in the cups)."""
    if scope == "leagues":
        forecasts = forecast_world_with_champion(world_path, tuple(sorted(seasons)))
    else:
        parameters = CupsParameters.from_config(spine)
        with open_world(world_path) as world:
            matches = read_club_matches(world, list(parameters.leagues), list(parameters.cups),
                                        parameters.first_league_season,
                                        parameters.first_cup_season, max(seasons))  # fmt: skip
        forecasts = forecast_cups(matches, parameters, tuple(sorted(seasons)))
    goals = {}
    for forecast in forecasts:
        matrix = forecast.matrix
        goals[(forecast.match_id, "home")] = float(np.arange(matrix.shape[0]) @ matrix.sum(axis=1))
        goals[(forecast.match_id, "away")] = float(np.arange(matrix.shape[1]) @ matrix.sum(axis=0))
    found = {}
    for count, laws in sides.items():
        pairs = [(float(np.arange(len(law)) @ law), goals[key]) for key, law in laws.items()
                 if key in goals]  # fmt: skip
        if len(pairs) > 2:
            found[count] = (len(pairs), float(np.corrcoef(np.array(pairs).T)[0, 1]))
    return found


def look_up_alarm_limit(research: dict, scope: str, key: str, kind: str) -> float:
    """`[alarm.team_counts]` "<scope>.<key>" once the hindsight limits are approved
    (plan §8); else the kind's limit."""
    named = research["alarm"].get("team_counts", {}).get(f"{scope}.{key}")
    return float(named if named is not None else research["alarm"][kind]["max_gain"])


def look_up_warn_share(research: dict) -> float:
    """`warn_share` of `[alarm.team_counts]`: a key whose gain is above that share of its limit
    is flagged "warning" in the verdict, never stopped (the condition of 31 October)."""
    return float(research["alarm"].get("team_counts", {}).get("warn_share", 0.5))


def look_up_absolute_rule(research: dict) -> tuple[float, float]:
    """The calibration test against a frequency reference: (the most expected calibration error,
    the chance band's coverage), `research.toml` `[verdict]`."""
    verdict = research["verdict"]
    return (float(verdict["frequency_reference_max_ece"]),
            float(verdict["frequency_reference_band"]))  # fmt: skip


def count_buckets_outside(model_lines: list[tuple[float, bool]], band: float) -> tuple[int, int]:
    """(buckets whose happened share lies outside its chance band, buckets): the buckets together
    hold `band`, each at `joint_coverage(band, buckets)` (`research.toml` `[verdict]`)."""
    found = bucket_lines(model_lines)
    each = compute_joint_coverage(band, len(found)) if found else band
    outside = 0
    for bucket in found:
        low, high = compute_chance_band(bucket.said_mean, bucket.count, each)
        outside += not low <= bucket.happened_share <= high
    return outside, len(found)


def judge_absolutely(item: KeyVerdict, model_lines: list, needed: int,
                     rule: tuple[float, float]) -> tuple[KeyVerdict, tuple[int, int]]:  # fmt: skip
    """`item` with its calibration test replaced by the absolute one, and its buckets outside the
    band: the gain's rule unchanged, the model's error under the most and no bucket outside."""
    max_ece, band = rule
    outside = count_buckets_outside(model_lines, band)
    passed = (item.high < 0 and item.seasons_improved >= needed and not item.alarm
              and item.calibration_model < max_ece and outside[0] == 0)  # fmt: skip
    return KeyVerdict(**{**item.__dict__, "passed": passed}), outside


@dataclass(frozen=True)
class ScopeResult:
    verdicts: list[KeyVerdict]
    qualifying: dict[str, tuple[int, int, float]]  # key: (forecasts, blocks, mean difference)
    limits: dict[str, float]  # key: its alarm's limit
    seconds: float
    fitted: tuple[str, ...] = ()  # the corrections fitted per season judged, as report lines
    # against a recent reference, per key: (buckets outside the chance band, buckets)
    outside: dict[str, tuple[int, int]] = field(default_factory=dict)


@run_on_one_thread
def run(world_path: Path, scope: str, stage: str, counts: tuple[str, ...], candidate: int = 1,
        say=print, windows: tuple[int | None, ...] = (None,),
        ) -> tuple[dict, tuple[int, int] | None, dict]:  # fmt: skip
    """Per reference window (None: every season before), per count, its `ScopeResult`; the
    shots and on target coherence; the correlations with the spine."""
    spine, research = (
        _load_toml("config/vesta/spine.toml"),
        _load_toml("config/vesta/research.toml"),
    )
    config = _load_toml("config/vesta/team_counts.toml")
    seasons = set(research["seasons"]["development" if stage == "development" else "closed"])
    needed = (int(research["verdict"]["min_seasons_improved"]) if stage == "development"
              else int(research["blind"]["min_seasons_improved"]))  # fmt: skip
    repetitions = int(research["verdict"]["bootstrap_repetitions"])
    seed = int(research["verdict"]["seed"])
    first_training = int(spine["walk_forward"]["first_training_season"])
    definitions = load_quantities()
    rule = look_up_absolute_rule(research)
    results: dict[int | None, dict[str, ScopeResult]] = {window: {} for window in windows}
    sides = {}
    for count in counts:
        started = time.monotonic()
        parameters = FitParameters.from_config(config, count)
        size = parameters.size
        with open_world(world_path) as world:
            matches = read_scope(world, scope, count, spine, first_training - 2, max(seasons))
        say(f"{scope} {count}: {len(matches)} matches read,"
            f" {sum(m.home_count is not None for m in matches)} with counts")
        history = build_history(matches, scope, count, size)
        wanted = seasons | {season - back for season in seasons for back in (1, 2)}
        forecasts = forecast_scope(matches, scope, parameters, wanted,
                                   lambda match, latest: check_known_before(match, latest,
                                                                            "the fit's matches"))
        if scope == "cups":
            with open_world(world_path) as world:
                elo = cups.read_elo_inputs_known(world, spine, max(seasons))
            for match, _, _ in forecasts:
                check_known_before(match, elo[match.match_id], "the results of its Elo ratings")
        calibrations, calibration_known, sizes, inputs_of = fit_corrections(
            forecasts, seasons, size, candidate, scope)
        fitted = [_format_fitted_line(count, season, calibrations[season], sizes.get(season),
                                      inputs_of[season]) for season in sorted(seasons)]  # fmt: skip
        for season in sorted(seasons):
            say(f"{scope} {count} {season}: {calibrations[season]} {sizes.get(season, '')}")
        if candidate == 2:  # the sizes fitted on the seasons before replace the weekly fits'
            forecasts = [(match, means, sizes[match.season][choose_size_group(scope, match)])
                         if match.season in sizes else (match, means, size_nb)
                         for match, means, size_nb in forecasts]  # fmt: skip
        for window in windows:
            judged = judge(count, scope, forecasts, calibrations, history, first_training, size,
                           definitions, calibration_known, window)  # fmt: skip
            say(f"{scope} {count} (reference {window or 'all'}): every input known before its"
                f" forecast (asserted)")
            sides[count] = judged.sides  # the same forecasts whatever the reference
            verdicts, qualifying, limits, outside = [], {}, {}, {}
            for key in ALL_KEYS[count]:
                if key is None or key not in judged.rows:
                    continue
                kind = definitions[key].quantity.distribution_kind
                limits[key] = look_up_alarm_limit(research, scope, key, kind)
                item = _judge_key(key, kind, judged.rows[key], judged.lines[key], repetitions,
                                  seed, needed, limits[key])  # fmt: skip
                if window is not None:  # a recent reference: the absolute calibration test
                    item, outside[key] = judge_absolutely(item, judged.lines[key][0], needed,
                                                          rule)  # fmt: skip
                verdicts.append(item)
                subset = [row for row in judged.rows[key] if row[6]]
                if scope == "cups":
                    qualifying[key] = (len(subset), len({row[5] for row in subset}),
                                       math.fsum(row[0] - row[1] for row in subset) / len(subset)
                                       if subset else math.nan)  # fmt: skip
            results[window][count] = ScopeResult(verdicts, qualifying, limits,
                                                 time.monotonic() - started, tuple(fitted),
                                                 outside)  # fmt: skip
        say(f"{scope} {count}: judged")
    coherence = None
    if "shots" in sides and "shots_on_target" in sides:
        coherence = count_dominance_violations(sides["shots"], sides["shots_on_target"])
        if coherence[1] and coherence[0] / coherence[1] > MAXIMUM_VIOLATIONS:
            for found in results.values():
                for count in ("shots", "shots_on_target"):
                    found[count] = ScopeResult(
                        [KeyVerdict(**{**item.__dict__, "passed": False})
                         for item in found[count].verdicts],
                        found[count].qualifying, found[count].limits, found[count].seconds,
                        found[count].fitted, found[count].outside)  # fmt: skip
    attempts = {count: sides[count] for count in ("shots", "shots_on_target") if count in sides}
    correlations = (correlate_with_spine(world_path, scope, seasons, spine, attempts)
                    if attempts else {})  # fmt: skip
    return results, coherence, correlations


def report(results: dict[str, ScopeResult], coherence, correlations, share: float, scope: str,
           stage: str, world_path: Path, world_sha: str, commit: str, seconds: float,
           candidate: int = 1, window: int | None = None,
           rule: tuple[float, float] | None = None) -> str:  # fmt: skip
    lines = [f"# Team counts, family A: the {stage} verdicts in the {scope}"
             + (f" (candidate {candidate})" if candidate != 1 else "")
             + (f", against the last {window} seasons (R{window})" if window else ""), "",
             f"World: `{world_path}` (sha256 {world_sha}). Code: `{commit}`. Written by"
             f" `python -m vesta.team_counts.counts_verdict`; the rule is"
             f" the plan declared before the run. {seconds:.0f} seconds.", "",
             "Log score of what happened, model against the competition's frequencies of the"
             " seasons before (lower is better); the interval is the block bootstrap of the"
             " difference. The joint keys' reference is the product of the two sides' smoothed"
             " frequencies (plan §5). Gain / limit: the key's gain as a share of its alarm's"
             f" limit; above {share:.0%} the key is flagged (warning), never stopped."]  # fmt: skip
    if window:
        lines += ["", f"The reference reads the competition's last {window} seasons only (the cups"
                  f" pooled below {MINIMUM_REFERENCE_MATCHES} matches, as before). The calibration"
                  f" test is the absolute one for a frequency reference (`research.toml`"
                  f" `[verdict]`): the model's expected calibration error under {rule[0]:.3f} and"
                  f" every bucket inside its chance band, the bands holding {rule[1]:.0%} together"
                  f" (each at {rule[1]:.2f}^(1/k) over the key's k buckets); the reference's error"
                  " is shown for information. Buckets outside: of the model's tenths of probability"
                  " over the key's lines."]  # fmt: skip
    if scope == "cups" and any(count in KEYS for count in results):
        lines += ["", "The cups' M15 and M17 sides are judged against that product, not against"
                  " the smoothed joint table of the leagues' 2.2.0 verdict: the two scopes are not"
                  " judged alike on those keys."]  # fmt: skip
    lines += ["", "| key | n | model | reference | difference [95%] | seasons improved |"
              " ranked probability model / reference | calibration model / reference |"
              + (" buckets outside |" if window else "") + " alarm limit | gain / limit |"
              " verdict |", "|---|---|---|---|---|---|---|---|" + ("---|" if window else "")
              + "---|---|---|"]  # fmt: skip
    for result in results.values():
        for item in result.verdicts:
            ranked = ("-" if item.ranked_model is None else
                      f"{item.ranked_model:.4f} / {item.ranked_reference:.4f}")  # fmt: skip
            verdict = "ALARM" if item.alarm else "pass" if item.passed else "fail"
            limit = result.limits[item.key]
            if not item.alarm and -item.difference > share * limit:
                verdict += " (warning)"
            lines.append(f"| {item.key} | {item.forecasts} | {item.model:.4f} |"
                         f" {item.reference:.4f} | {item.difference:+.4f} [{item.low:+.4f},"
                         f" {item.high:+.4f}] | {item.seasons_improved}/{item.seasons} |"
                         f" {ranked} | {item.calibration_model:.4f} /"
                         f" {item.calibration_reference:.4f} |"
                         + (" {}/{} |".format(*result.outside[item.key]) if window else "")
                         + f" {limit:.3f} |"
                         f" {-item.difference / limit:.0%} | {verdict} |")  # fmt: skip
    if candidate == 2:
        lines += ["", "## What motivated candidate 2", "",
                  "The diagnosis of 31 October, made on candidate 1's development forecasts before"
                  " this run (plan §13): in the leagues the totals' means were right, their spread"
                  " too narrow (a match total's variance 27.9 said against 30.9 of the errors for"
                  " shots, 8.9 against 9.9 on target) and the totals overreacting (slope of what"
                  " happened on what was said 0.83 and 0.86); in the cups the weekly fit's size"
                  " was near Poisson (about 50 for shots, 35 for corners) against a spread about"
                  " 2.5 times the mean, and the totals overreacting strongly (slope 0.28 to 0.78"
                  " by group, 0.30 to 0.57 by season for shots, 0.27 to 0.49 for corners); the"
                  " levels within about 0.5. The numbers below are candidate 2's own fits, out of"
                  " sample, not the diagnosis's."]  # fmt: skip
    if any(result.fitted for result in results.values()):
        lines += ["", "## The corrections fitted per season judged", "",
                  "On the forecasts of the two seasons before, cut at the season's first forecast:"
                  " each side's (intercept, slope) of the log mean, the total's (a, b), the"
                  " negative binomial's size per league (or the cups'), the tilt, and the matches"
                  f" read per league or cup group (thin: fewer than {THIN_INPUTS}).", "",
                  "| count | season | home | away | total | size | tilt | matches read |",
                  "|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for result in results.values():
            lines += list(result.fitted)
    if coherence is not None:
        violations, compared = coherence
        lines += ["", f"Shots and on target (plan §7): {violations} of {compared} sides break"
                  f" the dominance ({violations / max(compared, 1):.2%}; the limit is"
                  f" {MAXIMUM_VIOLATIONS:.0%})."]  # fmt: skip
    for count, (size, correlation) in correlations.items():
        lines.append(f"With the spine (information, plan §7): {count}, the correlation of a side's"
                     f" mean with the spine's expected goals of the side, {correlation:+.3f} on"
                     f" {size} sides.")
    if scope == "cups":
        lines += ["", "The qualifiers (information, plan §6 and §11): forecasts, bootstrap blocks"
                  " (competition and month) and the mean difference to the reference.", "",
                  "| key | forecasts | blocks | difference |", "|---|---|---|---|"]  # fmt: skip
        for result in results.values():
            for key, (size, blocks, difference) in result.qualifying.items():
                lines.append(f"| {key} | {size} | {blocks} | {difference:+.4f} |")
    return "\n".join(lines) + "\n"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("world", type=Path)
    parser.add_argument("--scope", choices=SCOPES, required=True)
    parser.add_argument("--stage", choices=("development", "blind"), default="development")
    parser.add_argument("--counts", default=None, help="comma-separated; default: the scope's")
    parser.add_argument("--candidate", type=int, choices=(1, 2), default=1)
    parser.add_argument("--references", default="all",
                        help='comma-separated: "all" the seasons before, or a number of seasons')
    parser.add_argument("--out", type=Path, default=None)
    options = parser.parse_args(arguments)
    counts = (tuple(options.counts.split(",")) if options.counts else
              tuple(FAMILY_A_KEYS) if options.scope == "leagues" else COUNTS)  # fmt: skip
    windows = tuple(None if item == "all" else int(item)
                    for item in options.references.split(","))  # fmt: skip
    started = time.monotonic()
    results, coherence, correlations = run(options.world, options.scope, options.stage, counts,
                                           options.candidate, windows=windows)  # fmt: skip
    research = _load_toml("config/vesta/research.toml")
    world_sha, commit = hash_file(options.world), run_git(ROOT, "rev-parse", "HEAD").strip()
    text = "\n".join(report(results[window], coherence, correlations, look_up_warn_share(research),
                            options.scope, options.stage, options.world, world_sha, commit,
                            time.monotonic() - started, options.candidate, window,
                            look_up_absolute_rule(research))
                     for window in windows)  # fmt: skip
    suffix = f"-candidate-{options.candidate}" if options.candidate != 1 else ""
    out = options.out or ROOT / (f"docs/research/2026-10-31-team-counts-{options.scope}-"
                                 f"{options.stage}{suffix}.md")  # fmt: skip
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
