"""The verdict of goals in time (step 3): every quantity scored against what happened.

For each match of the development seasons, the population of goals in time is drawn from the
champion's matrix of that match, with the first-half shares and the shares of the manners read
from the training seasons only (the seasons before the one forecast, from the first training
season). Each step-3 quantity is scored against what the match's story says happened, with the
deciding metric of its kind (`vesta.research.scores`), beside a reference: the frequencies of the
same quantity in the training seasons (each outcome counted half a time more, so none is zero).
The difference, model minus reference, gets its interval by the block bootstrap of the
discipline; the calibration of the declared lines is measured for both.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from vesta.forecast_reading import read_line
from vesta.quantities import QuantityDefinition
from vesta.research.bootstrap import assign_block, bootstrap_mean
from vesta.research.calibration import measure_calibration_error
from vesta.research.scores import METRICS_BY_KIND, OutcomeError, score
from vesta.spine.goals_in_time import (
    METHODS,
    GoalsInTimeParameters,
    TimeShares,
    simulate,
)
from vesta.spine.goals_in_time_publication import (
    FIRST_GOAL_BANDS,
    GOAL_BANDS,
    HALF_TIME_FULL_TIME,
    SIDES_OR_NONE,
    build_goals_in_time_distributions,
)
from vesta.spine.match_forecast import MatchForecast
from vesta.spine.world import MatchStory

SMOOTHING = 0.5  # each outcome of the reference counted half a time more
LARGEST_COUNT = 20  # the reference's counts run at least this far, so no outcome falls outside
# the names of the outcomes of each categorical or ordered quantity, in their order
OUTCOME_NAMES = {
    "M7.first_goal": SIDES_OR_NONE,
    "M7.last_goal": SIDES_OR_NONE,
    "M9.result.first_half": ("home", "draw", "away"),
    "M9.half_time_full_time": tuple(f"{half}/{full}" for half in HALF_TIME_FULL_TIME
                                    for full in HALF_TIME_FULL_TIME),
    "M10.result.second_half": ("home", "draw", "away"),
    "M10.highest_scoring_half": ("first", "second", "equal"),
    "M13.first_goal_method": (*METHODS, "none"),
}  # fmt: skip


def find_outcomes(story: MatchStory) -> dict[tuple[str, str], object]:
    """What happened, per (quantity key, subject: match, home or away), in the shape the scores
    of its kind read."""
    (home_half, away_half), (home_full, away_full) = story.half_time, story.full_time
    second = (home_full - home_half, away_full - away_half)
    goals = story.goals
    first, last = (goals[0], goals[-1]) if goals else (None, None)
    happened: dict[tuple[str, str], object] = {
        ("M7.first_goal", "match"): first[1] if first else "none",
        ("M7.last_goal", "match"): last[1] if last else "none",
        ("M8.first_goal_time", "match"): first[0] if first else None,
        ("M9.result.first_half", "match"): _classify_result(home_half, away_half),
        ("M9.score.first_half", "match"): (home_half, away_half),
        (
            "M9.half_time_full_time",
            "match",
        ): f"{_classify_result(home_half, away_half)}/{_classify_result(home_full, away_full)}",
        ("M10.result.second_half", "match"): _classify_result(*second),
        ("M10.score.second_half", "match"): second,
        ("M10.highest_scoring_half", "match"): _classify_highest_half(
            home_half + away_half, sum(second)
        ),
        ("M12.goals.first_half", "match"): home_half + away_half,
        ("M12.goals.second_half", "match"): second[0] + second[1],
    }
    for start, end in GOAL_BANDS:
        happened[(f"M8.goal_in_band.band={start}-{end}", "match")] = any(
            start <= minute <= end for minute, _, _ in goals
        )
    for line in (0.5, 1.5):
        happened[(f"M12.both_halves_over.line={line}", "match")] = (
            home_half + away_half > line and second[0] + second[1] > line
        )
    methods = [method for _, _, method in goals]
    if None not in methods:
        happened[("M13.first_goal_method", "match")] = first[2] if first else "none"
        for method in METHODS:
            happened[(f"M13.goal_by_method.method={method}", "match")] = method in methods
    own_goals_for = {side for _, side, method in goals if method == "own_goal"}
    happened[("M23.own_goal", "match")] = bool(own_goals_for)
    # an own goal counted for one side is scored by a player of the other
    happened[("M23.own_goal.team", "away")] = "home" in own_goals_for
    happened[("M23.own_goal.team", "home")] = "away" in own_goals_for
    lowest, highest = _find_running_extremes(goals)
    for side, own, other, trailed in (("home", 0, 1, lowest < 0), ("away", 1, 0, highest > 0)):
        halves = ((home_half, away_half), second)
        wins_first = halves[0][own] > halves[0][other]
        wins_second = halves[1][own] > halves[1][other]
        wins = story.full_time[own] > story.full_time[other]
        happened[("M11.wins_either_half.team", side)] = wins_first or wins_second
        happened[("M11.wins_both_halves.team", side)] = wins_first and wins_second
        happened[("M11.scores_in_both_halves.team", side)] = (halves[0][own] > 0
                                                              and halves[1][own] > 0)  # fmt: skip
        happened[("M11.comes_from_behind_to_win.team", side)] = trailed and wins
        happened[("M11.leads_at_half_time_and_wins.team", side)] = wins_first and wins
    return happened


@dataclass(frozen=True)
class KeyResult:
    key: str
    kind: str
    metric: str
    forecasts: int
    model: float
    reference: float
    difference: float
    interval_low: float
    interval_high: float
    seasons_improved: int
    seasons: int
    calibration_error_model: float
    calibration_error_reference: float


def check_goals_in_time(
    forecasts: list[MatchForecast],
    stories: dict[int, MatchStory],
    season_of: dict[int, int],
    shares_by_season: dict[int, TimeShares],
    definitions: dict[str, QuantityDefinition],
    parameters: GoalsInTimeParameters,
    repetitions: int,
    seed: int,
    first_training_season: int,
    added_by_season: dict[int, tuple[float, float]] | None = None,
) -> list[KeyResult]:
    """The results of every step-3 quantity on the forecast matches that have a story; the
    references are read from the stories of the training seasons (`season_of` gives the season of
    every match with a story, the forecast ones and the training ones); `added_by_season` the added
time of each season read from the live chain's profile (`vesta.live.timing`), else the
parameters' own."""
    references = _build_references(stories, season_of, {forecast.season for forecast in forecasts},
                                   definitions, first_training_season)  # fmt: skip
    checked = [forecast for forecast in forecasts
               if forecast.match_id in stories and references.get(forecast.season)]  # fmt: skip
    pairs: dict[str, list[tuple[float, float, int, str]]] = {}
    lines: dict[str, list[tuple[list[tuple[float, bool]], list[tuple[float, bool]]]]] = {}
    for forecast in checked:
        story = stories[forecast.match_id]
        used = parameters
        if added_by_season is not None:
            used = replace(parameters, added_time_minutes=added_by_season[forecast.season])
        population = simulate(forecast.matrix, shares_by_season[forecast.season], used,
                              forecast.match_id)  # fmt: skip
        published = build_goals_in_time_distributions(
            population, story.match_id, story.home_team_id, story.away_team_id, "check"
        )
        happened = find_outcomes(story)
        block = assign_block(forecast.competition, forecast.forecast_time)
        for item in published:
            quantity = definitions[item.quantity_key].quantity
            side = _find_side(quantity.subject_kind, item.subject_id, story)
            if (item.quantity_key, side) not in happened:
                continue
            kind = quantity.distribution_kind
            reference = references[forecast.season][(item.quantity_key, side)]
            outcome = happened[(item.quantity_key, side)]
            try:
                model_score = score(kind, item.values, outcome)
                reference_score = score(kind, reference, outcome)
            except OutcomeError:
                continue
            metric = METRICS_BY_KIND[kind][0]
            pairs.setdefault(item.quantity_key, []).append(
                (model_score[metric], reference_score[metric], forecast.season, block)
            )
            model_lines, reference_lines = lines.setdefault(item.quantity_key, ([], []))
            for line in definitions[item.quantity_key].quantity.lines:
                hit = _has_line_happened(kind, item.values, line, outcome)
                model_lines.append((read_line(kind, item.values, line), hit))
                reference_lines.append((read_line(kind, reference, line), hit))
    return [
        _build_key_result(key, definitions[key].quantity.distribution_kind, rows, lines[key],
                          repetitions, seed)  # fmt: skip
        for key, rows in sorted(pairs.items())
    ]


def _build_key_result(key, kind, rows, key_lines, repetitions, seed) -> KeyResult:
    differences = [model - reference for model, reference, _, _ in rows]
    mean, low, high = bootstrap_mean(differences, [row[3] for row in rows], repetitions, seed)
    by_season: dict[int, list[float]] = {}
    for difference, (_, _, season, _) in zip(differences, rows, strict=True):
        by_season.setdefault(season, []).append(difference)
    improved = sum(1 for values in by_season.values() if math.fsum(values) < 0)
    return KeyResult(
        key=key,
        kind=kind,
        metric=METRICS_BY_KIND[kind][0],
        forecasts=len(rows),
        model=math.fsum(row[0] for row in rows) / len(rows),
        reference=math.fsum(row[1] for row in rows) / len(rows),
        difference=mean,
        interval_low=low,
        interval_high=high,
        seasons_improved=improved,
        seasons=len(by_season),
        calibration_error_model=measure_calibration_error(key_lines[0]),
        calibration_error_reference=measure_calibration_error(key_lines[1]),
    )


def _build_references(stories, season_of, forecast_seasons, definitions, first_training_season):
    """Per season forecast, the frequency distribution of every quantity in the training seasons:
    the matches of the seasons before it, from the first training season."""
    happened_by_season: dict[int, list[dict]] = {}
    for match_id, story in stories.items():
        if match_id in season_of:
            happened_by_season.setdefault(season_of[match_id], []).append(find_outcomes(story))
    references = {}
    for season in sorted(forecast_seasons):
        training = [happened for earlier, rows in happened_by_season.items()
                    if first_training_season <= earlier < season for happened in rows]  # fmt: skip
        keys = {key for happened in training for key in happened}
        references[season] = {
            key: _smooth_frequencies(key[0], definitions[key[0]].quantity.distribution_kind,
                                     [happened[key] for happened in training if key in happened])
            for key in keys
        }  # fmt: skip
    return references


def _smooth_frequencies(key: str, kind: str, outcomes_seen: list[object]) -> object:
    """The smoothed frequencies of the outcomes, in the shape of the kind."""
    if kind == "binary":
        return {"yes": (sum(bool(seen) for seen in outcomes_seen) + SMOOTHING)
                / (len(outcomes_seen) + 2 * SMOOTHING)}  # fmt: skip
    if kind in ("categorical", "ordered"):
        names = OUTCOME_NAMES[key]
        counts = {name: sum(seen == name for seen in outcomes_seen) + SMOOTHING for name in names}
        total = sum(counts.values())
        if kind == "ordered":
            return [[name, counts[name] / total] for name in names]
        return {name: counts[name] / total for name in names}
    if kind == "count":
        size = max(max(outcomes_seen) + 2, LARGEST_COUNT + 1)
        counts = np.bincount(outcomes_seen, minlength=size) + SMOOTHING
        return {"probabilities": (counts / counts.sum()).tolist(), "tail_from": size, "tail": 0.0}
    if kind == "joint_count":
        size = 11
        counts = np.full((size, size), SMOOTHING)
        for home, away in outcomes_seen:
            counts[min(home, size - 1), min(away, size - 1)] += 1
        return {"matrix": (counts / counts.sum()).tolist(), "tail": 0.0}
    if kind == "time_to_event":
        counts = [sum(1 for seen in outcomes_seen if seen is not None and start <= seen <= end)
                  + SMOOTHING for start, end in FIRST_GOAL_BANDS]  # fmt: skip
        never = sum(1 for seen in outcomes_seen if seen is None) + SMOOTHING
        total = sum(counts) + never
        return {"bands": [[start, end, count / total]
                          for (start, end), count in zip(FIRST_GOAL_BANDS, counts, strict=True)],
                "never": never / total}  # fmt: skip
    raise ValueError(f"no reference for the kind {kind!r}")


def _has_line_happened(kind: str, values: object, line: str, outcome: object) -> bool:
    """Whether a line happened: the line read on the distribution that puts everything on what
    happened."""
    return read_line(kind, _make_point_mass(kind, values, outcome), line) > 0.5


def _make_point_mass(kind: str, values: object, outcome: object) -> object:
    if kind == "binary":
        return {"yes": 1.0 if outcome else 0.0}
    if kind == "categorical":
        return {name: 1.0 if name == outcome else 0.0 for name in values}
    if kind == "ordered":
        return [[name, 1.0 if name == outcome else 0.0] for name, _ in values]
    if kind == "count":
        size = max(len(values["probabilities"]), outcome + 1)
        probabilities = [0.0] * size
        probabilities[outcome] = 1.0
        return {"probabilities": probabilities, "tail_from": size, "tail": 0.0}
    if kind == "joint_count":
        size = len(values["matrix"])
        matrix = [[0.0] * size for _ in range(size)]
        matrix[min(outcome[0], size - 1)][min(outcome[1], size - 1)] = 1.0
        return {"matrix": matrix, "tail": 0.0}
    if kind == "time_to_event":
        bands = [[start, end, 1.0 if outcome is not None and start <= outcome <= end else 0.0]
                 for start, end, _ in values["bands"]]  # fmt: skip
        return {"bands": bands, "never": 1.0 if outcome is None else 0.0}
    raise ValueError(f"no point mass for the kind {kind!r}")


def _find_side(subject_kind: str, subject_id: int, story: MatchStory) -> str:
    if subject_kind != "team":
        return "match"
    return "home" if subject_id == story.home_team_id else "away"


def _classify_result(home: int, away: int) -> str:
    return "home" if home > away else "draw" if home == away else "away"


def _classify_highest_half(first: int, second: int) -> str:
    return "first" if first > second else "second" if second > first else "equal"


def _find_running_extremes(goals) -> tuple[int, int]:
    difference, lowest, highest = 0, 0, 0
    for _, side, _ in goals:
        difference += 1 if side == "home" else -1
        lowest, highest = min(lowest, difference), max(highest, difference)
    return lowest, highest
