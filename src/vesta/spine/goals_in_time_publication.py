"""What goals in time publish for a match, read from its population (the keys of step 3 in
`docs/FORECAST_LOG.md` §5.1): the first and last goal, the time of the first goal and goals in a
band, the halves, the halves of each team, the manner of the goals and the own goals. Every
distribution is a weighted count over the same simulated matches, so they agree with one another
and, where they depend only on the final score, with the matrix exactly.

An own goal counts for the side it benefits; the team that scores it (`M23.own_goal.team`) is the
other one. M14 (extra time, penalties, the tie) waits for the model of two-legged ties.
"""

import numpy as np

from vesta.forecast_log.writing import PublishedDistribution
from vesta.spine.goals_in_time import METHODS, Population
from vesta.spine.score_matrix import SIZE

FIRST_GOAL_BANDS = ((1, 10), (11, 20), (21, 30), (31, 45), (46, 60), (61, 75), (76, 90))
GOAL_BANDS = ((1, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90), (81, 90))
HALF_TIME_FULL_TIME = ("home", "draw", "away")
SIDES_OR_NONE = ("home", "away", "none")


def build_goals_in_time_distributions(
    population: Population, match_id: int, home_team_id: int, away_team_id: int, version: str
) -> list[PublishedDistribution]:
    """Every step-3 distribution of one match (M7, M8, M9-M12, M13, M23)."""
    weights = population.weight
    first_half = ((population.home_half == 1).sum(axis=1), (population.away_half == 1).sum(axis=1))
    full_time = (population.home_goals, population.away_goals)
    second_half = (full_time[0] - first_half[0], full_time[1] - first_half[1])
    first_side, first_minute, first_method = _find_first_goal(population)
    last_side = _find_last_goal_side(population)
    lowest, highest = _find_running_extremes(population)

    def make_item(key: str, subject: int, values: object) -> PublishedDistribution:
        return PublishedDistribution(key, subject, version, values)

    published = [
        make_item("M7.first_goal", match_id, _weigh_categories(weights, first_side, SIDES_OR_NONE)),
        make_item("M7.last_goal", match_id, _weigh_categories(weights, last_side, SIDES_OR_NONE)),
        make_item("M8.first_goal_time", match_id, _weigh_time_bands(weights, first_minute)),
        make_item("M9.result.first_half", match_id, _weigh_results(weights, *first_half)),
        make_item("M9.score.first_half", match_id, _weigh_joint_scores(weights, *first_half)),
        make_item(
            "M9.half_time_full_time",
            match_id,
            _weigh_half_time_full_time(weights, first_half, full_time),
        ),
        make_item("M10.result.second_half", match_id, _weigh_results(weights, *second_half)),
        make_item("M10.score.second_half", match_id, _weigh_joint_scores(weights, *second_half)),
        make_item(
            "M10.highest_scoring_half",
            match_id,
            _weigh_highest_half(weights, first_half, second_half),
        ),
        make_item("M12.goals.first_half", match_id, _count(weights, first_half[0] + first_half[1])),
        make_item(
            "M12.goals.second_half", match_id, _count(weights, second_half[0] + second_half[1])
        ),
        make_item(
            "M13.first_goal_method",
            match_id,
            _weigh_categories(weights, first_method, (*METHODS, "none")),
        ),
    ]
    for start, end in GOAL_BANDS:
        in_band = ((population.home_minute >= start) & (population.home_minute <= end)).any(axis=1)
        in_band |= ((population.away_minute >= start) & (population.away_minute <= end)).any(axis=1)
        published.append(make_item(f"M8.goal_in_band.band={start}-{end}", match_id,
                                   _weigh_binary(weights, in_band)))  # fmt: skip
    for line in (0.5, 1.5):
        over = ((first_half[0] + first_half[1] > line) & (second_half[0] + second_half[1] > line))
        published.append(make_item(f"M12.both_halves_over.line={line}", match_id,
                                   _weigh_binary(weights, over)))  # fmt: skip
    for index, method in enumerate(METHODS):
        scored = (population.home_method == index).any(axis=1)
        scored |= (population.away_method == index).any(axis=1)
        published.append(make_item(f"M13.goal_by_method.method={method}", match_id,
                                   _weigh_binary(weights, scored)))  # fmt: skip
    own_goal = METHODS.index("own_goal")
    # an own goal credited to the home side is scored by an away player, and the reverse
    away_own_goal = (population.home_method == own_goal).any(axis=1)
    home_own_goal = (population.away_method == own_goal).any(axis=1)
    any_own_goal = home_own_goal | away_own_goal
    published.append(make_item("M23.own_goal", match_id, _weigh_binary(weights, any_own_goal)))
    published.append(
        make_item("M23.own_goal.team", home_team_id, _weigh_binary(weights, home_own_goal))
    )
    published.append(
        make_item("M23.own_goal.team", away_team_id, _weigh_binary(weights, away_own_goal))
    )
    sides = (
        (home_team_id, 0, lowest < 0),  # the home side trailed when the running difference fell
        (away_team_id, 1, highest > 0),
    )
    for team, own, trailed in sides:
        other = 1 - own
        wins_first = first_half[own] > first_half[other]
        wins_second = second_half[own] > second_half[other]
        wins = full_time[own] > full_time[other]
        team_values = {
            "M11.wins_either_half.team": wins_first | wins_second,
            "M11.wins_both_halves.team": wins_first & wins_second,
            "M11.scores_in_both_halves.team": (first_half[own] > 0) & (second_half[own] > 0),
            "M11.comes_from_behind_to_win.team": trailed & wins,
            "M11.leads_at_half_time_and_wins.team": wins_first & wins,
        }
        published += [
            make_item(key, team, _weigh_binary(weights, value))
            for key, value in team_values.items()
        ]
    return published


def _find_first_goal(population: Population) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per simulated match: which side scored first (home, away or none), at which reported
    minute (0 when none), and its manner (`METHODS`, or none)."""
    home_time = np.where(np.isnan(population.home_time[:, 0]), np.inf, population.home_time[:, 0])
    away_time = np.where(np.isnan(population.away_time[:, 0]), np.inf, population.away_time[:, 0])
    home_first = home_time < away_time
    none = np.isinf(home_time) & np.isinf(away_time)
    side = np.where(none, "none", np.where(home_first, "home", "away"))
    minute = np.where(none, 0, np.where(home_first, population.home_minute[:, 0],
                                        population.away_minute[:, 0]))  # fmt: skip
    method_index = np.where(home_first, population.home_method[:, 0], population.away_method[:, 0])
    method = np.where(none, "none", np.array(METHODS, dtype=object)[np.maximum(method_index, 0)])
    return side, minute, method


def _find_last_goal_side(population: Population) -> np.ndarray:
    home_last = np.nanmax(np.where(np.isnan(population.home_time), -np.inf, population.home_time),
                          axis=1)  # fmt: skip
    away_last = np.nanmax(np.where(np.isnan(population.away_time), -np.inf, population.away_time),
                          axis=1)  # fmt: skip
    none = np.isinf(home_last) & np.isinf(away_last)
    return np.where(none, "none", np.where(home_last > away_last, "home", "away"))


def _find_running_extremes(population: Population) -> tuple[np.ndarray, np.ndarray]:
    """The lowest and highest home-minus-away score of each simulated match along its goals."""
    times = np.concatenate([population.home_time, population.away_time], axis=1)
    signs = np.concatenate([np.ones_like(population.home_time),
                            -np.ones_like(population.away_time)], axis=1)  # fmt: skip
    order = np.argsort(np.where(np.isnan(times), np.inf, times), axis=1, kind="stable")
    steps = np.where(np.isnan(np.take_along_axis(times, order, axis=1)), 0.0,
                     np.take_along_axis(signs, order, axis=1))  # fmt: skip
    running = np.cumsum(steps, axis=1)
    return np.minimum(running.min(axis=1), 0.0), np.maximum(running.max(axis=1), 0.0)


def _weigh_binary(weights: np.ndarray, happened: np.ndarray) -> dict:
    return {"yes": _clip_probability(float(np.sum(weights * happened)))}


def _weigh_categories(weights: np.ndarray, outcomes: np.ndarray, names: tuple[str, ...]) -> dict:
    return {name: float(np.sum(weights * (outcomes == name))) for name in names}


def _weigh_results(weights: np.ndarray, home: np.ndarray, away: np.ndarray) -> list:
    return [["home", float(np.sum(weights * (home > away)))],
            ["draw", float(np.sum(weights * (home == away)))],
            ["away", float(np.sum(weights * (home < away)))]]  # fmt: skip


def _weigh_joint_scores(weights: np.ndarray, home: np.ndarray, away: np.ndarray) -> dict:
    matrix = np.zeros((SIZE, SIZE))
    np.add.at(matrix, (np.minimum(home, SIZE - 1), np.minimum(away, SIZE - 1)), weights)
    return {"matrix": matrix.tolist(), "tail": 0.0}


def _count(weights: np.ndarray, counts: np.ndarray) -> dict:
    probabilities = np.bincount(counts, weights=weights, minlength=1)
    return {"probabilities": probabilities.tolist(), "tail_from": len(probabilities), "tail": 0.0}


def _weigh_time_bands(weights: np.ndarray, minute: np.ndarray) -> dict:
    bands = [[start, end, float(np.sum(weights * ((minute >= start) & (minute <= end))))]
             for start, end in FIRST_GOAL_BANDS]  # fmt: skip
    return {"bands": bands, "never": float(np.sum(weights * (minute == 0)))}


def _weigh_half_time_full_time(weights, first_half, full_time) -> dict:
    def classify_outcome(home, away):
        return np.where(home > away, "home", np.where(home == away, "draw", "away"))

    pair = np.char.add(np.char.add(classify_outcome(*first_half).astype(str), "/"),
                       classify_outcome(*full_time).astype(str))  # fmt: skip
    names = tuple(f"{half}/{full}" for half in HALF_TIME_FULL_TIME for full in HALF_TIME_FULL_TIME)
    return _weigh_categories(weights, pair, names)


def _weigh_highest_half(weights, first_half, second_half) -> dict:
    first, second = first_half[0] + first_half[1], second_half[0] + second_half[1]
    outcome = np.where(first > second, "first", np.where(first < second, "second", "equal"))
    return _weigh_categories(weights, outcome, ("first", "second", "equal"))


def _clip_probability(probability: float) -> float:
    """A weighted share can stray beyond [0, 1] by the last bit."""
    return min(max(probability, 0.0), 1.0)
