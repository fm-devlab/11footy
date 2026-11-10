"""The score of one forecast against what happened, by kind of distribution.

One table of metrics serves the public check (`docs/FORECAST_LOG.md` §9) and the verdicts of the
research discipline, so a published version and a candidate are judged by the same code. Every
metric is a proper score and lower is better.

What happened has the shape of its kind: a bool (binary), an outcome name (categorical,
ordered), a count (count), a pair of counts (joint_count), the minute of the first event or
None when it never happened (time_to_event), a place from 1 (rank).
"""

import math

from vesta.distributions import (
    BINARY,
    CATEGORICAL,
    COUNT,
    JOINT_COUNT,
    ORDERED,
    RANK,
    TIME_TO_EVENT,
)

# The probability inside a logarithm is kept at least this large: a forecast that gave exactly 0
# to what happened scores a large finite loss (about 34.5), so one such case is visible in a mean
# without making it infinite.
LOG_FLOOR = 1e-15

LOG_LOSS = "log_loss"
BRIER = "brier"
RANKED_PROBABILITY = "ranked_probability_score"

# The metric that decides a verdict for each kind (the first) and the others reported beside it.
METRICS_BY_KIND = {
    BINARY: (LOG_LOSS, BRIER),
    CATEGORICAL: (LOG_LOSS, BRIER),
    ORDERED: (RANKED_PROBABILITY, LOG_LOSS, BRIER),
    COUNT: (RANKED_PROBABILITY, LOG_LOSS),
    JOINT_COUNT: (LOG_LOSS,),
    TIME_TO_EVENT: (LOG_LOSS,),
    RANK: (RANKED_PROBABILITY, LOG_LOSS),
}


class OutcomeError(ValueError):
    """An outcome that the distribution's kind cannot be scored against."""


def score(kind: str, values: object, happened: object) -> dict[str, float]:
    """Every metric of `METRICS_BY_KIND[kind]` for one forecast, by metric name."""
    categories, realised = _split_categories(kind, values, happened)
    result = {LOG_LOSS: -math.log(max(categories[realised], LOG_FLOOR))}
    metrics = METRICS_BY_KIND[kind]
    if BRIER in metrics:
        result[BRIER] = math.fsum(
            (p - (1.0 if index == realised else 0.0)) ** 2 for index, p in enumerate(categories)
        )
    if RANKED_PROBABILITY in metrics:
        result[RANKED_PROBABILITY] = _score_ranked_probability(categories, realised, kind)
    return result




def _split_categories(kind: str, values: object, happened: object) -> tuple[list[float], int]:
    """The distribution as a list of exclusive categories in their order, and the index of the
    one that happened."""
    if kind == BINARY:
        if not isinstance(happened, bool):
            raise OutcomeError(f"a binary outcome is true or false, got {happened!r}")
        return [values["yes"], 1.0 - values["yes"]], 0 if happened else 1
    if kind in (CATEGORICAL, ORDERED):
        pairs = list(values.items()) if kind == CATEGORICAL else [tuple(p) for p in values]
        names = [name for name, _ in pairs]
        if happened not in names:
            raise OutcomeError(f"{happened!r} is not an outcome of {names}")
        return [p for _, p in pairs], names.index(happened)
    if kind == COUNT:
        count = _require_whole(happened)
        categories = [*values["probabilities"], values["tail"]]
        return categories, min(count, values["tail_from"])
    if kind == JOINT_COUNT:
        home, away = (_require_whole(goals) for goals in happened)
        matrix = values["matrix"]
        categories = [p for row in matrix for p in row] + [values["tail"]]
        inside = home < len(matrix) and away < len(matrix[0])
        return categories, home * len(matrix[0]) + away if inside else len(categories) - 1
    if kind == TIME_TO_EVENT:
        bands = values["bands"]
        categories = [band[2] for band in bands] + [values["never"]]
        if happened is None:
            return categories, len(bands)
        minute = _require_whole(happened)
        for index, (start, end, _) in enumerate(bands):
            if start <= minute <= end:
                return categories, index
        raise OutcomeError(f"minute {minute} lies in no published band")
    if kind == RANK:
        place = _require_whole(happened)
        if not 1 <= place <= len(values):
            raise OutcomeError(f"place {place} is outside 1 to {len(values)}")
        return list(values), place - 1
    raise OutcomeError(f"no score for the distribution kind {kind!r}")


def _score_ranked_probability(categories: list[float], realised: int, kind: str) -> float:
    """Sum of the squared gaps between the forecast's and the outcome's cumulative probabilities
    over the ordered categories. Divided by the number of gaps for ordered outcomes and places
    (between 0 and 1, the usual form of the 1X2 score); left in counts for a count,
    where it is the discrete continuous ranked probability score."""
    cumulative, total = 0.0, 0.0
    for index, probability in enumerate(categories[:-1]):
        cumulative += probability
        total += (cumulative - (1.0 if index >= realised else 0.0)) ** 2
    gaps = len(categories) - 1
    return total if kind == COUNT else total / gaps


def _require_whole(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OutcomeError(f"a count, a minute or a place is a whole number, got {value!r}")
    return value
