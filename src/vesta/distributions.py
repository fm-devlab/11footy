"""The seven kinds of distribution Vesta publishes, and their check (`docs/FORECAST_LOG.md` §6).

A distribution is kept in the very shape it has in the forecast log's JSON (dicts and lists of
floats), so that the writer, the reader of lines (`vesta.forecast_reading`) and the scores of the
research discipline read one representation and nothing is converted on the way. Standard
library only: the readers import the reader of lines, which imports this module.
"""

import math

BINARY = "binary"
CATEGORICAL = "categorical"
ORDERED = "ordered"
COUNT = "count"
JOINT_COUNT = "joint_count"
TIME_TO_EVENT = "time_to_event"
RANK = "rank"
KINDS = (BINARY, CATEGORICAL, ORDERED, COUNT, JOINT_COUNT, TIME_TO_EVENT, RANK)

# A published distribution sums to one up to the rounding of the floats it is made of; a larger
# gap is a bug of the model, never something to normalise away.
SUM_TOLERANCE = 1e-9


# A probability made as a sum of floats can stray outside [0, 1] by the last bits (a band holding
# every path weighs 1.0000000000000002); within this it is held at the bound before the check.
EDGE_TOLERANCE = 1e-12


def hold_at_edges(values: object) -> object:
    """The distribution with every float within `EDGE_TOLERANCE` beyond 0 or 1 held at the bound:
    the rounding of a sum, not a probability out of range (anything farther is left for the check
    to refuse). Whole numbers (band edges, the tail's start) are never touched."""
    if isinstance(values, float):
        if 1.0 < values <= 1.0 + EDGE_TOLERANCE:
            return 1.0
        if -EDGE_TOLERANCE <= values < 0.0:
            return 0.0
        return values
    if isinstance(values, dict):
        return {key: hold_at_edges(value) for key, value in values.items()}
    if isinstance(values, list):
        return [hold_at_edges(value) for value in values]
    return values


class DistributionError(ValueError):
    """A distribution that does not have the shape of its kind or does not sum to one."""


def list_probabilities(kind: str, values: object) -> list[float]:
    """Every probability of a distribution, in its order: what must be in [0, 1] and sum to 1."""
    if kind == BINARY:
        yes = _require_mapping(values, kind)["yes"]
        return [yes, 1.0 - yes]
    if kind == CATEGORICAL:
        return list(_require_mapping(values, kind).values())
    if kind == ORDERED:
        return [pair[1] for pair in _require_sequence(values, kind)]
    if kind == COUNT:
        mapping = _require_mapping(values, kind)
        return [*mapping["probabilities"], mapping["tail"]]
    if kind == JOINT_COUNT:
        mapping = _require_mapping(values, kind)
        return [p for row in mapping["matrix"] for p in row] + [mapping["tail"]]
    if kind == TIME_TO_EVENT:
        mapping = _require_mapping(values, kind)
        return [band[2] for band in mapping["bands"]] + [mapping["never"]]
    if kind == RANK:
        return list(_require_sequence(values, kind))
    raise DistributionError(f"unknown distribution kind {kind!r}; the kinds are {KINDS}")


def check_distribution(kind: str, values: object) -> None:
    """Raise `DistributionError` unless every probability is in [0, 1] and they sum to one."""
    try:
        probabilities = list_probabilities(kind, values)
    except (KeyError, IndexError, TypeError) as error:
        message = f"a {kind} distribution lacks a part of its shape: {error}"
        raise DistributionError(message) from error
    for probability in probabilities:
        if not (isinstance(probability, int | float) and 0.0 <= probability <= 1.0):
            raise DistributionError(f"a {kind} distribution holds {probability!r}, not in [0, 1]")
    total = math.fsum(probabilities)
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise DistributionError(f"a {kind} distribution sums to {total!r}, not to 1")
    if kind == COUNT and values["tail_from"] != len(values["probabilities"]):
        raise DistributionError("a count distribution's tail must start after its last count")
    if kind == JOINT_COUNT and len({len(row) for row in values["matrix"]}) > 1:
        raise DistributionError("a joint count distribution's matrix rows differ in length")


def compute_expected_count(values: dict) -> float:
    """The mean of a count distribution, its tail counted at `tail_from` (a lower bound, said so
    by the tail being published beside it)."""
    probabilities = values["probabilities"]
    body = math.fsum(count * p for count, p in enumerate(probabilities))
    return body + values["tail_from"] * values["tail"]


def _require_mapping(values: object, kind: str) -> dict:
    if not isinstance(values, dict):
        raise DistributionError(f"a {kind} distribution is a JSON object, got {type(values)}")
    return values


def _require_sequence(values: object, kind: str) -> list:
    if not isinstance(values, list):
        raise DistributionError(f"a {kind} distribution is a JSON array, got {type(values)}")
    return values
