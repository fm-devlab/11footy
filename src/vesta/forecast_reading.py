"""The reader of lines: the probability of one line read from a published distribution.

Every line the window shows ("over 9.5 corners", "home win", "scores 2 or more") is defined here
once (`docs/FORECAST_LOG.md` §6, decision A2 of 23 October 2026). Vesta calls it to write the
`probabilities` rows of the pre-match and full-time stages; the readers import it to read the
lines of a stored distribution, so the two can never disagree. It is arithmetic on stored numbers,
standard library only.

A line key is an operation, alone or with its arguments after `=`:

- binary: `yes`, `no`;
- categorical, ordered: `outcome=<name>`, `any=<name>,<name>,...`;
- count: `over=<k.5>`, `under=<k.5>`, `exactly=<k>`, `at_least=<k>`, `at_most=<k>`,
  `between=<a>,<b>` (both included), `odd`, `even`;
- joint_count (rows home goals, columns away goals): `home_win`, `draw`, `away_win`,
  `score=<h>-<a>`, `home_scores`, `away_scores`, `both_score`, `total_over=<k.5>`,
  `total_under=<k.5>`, `home_over=<k.5>`, `away_over=<k.5>`, `margin=<home or away>,<k>` (wins
  by exactly k), `margin_at_least=<home or away>,<k>` (wins by k or more);
- time_to_event: `by=<minute>` (the end of a band), `band=<start>-<end>`, `never`;
- rank: `place=<k>`, `top=<k>`, `bottom=<k>`, `between=<a>,<b>`.

A line that needs to look inside the tail of a distribution (an exact count beyond the published
counts, a parity, a result decided outside the matrix) is refused while the tail holds any
probability: the model must publish a support wide enough for the lines of its quantity.
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


class LineError(ValueError):
    """A line key that does not exist for the kind, or that the distribution cannot answer."""


def round_to_whole_percentages(probabilities: list[float]) -> list[int]:
    """The whole percentages a set of outcomes that exclude each other is shown as, adding up to
    100: each rounded down, then the missing points one at a time to the largest remainders, ties
    broken by the order given (the rule of the percentages shown between 1 and 99, point 2). The
    percentages are read to 10^-9, so a float's last bit never moves a point (0.29 is 29)."""
    scaled = [round(100 * probability, 9) for probability in probabilities]
    shown = [math.floor(value) for value in scaled]
    by_remainder = sorted(range(len(scaled)), key=lambda index: (shown[index] - scaled[index],
                                                                 index))  # fmt: skip
    for index in by_remainder[:max(100 - sum(shown), 0)]:
        shown[index] += 1
    return shown


def read_line(kind: str, values: object, line_key: str) -> float:
    """The probability of `line_key` under the distribution `values` of kind `kind`.

    A sum of floats can stray from [0, 1] by the last bit (1.0000000000000002 for a line that
    covers the whole distribution); the result is held to [0, 1], since a distribution that sums
    to one is checked before it is written (`vesta.distributions.check_distribution`)."""
    operation, _, arguments = line_key.partition("=")
    reader = _READERS.get(kind)
    if reader is None:
        raise LineError(f"no lines for the distribution kind {kind!r}")
    return min(max(reader(values, operation, arguments, line_key), 0.0), 1.0)


def _read_binary(values: dict, operation: str, arguments: str, line_key: str) -> float:
    if operation == "yes" and not arguments:
        return values["yes"]
    if operation == "no" and not arguments:
        return 1.0 - values["yes"]
    raise LineError(f"unknown binary line {line_key!r}")


def _read_named(values: dict | list, operation: str, arguments: str, line_key: str) -> float:
    probabilities = dict(values) if isinstance(values, list) else values
    if operation == "outcome":
        names = [arguments]
    elif operation == "any":
        names = arguments.split(",")
    else:
        raise LineError(f"unknown line {line_key!r} of named outcomes")
    missing = [name for name in names if name not in probabilities]
    if missing:
        raise LineError(f"line {line_key!r}: no outcome named {missing}")
    return math.fsum(probabilities[name] for name in names)


def _count(values: dict, operation: str, arguments: str, line_key: str) -> float:
    probabilities, tail_from, tail = values["probabilities"], values["tail_from"], values["tail"]
    if operation in ("over", "under"):
        threshold = _parse_half_line(arguments, line_key)
        at_least = math.ceil(threshold)
        over = _sum_tail(probabilities, tail_from, tail, at_least)
        return over if operation == "over" else 1.0 - over
    if operation == "at_least":
        return _sum_tail(probabilities, tail_from, tail, _parse_whole(arguments, line_key))
    if operation == "at_most":
        return 1.0 - _sum_tail(
            probabilities, tail_from, tail, _parse_whole(arguments, line_key) + 1
        )
    if operation == "exactly":
        count = _parse_whole(arguments, line_key)
        _refuse_inside_tail(count >= tail_from, tail, line_key)
        return probabilities[count] if count < len(probabilities) else 0.0
    if operation == "between":
        low, high = _parse_whole_pair(arguments, line_key)
        _refuse_inside_tail(high >= tail_from, tail, line_key)
        return math.fsum(probabilities[low : high + 1])
    if operation in ("odd", "even") and not arguments:
        _refuse_inside_tail(True, tail, line_key)
        remainder = 1 if operation == "odd" else 0
        return math.fsum(p for count, p in enumerate(probabilities) if count % 2 == remainder)
    raise LineError(f"unknown count line {line_key!r}")


def _read_joint_count(values: dict, operation: str, arguments: str, line_key: str) -> float:
    matrix, tail = values["matrix"], values["tail"]
    cells = [(home, away, p) for home, row in enumerate(matrix) for away, p in enumerate(row)]
    if operation == "score":
        home, _, away = arguments.partition("-")
        home_goals, away_goals = _parse_whole(home, line_key), _parse_whole(away, line_key)
        inside = home_goals < len(matrix) and away_goals < len(matrix[0])
        _refuse_inside_tail(not inside, tail, line_key)
        return matrix[home_goals][away_goals] if inside else 0.0
    _refuse_inside_tail(True, tail, line_key)
    test = _choose_joint_test(operation, arguments, line_key)
    return math.fsum(p for home, away, p in cells if test(home, away))


def _choose_joint_test(operation: str, arguments: str, line_key: str):
    """The condition on (home goals, away goals) that a joint line counts."""
    simple = {
        "home_win": lambda home, away: home > away,
        "draw": lambda home, away: home == away,
        "away_win": lambda home, away: home < away,
        "home_scores": lambda home, _away: home > 0,
        "away_scores": lambda _home, away: away > 0,
        "both_score": lambda home, away: home > 0 and away > 0,
    }
    if operation in simple and not arguments:
        return simple[operation]
    if operation in ("total_over", "total_under", "home_over", "away_over"):
        threshold = _parse_half_line(arguments, line_key)
        if operation == "total_over":
            return lambda home, away: home + away > threshold
        if operation == "total_under":
            return lambda home, away: home + away < threshold
        if operation == "home_over":
            return lambda home, _away: home > threshold
        return lambda _home, away: away > threshold
    if operation in ("margin", "margin_at_least"):
        side, _, size = arguments.partition(",")
        margin = _parse_whole(size, line_key)
        if margin < 1:
            raise LineError(f"line {line_key!r}: a winning margin is at least 1")
        exact = operation == "margin"
        if side == "home":
            return lambda home, away: home - away == margin if exact else home - away >= margin
        if side == "away":
            return lambda home, away: away - home == margin if exact else away - home >= margin
    raise LineError(f"unknown joint count line {line_key!r}")


def _read_time_to_event(values: dict, operation: str, arguments: str, line_key: str) -> float:
    bands = values["bands"]
    if operation == "never" and not arguments:
        return values["never"]
    if operation == "by":
        minute = _parse_whole(arguments, line_key)
        if minute not in {end for _, end, _ in bands}:
            raise LineError(f"line {line_key!r}: {minute} is not the end of a published band")
        return math.fsum(probability for _, end, probability in bands if end <= minute)
    if operation == "band":
        start, _, end = arguments.partition("-")
        wanted = (_parse_whole(start, line_key), _parse_whole(end, line_key))
        for band_start, band_end, probability in bands:
            if (band_start, band_end) == wanted:
                return probability
        raise LineError(f"line {line_key!r}: no published band {wanted}")
    raise LineError(f"unknown time-to-event line {line_key!r}")


def _rank(values: list, operation: str, arguments: str, line_key: str) -> float:
    places = len(values)
    if operation == "between":
        low, high = _parse_whole_pair(arguments, line_key)
    elif operation in ("place", "top", "bottom"):
        rank = _parse_whole(arguments, line_key)
        low, high = {"place": (rank, rank), "top": (1, rank),
                     "bottom": (places - rank + 1, places)}[operation]
    else:
        raise LineError(f"unknown rank line {line_key!r}")
    if not 1 <= low <= high <= places:
        raise LineError(f"line {line_key!r}: places run from 1 to {places}")
    return math.fsum(values[low - 1 : high])


def _sum_tail(probabilities: list[float], tail_from: int, tail: float, at_least: int) -> float:
    """P(count >= at_least): the published counts from `at_least`, plus the tail when the tail
    lies wholly at or above it."""
    if at_least > tail_from and tail > 0.0:
        raise LineError(f"at least {at_least} looks inside the tail, which starts at {tail_from}")
    return math.fsum(probabilities[at_least:]) + tail


def _refuse_inside_tail(needs_tail: bool, tail: float, line_key: str) -> None:
    if needs_tail and tail > 0.0:
        raise LineError(f"line {line_key!r} needs the inside of the tail, which holds {tail!r}")


def _parse_half_line(text: str, line_key: str) -> float:
    try:
        threshold = float(text)
    except ValueError as error:
        raise LineError(f"line {line_key!r}: {text!r} is not a number") from error
    if threshold < 0 or threshold % 1 != 0.5:
        raise LineError(f"line {line_key!r}: over and under take a half line such as 2.5")
    return threshold


def _parse_whole(text: str, line_key: str) -> int:
    if not text.isdigit():
        raise LineError(f"line {line_key!r}: {text!r} is not a whole number")
    return int(text)


def _parse_whole_pair(text: str, line_key: str) -> tuple[int, int]:
    low, _, high = text.partition(",")
    pair = _parse_whole(low, line_key), _parse_whole(high, line_key)
    if pair[0] > pair[1]:
        raise LineError(f"line {line_key!r}: the first bound is above the second")
    return pair


_READERS = {
    BINARY: _read_binary,
    CATEGORICAL: _read_named,
    ORDERED: _read_named,
    COUNT: _count,
    JOINT_COUNT: _read_joint_count,
    TIME_TO_EVENT: _read_time_to_event,
    RANK: _rank,
}
