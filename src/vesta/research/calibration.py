"""Calibration: what Vesta said beside how often it happened.

Lines of forecasts (a probability and whether it happened) are grouped by tenths of probability;
each group carries the band a perfectly calibrated forecaster would show on that many cases, so a
reader sees whether a gap is larger than chance (`docs/FORECAST_LOG.md` §4,
`calibration_buckets`). The same groups give the expected calibration error that a verdict
must not worsen.
"""

import math
from dataclasses import dataclass

BUCKETS = 10  # tenths of probability, the public calibration's grain
BAND_COVERAGE = 0.90  # the chance band holds 90 % of the shares of a calibrated forecaster


@dataclass(frozen=True)
class Bucket:
    said_from: float
    said_to: float
    said_mean: float
    happened_share: float
    count: int
    chance_band_low: float
    chance_band_high: float


def bucket_lines(lines: list[tuple[float, bool]]) -> list[Bucket]:
    """The non-empty tenths of `lines` (probability, happened), lowest first."""
    groups: dict[int, list[tuple[float, bool]]] = {}
    for probability, happened in lines:
        groups.setdefault(_find_tenth(probability), []).append((probability, happened))
    result = []
    for index in sorted(groups):
        group = groups[index]
        count = len(group)
        said_mean = math.fsum(p for p, _ in group) / count
        low, high = compute_chance_band(said_mean, count)
        result.append(
            Bucket(
                said_from=index / BUCKETS,
                said_to=(index + 1) / BUCKETS,
                said_mean=said_mean,
                happened_share=sum(1 for _, happened in group if happened) / count,
                count=count,
                chance_band_low=low,
                chance_band_high=high,
            )
        )
    return result


def measure_calibration_error(lines: list[tuple[float, bool]]) -> float:
    """The gap between said and happened, averaged over the buckets weighted by their size."""
    groups = bucket_lines(lines)
    total = sum(group.count for group in groups)
    if total == 0:
        raise ValueError("no lines to measure the calibration of")
    gaps = (group.count * abs(group.said_mean - group.happened_share) for group in groups)
    return math.fsum(gaps) / total


def compute_chance_band(
    probability: float, count: int, coverage: float = BAND_COVERAGE
) -> tuple[float, float]:
    """The shares between which a calibrated forecaster's hit rate falls with probability
    `coverage` on `count` cases: the exact binomial quantiles, as shares."""
    if count <= 0:
        raise ValueError("a chance band needs at least one case")
    tail = (1.0 - coverage) / 2.0
    cumulative, low, high = 0.0, None, count
    for hits in range(count + 1):
        cumulative += _compute_binomial_probability(hits, count, probability)
        if low is None and cumulative >= tail:
            low = hits
        if cumulative >= 1.0 - tail:
            high = hits
            break
    return low / count, high / count


def compute_joint_coverage(coverage: float, buckets_held: int) -> float:
    """Each bucket's chance band's coverage so that a calibrated forecaster's `buckets_held`
    buckets all fall inside their bands with probability `coverage` together (Sidak:
    `coverage ** (1 / k)`, the buckets taken as independent). The absolute calibration test
    against a frequency reference uses it (`research.toml` `[verdict]`,
    1 November 2026): a band of `coverage` for each of about ten buckets alone would fail a
    calibrated model about two times in three."""
    if buckets_held <= 0:
        raise ValueError("a joint band needs at least one bucket")
    return coverage ** (1.0 / buckets_held)


def _compute_binomial_probability(hits: int, count: int, probability: float) -> float:
    if probability <= 0.0:
        return 1.0 if hits == 0 else 0.0
    if probability >= 1.0:
        return 1.0 if hits == count else 0.0
    log_probability = (
        math.lgamma(count + 1)
        - math.lgamma(hits + 1)
        - math.lgamma(count - hits + 1)
        + hits * math.log(probability)
        + (count - hits) * math.log1p(-probability)
    )
    return math.exp(log_probability)


def _find_tenth(probability: float) -> int:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"a probability is in [0, 1], got {probability!r}")
    return min(int(probability * BUCKETS), BUCKETS - 1)  # 1.0 falls in the last tenth
