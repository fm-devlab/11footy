"""The paired block bootstrap of a difference of scores.

Forecasts of the same week and competition share their news (an injury, a weather, a refereeing
line), so they are not independent: the interval of a mean difference is drawn by resampling
whole blocks (competition x ISO week).
Seeded, so a verdict is reproduced to the bit by its command.
"""

import math
import random
from datetime import datetime


def assign_block(competition: str, forecast_time_utc: str) -> str:
    """The block of a forecast: its competition and the ISO week of its forecast time."""
    when = datetime.fromisoformat(forecast_time_utc.replace("Z", "+00:00"))
    year, week, _ = when.isocalendar()
    return f"{competition}|{year}-{week:02d}"


def bootstrap_mean(
    values: list[float], blocks: list[str], repetitions: int, seed: int, coverage: float = 0.95
) -> tuple[float, float, float]:
    """The mean of `values` and its `coverage` percentile interval over resampled blocks.

    A paired comparison passes the per-forecast difference candidate minus champion.
    """
    if len(values) != len(blocks) or not values:
        raise ValueError("one block per value, and at least one value")
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for value, block in zip(values, blocks, strict=True):
        sums[block] = sums.get(block, 0.0) + value
        counts[block] = counts.get(block, 0) + 1
    names = sorted(sums)  # a fixed order, so the seed alone decides the draws
    block_sums = [sums[name] for name in names]
    block_counts = [counts[name] for name in names]
    generator = random.Random(seed)
    means = []
    for _ in range(repetitions):
        drawn = [generator.randrange(len(names)) for _ in names]
        total = math.fsum(block_sums[index] for index in drawn)
        means.append(total / sum(block_counts[index] for index in drawn))
    means.sort()
    tail = (1.0 - coverage) / 2.0
    return (
        math.fsum(values) / len(values),
        _interpolate_quantile(means, tail),
        _interpolate_quantile(means, 1.0 - tail),
    )


def _interpolate_quantile(ordered: list[float], level: float) -> float:
    """The linear-interpolation quantile of a sorted list (numpy's default rule)."""
    position = level * (len(ordered) - 1)
    below = math.floor(position)
    above = min(below + 1, len(ordered) - 1)
    return ordered[below] + (ordered[above] - ordered[below]) * (position - below)
