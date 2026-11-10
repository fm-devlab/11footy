"""Calibration buckets, the chance band and the block bootstrap."""

import random

import pytest

from vesta.research.bootstrap import assign_block, bootstrap_mean
from vesta.research.calibration import bucket_lines, compute_chance_band, measure_calibration_error


def test_lines_are_grouped_by_tenths_and_one_falls_in_the_last():
    groups = bucket_lines([(0.05, False), (0.12, True), (0.18, False), (1.0, True)])
    assert [(group.said_from, group.count) for group in groups] == [(0.0, 1), (0.1, 2), (0.9, 1)]
    assert groups[1].said_mean == pytest.approx(0.15)
    assert groups[1].happened_share == 0.5


def test_a_calibrated_forecaster_has_no_calibration_error():
    lines = [(0.25, True)] + [(0.25, False)] * 3 + [(0.75, True)] * 3 + [(0.75, False)]
    assert measure_calibration_error(lines) == pytest.approx(0.0)


def test_calibration_error_is_the_weighted_gap():
    lines = [(0.9, False)] * 2 + [(0.5, True)] * 2
    assert measure_calibration_error(lines) == pytest.approx((2 * 0.9 + 2 * 0.5) / 4)


def test_chance_band_holds_the_said_probability_and_narrows_with_the_count():
    low_small, high_small = compute_chance_band(0.3, 20)
    low_large, high_large = compute_chance_band(0.3, 2000)
    assert low_small <= 0.3 <= high_small
    assert low_large <= 0.3 <= high_large
    assert high_large - low_large < high_small - low_small


def test_chance_band_covers_ninety_percent_of_simulated_calibrated_shares():
    generator = random.Random(20261023)
    low, high = compute_chance_band(0.4, 100)
    shares = [sum(generator.random() < 0.4 for _ in range(100)) / 100 for _ in range(2000)]
    inside = sum(low <= share <= high for share in shares) / len(shares)
    assert 0.88 <= inside <= 0.97


def test_chance_band_of_a_certainty_is_a_point():
    assert compute_chance_band(0.0, 50) == (0.0, 0.0)
    assert compute_chance_band(1.0, 50) == (1.0, 1.0)


def test_block_is_the_competition_and_the_iso_week():
    assert assign_block("SA", "2026-10-23T18:45:00Z") == "SA|2026-43"
    assert assign_block("SA", "2026-10-25T12:30:00Z") == "SA|2026-43"
    assert assign_block("SA", "2026-10-26T19:00:00Z") == "SA|2026-44"


def test_bootstrap_is_reproduced_by_its_seed_and_holds_the_mean():
    generator = random.Random(7)
    values = [generator.gauss(-0.01, 0.2) for _ in range(400)]
    blocks = [f"PL|{index // 10}" for index in range(400)]
    first = bootstrap_mean(values, blocks, repetitions=500, seed=20261023)
    again = bootstrap_mean(values, blocks, repetitions=500, seed=20261023)
    assert first == again
    mean, low, high = first
    assert low < mean < high


def test_bootstrap_of_a_constant_is_the_constant():
    mean, low, high = bootstrap_mean([0.5] * 30, ["a", "b", "c"] * 10, 100, seed=1)
    assert mean == low == high == pytest.approx(0.5)


def test_bootstrap_needs_one_block_per_value():
    with pytest.raises(ValueError, match="one block per value"):
        bootstrap_mean([1.0, 2.0], ["a"], 10, seed=1)

def test_the_joint_band_holds_a_calibrated_forecasters_buckets_together_at_its_coverage():
    """Ten buckets each at 0.90 ** (1/10) are all inside together 90 % of the time: a calibrated
    forecaster simulated on ten buckets of 2,000 cases fails the joint band about one time in ten,
    the per-bucket band about two times in three."""
    import numpy as np

    from vesta.research.calibration import compute_chance_band, compute_joint_coverage

    assert compute_joint_coverage(0.90, 1) == 0.90
    assert compute_joint_coverage(0.90, 10) ** 10 == pytest.approx(0.90)
    state = np.random.RandomState(20261101)
    said = np.linspace(0.05, 0.95, 10)
    joint = [compute_chance_band(p, 2000, compute_joint_coverage(0.90, 10)) for p in said]
    alone = [compute_chance_band(p, 2000, 0.90) for p in said]
    trials = 400
    fails_joint = fails_alone = 0
    for _ in range(trials):
        shares = state.binomial(2000, said) / 2000
        fails_joint += any(not lo <= s <= hi for s, (lo, hi) in zip(shares, joint, strict=True))
        fails_alone += any(not lo <= s <= hi for s, (lo, hi) in zip(shares, alone, strict=True))
    assert fails_joint / trials < 0.16 and fails_alone / trials > 0.5
