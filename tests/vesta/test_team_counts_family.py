"""Family A's team counts: the new keys read from one joint distribution, the joint reference as the
product of the sides, the pooled cup fit finding the terms it was drawn from, and the dominance of
shots over shots on target."""

import math
from dataclasses import replace

import numpy as np

from vesta.distributions import check_distribution
from vesta.forecast_reading import read_line
from vesta.quantities import load_quantities
from vesta.team_counts.counts_verdict import (
    build_product_reference,
    count_dominance_violations,
    fold_outcomes,
)
from vesta.team_counts.cups import CupCountMatch, fit_pooled
from vesta.team_counts.model import FAMILY_A_KEYS, count_distributions


def test_the_new_keys_are_in_the_catalogue_with_their_kinds():
    definitions = load_quantities()
    for keys in FAMILY_A_KEYS.values():
        total, team, sides = keys
        assert definitions[total].quantity.distribution_kind == "count"
        assert definitions[team].quantity.distribution_kind == "count"
        if sides is not None:
            assert definitions[sides].quantity.distribution_kind == "joint_count"


def test_the_shots_keys_agree_and_offsides_have_no_sides():
    total, home, away, sides = count_distributions("shots", 1, 10, 20, (14.0, 10.0), 9.0, 41, "v")
    for item, kind in ((total, "count"), (home, "count"), (away, "count"), (sides, "joint_count")):
        check_distribution(kind, item.values)
    assert math.isclose(read_line("count", total.values, "over=23.5"),
                        read_line("joint_count", sides.values, "total_over=23.5"),
                        abs_tol=1e-12)  # fmt: skip
    offsides = count_distributions("offsides", 1, 10, 20, (1.9, 1.7), 12.0, 16, "v")
    assert [item.quantity_key for item in offsides] == ["M19.offsides", "M19.offsides.team",
                                                       "M19.offsides.team"]  # fmt: skip
    assert ("M19.offsides.sides", "match") not in fold_outcomes("offsides", 3, 1, 16)
    assert fold_outcomes("offsides", 30, 1, 16)[("M19.offsides.team", "home")] == 15


def test_the_joint_reference_is_the_product_of_the_sides():
    reference = np.array(build_product_reference([(1, 0), (2, 1), (1, 1)], 4)["matrix"])
    assert math.isclose(reference.sum(), 1.0)
    home, away = reference.sum(axis=1), reference.sum(axis=0)
    assert np.allclose(reference, np.outer(home, away))
    assert math.isclose(home[1], 2.5 / 5.0)  # two of three, smoothed by half a case a cell


def _match(index, home, away, is_cup, counts, ratings=(None, None)):
    stamp = "2020-01-01T15:00:00Z"
    return CupCountMatch(index, "CL" if is_cup else "PL", "CL" if is_cup else "PL", is_cup,
                         2020, home, away, stamp, stamp, False, *ratings, *counts,
                         "feed")  # fmt: skip


def test_the_pooled_fit_finds_the_home_terms_and_the_elo_slope_it_was_drawn_from():
    rng = np.random.default_rng(11)
    rating = {team: 1500 + 150 * rng.normal() for team in range(40)}
    rows, index = [], 0
    for _ in range(4):
        for home in range(20):  # the league
            for away in range(20):
                if home != away:
                    index += 1
                    mu = (math.exp(2.4 + 0.12), math.exp(2.4))
                    rows.append(_match(index, home, away, False,
                                       (int(rng.poisson(mu[0])), int(rng.poisson(mu[1])))))
    for _ in range(3000):  # the cup, clubs the league never meets among them
        home, away = rng.choice(40, 2, replace=False)
        gap = (rating[home] - rating[away]) / 400
        mu = (math.exp(2.5 + 0.08 + 0.6 * gap), math.exp(2.5 - 0.6 * gap))
        index += 1
        rows.append(_match(index, int(home), int(away), True,
                           (int(rng.poisson(mu[0])), int(rng.poisson(mu[1]))),
                           (rating[home], rating[away])))  # fmt: skip
    fit = fit_pooled(rows, np.ones(len(rows)), 3.0)
    assert fit is not None
    assert abs(fit.home[False] - 0.12) < 0.03 and abs(fit.home[True] - 0.08) < 0.03
    assert abs(fit.elo - 0.6) < 0.08
    assert abs(fit.levels["CL"] - 2.5) < 0.05 and fit.size_nb > 20


def test_on_target_must_lie_below_shots():
    shots = {(1, "home"): np.array([0.1, 0.2, 0.3, 0.4])}
    below = {(1, "home"): np.array([0.4, 0.3, 0.2, 0.1])}
    above = {(1, "home"): np.array([0.0, 0.1, 0.2, 0.7])}
    assert count_dominance_violations(shots, below) == (0, 1)
    assert count_dominance_violations(shots, above) == (1, 1)


def test_a_season_that_did_not_record_offsides_is_unknown():
    from vesta.team_counts.cups import _drop_unrecorded_offsides

    recorded = [_match(index, 1, 2, True, (index % 3, 1), (1500.0, 1500.0)) for index in range(10)]
    unrecorded = [replace(_match(100 + index, 1, 2, True, (0, 0), (1500.0, 1500.0)), season=2017)
                  for index in range(10)]  # fmt: skip
    kept = _drop_unrecorded_offsides(recorded + unrecorded)
    assert all(match.home_count is not None for match in kept if match.season == 2020)
    assert all(match.home_count is None for match in kept if match.season == 2017)


def test_an_input_known_after_the_forecast_fails_the_run():
    import pytest

    from vesta.team_counts.counts_verdict import LeakageError, check_known_before, seconds_of
    from vesta.team_counts.model import CountMatch

    match = _match(1, 1, 2, True, (3, 2), (1500.0, 1500.0))
    check_known_before(match, seconds_of(match.forecast_time) - 1, "an input")  # in time: no error
    with pytest.raises(LeakageError):
        check_known_before(match, seconds_of(match.forecast_time), "an input")
    # the verdict's path: a calibration fitted on an outcome known after the forecast fails it
    from vesta.team_counts.counts_verdict import judge
    from vesta.team_counts.model import Calibration

    league = CountMatch(7, "PL", 2020, 1, 2, "2020-09-12T14:00:00Z", "2020-09-12T16:30:00Z", 5,
                        4, "report")  # fmt: skip
    history = {
        ("PL", 2019): [
            (seconds_of("2019-09-01T16:30:00Z"), fold_outcomes("corners", 5, 5, 21)),
            (seconds_of("2020-12-01T16:30:00Z"), fold_outcomes("corners", 9, 9, 21)),
        ]
    }
    forecasts = [(league, (5.0, 4.0), 10.0)]
    calm = {2020: seconds_of("2020-08-01T00:00:00Z")}
    judged = judge("corners", "leagues", forecasts, {2020: Calibration()}, history, 2015, 21,
                   load_quantities(), calm)  # fmt: skip
    assert judged.rows["M15.corners"]  # the late outcome of the reference is left out, no error
    with pytest.raises(LeakageError):
        judge("corners", "leagues", forecasts, {2020: Calibration()}, history, 2015, 21,
              load_quantities(), {2020: seconds_of("2020-12-01T16:30:00Z")})  # fmt: skip


def test_candidate_two_finds_an_overreacting_total_and_the_forecasts_spread():
    from vesta.team_counts.model import CountMatch, fit_candidate_two

    rng = np.random.default_rng(5)
    forecasts = []
    for index in range(2500):
        # the gap between the sides is real; the match's pace the forecast says is half noise,
        # so the sides' slopes split the difference and the total still overreacts
        gap, pace = rng.normal(0.0, 0.3), rng.normal(0.0, 0.25)
        said = (12.0 * math.exp(gap + pace), 11.0 * math.exp(-gap + pace))
        true = [mean * math.exp(-0.5 * pace) for mean in said]
        home, away = (int(rng.negative_binomial(8.0, 8.0 / (8.0 + mean))) for mean in true)
        stamp = "2020-01-01T15:00:00Z"
        forecasts.append((CountMatch(index, "PL", 2020, 1, 2, stamp, stamp, home, away, "feed"),
                          said, 1e6))  # fmt: skip
    calibration, sizes = fit_candidate_two(forecasts, 41, lambda match: "PL")
    assert calibration.total[1] < 0.9  # the total's slope pulls the overreaction back
    assert 5.0 < sizes["PL"] < 12.0  # the forecasts' own spread, not the weekly fit's Poisson


def test_a_recent_reference_reads_only_the_last_seasons():
    from vesta.team_counts.counts_verdict import judge, seconds_of
    from vesta.team_counts.model import Calibration, CountMatch

    league = CountMatch(7, "PL", 2020, 1, 2, "2020-09-12T14:00:00Z", "2020-09-12T16:30:00Z", 5,
                        4, "report")  # fmt: skip
    known = seconds_of("2019-06-01T00:00:00Z")
    history = {("PL", 2016): [(known, fold_outcomes("corners", 15, 15, 21))] * 50,
               ("PL", 2019): [(known, fold_outcomes("corners", 5, 4, 21))] * 50}  # fmt: skip
    forecasts = [(league, (5.0, 4.0), 1e6)]

    def reference_loss(window):
        judged = judge("corners", "leagues", forecasts, {2020: Calibration()}, history, 2015, 21,
                       load_quantities(), window=window)  # fmt: skip
        return judged.rows["M15.corners"][0][1]

    # every season before still sees 2016's thirty corners; the last two seasons only 2019's nine
    assert reference_loss(2) < reference_loss(None)
    assert reference_loss(3) < reference_loss(None) and reference_loss(5) == reference_loss(None)


def test_the_absolute_calibration_test_wants_a_small_error_and_every_bucket_in_its_band():
    from vesta.team_counts.counts_verdict import count_buckets_outside, judge_absolutely
    from vesta.team_counts.verdict import KeyVerdict

    calibrated = [(0.3, index % 10 < 3) for index in range(1000)]
    off = [(0.3, index % 10 < 4) for index in range(1000)]  # 40 % where 30 % was said
    assert count_buckets_outside(calibrated, 0.90) == (0, 1)
    assert count_buckets_outside(off, 0.90) == (1, 1)
    # two buckets hold 90 % together, each at 0.90 ** (1/2): a share just outside one bucket's
    # own 90 % band stays inside its joint band
    edge = [(0.3, index < 326) for index in range(1000)]
    edge += [(0.7, index % 10 < 7) for index in range(1000)]
    assert count_buckets_outside(edge, 0.90) == (0, 2)
    assert count_buckets_outside([line for line in edge if line[0] == 0.3], 0.90) == (1, 1)
    gained = KeyVerdict("M20.shots", "count", 1000, 2.9, 3.0, -0.1, -0.12, -0.08, 4, 5, None,
                        None, 0.0, 0.0, True, False)  # fmt: skip
    assert judge_absolutely(gained, calibrated, 3, (0.02, 0.90))[0].passed
    assert not judge_absolutely(gained, off, 3, (0.02, 0.90))[0].passed
    worse = replace(gained, calibration_model=0.03)
    assert not judge_absolutely(worse, calibrated, 3, (0.02, 0.90))[0].passed
    few = replace(gained, seasons_improved=2)
    assert not judge_absolutely(few, calibrated, 3, (0.02, 0.90))[0].passed
