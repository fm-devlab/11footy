"""Team counts: the three keys of a count read from one joint distribution, and a weekly fit that
finds the strengths it was drawn from."""

import math

import numpy as np

from vesta.distributions import check_distribution
from vesta.forecast_reading import read_line
from vesta.team_counts.model import (
    CountMatch,
    FitParameters,
    count_distributions,
    fit_week,
    make_side_distribution,
)
from vesta.team_counts.verdict import fold_outcomes, smooth_frequencies


def test_the_keys_of_a_count_agree_and_sum_to_one():
    total, home, away, sides = count_distributions("corners", 1, 10, 20, (5.5, 4.0), 12.0, 21, "v")
    for item, kind in ((total, "count"), (home, "count"), (away, "count"), (sides, "joint_count")):
        check_distribution(kind, item.values)
    over = read_line("count", total.values, "over=9.5")
    assert math.isclose(over, read_line("joint_count", sides.values, "total_over=9.5"),
                        abs_tol=1e-12)  # fmt: skip
    assert home.subject_id == 10 and away.subject_id == 20
    assert abs(sum(p * k for k, p in enumerate(home.values["probabilities"])) - 5.5) < 0.05


def test_a_side_is_poisson_without_overdispersion_and_folds_its_tail():
    poisson = make_side_distribution(3.0, math.inf, 11)
    assert math.isclose(poisson.sum(), 1.0) and poisson[-1] > 0
    assert make_side_distribution(3.0, 2.0, 11)[0] > poisson[0]  # a wider law puts more on nothing


def test_a_weekly_fit_finds_the_home_advantage_it_was_drawn_from():
    rng = np.random.default_rng(7)
    teams = list(range(10))
    strength = rng.normal(0, 0.2, 10)
    matches, day = [], 0
    for _ in range(6):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                day += 1
                stamp = f"2020-{1 + day // 400:02d}-01T15:00:00Z"
                mu = (math.exp(1.5 + 0.15 + strength[home]), math.exp(1.5 + strength[away]))
                matches.append(CountMatch(day, "PL", 2020, home, away, stamp, stamp,
                                          int(rng.poisson(mu[0])), int(rng.poisson(mu[1])),
                                          "report"))  # fmt: skip
    week = 1_700_000_000  # after every match above
    fit = fit_week(matches, week, FitParameters(10_000, 1e9, 0.5, 100, 21))
    assert fit is not None and abs(fit.home - 0.15) < 0.05 and fit.size_nb > 20


def test_what_happened_is_folded_like_the_distributions():
    happened = fold_outcomes("cards", 12, 3, 11)
    assert happened[("M17.cards.team", "home")] == 10
    assert happened[("M17.cards.sides", "match")] == (10, 3)
    reference = smooth_frequencies("count", [3, 4, 30 // 3], 11, total=True)
    assert len(reference["probabilities"]) == 21
