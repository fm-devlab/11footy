"""The player thresholds: the minutes bins and the mixture are whole distributions, the rates
shrink to the group, the dispersion is found again, the match terms read the side's own level,
and a keeper's saves are the opponent's saved shots while he was on the pitch."""

import math
import sqlite3

import numpy as np
from scipy.stats import nbinom

from vesta.players.thresholds import (
    COUNTS,
    SAVES,
    Appearance,
    GroupRates,
    MatchTerms,
    count_cells,
    estimate_own_rates,
    estimate_reference_rates,
    fit_size,
    read_counts,
    split_into_minute_bins,
)

GROUP = GroupRates(np.full(len(COUNTS), 0.5), np.full(len(COUNTS), 0.4), 2.0, 1.3)


def _appearance(season=2020, minutes=90.0, value=2.0, conceded=1.0):
    return Appearance("2020-01-01T12:00:00Z", season, minutes, np.full(len(COUNTS), value),
                      conceded)  # fmt: skip


def test_the_minutes_if_he_plays_fall_in_bins_that_add_up():
    minutes = np.zeros(91)
    minutes[0], minutes[20], minutes[90] = 0.5, 0.1, 0.4  # a chance of not playing is left out
    weights, means = split_into_minute_bins(minutes)
    assert abs(weights.sum() - 1) < 1e-12
    assert abs(weights[0] - 0.2) < 1e-12 and abs(weights[3] - 0.8) < 1e-12
    assert means[0] == 20 and means[3] == 90


def test_the_mixture_is_a_whole_distribution_and_a_substitute_shoots_less():
    for size in (2.0, math.inf):
        cells = count_cells(np.array([0.3, 2.0]), np.array([0.5, 0.5]), size, 10)
        assert abs(cells.sum() - 1) < 1e-12 and (cells >= 0).all()
    starter = count_cells(np.array([2.0]), np.array([1.0]), 3.0, 10)
    substitute = count_cells(np.array([0.4]), np.array([1.0]), 3.0, 10)
    assert starter[2:].sum() > substitute[2:].sum()


def test_a_player_without_history_takes_his_groups_rates_and_the_reference_too():
    per_ninety, ratio = estimate_own_rates([], 2020, GROUP)
    assert np.allclose(per_ninety, GROUP.per_ninety) and ratio == GROUP.saves_per_goal
    assert np.allclose(estimate_reference_rates([], GROUP), GROUP.per_match)


def test_the_rates_read_two_seasons_shrink_and_skip_what_the_source_lacks():
    old = _appearance(season=2018, value=9.0)  # three seasons back: not read
    gap = Appearance("2020-01-01T12:00:00Z", 2020, 90.0, np.full(len(COUNTS), math.nan), math.nan)
    per_ninety, _ = estimate_own_rates([old, _appearance(), gap], 2020, GROUP)
    # one match (one back from the newest: 0.97) of two per 90, five nineties of the group
    expected = (0.97 * 2.0 + 5 * 0.5) / (0.97 + 5)
    assert np.allclose(per_ninety, expected)


def test_the_dispersion_is_found_again():
    rng = np.random.default_rng(3)
    means = rng.uniform(0.5, 3.0, 20_000)
    counts = nbinom.rvs(2.0, 2.0 / (2.0 + means), random_state=rng)
    assert fit_size(means, counts) in (1.5, 2.0, 3.0)
    assert math.isinf(fit_size(means, rng.poisson(means)))


def test_the_attack_term_is_the_sides_goals_against_its_own_level():
    terms = MatchTerms()
    terms.set_attack_factors([(1, 10, 20, 2.0, 1.0), (2, 10, 30, 1.0, 1.5)])
    assert terms.look_up_factor(1, 10, "attack") == 1.0  # its first match: no level yet
    assert abs(terms.look_up_factor(2, 10, "attack") - 0.5) < 1e-12
    assert terms.expected_goals[(2, 10)] == 1.5
    assert terms.look_up_factor(9, 10, "fouls") == 1.0  # no fit: no change


def test_a_keepers_saves_are_the_opponents_saved_shots_while_he_played():
    world = sqlite3.connect(":memory:")
    world.execute("CREATE TABLE player_match_statistics (match_id, player_id, shots,"
                  " shots_on_target, key_passes, fouls_committed, fouls_suffered, yellow_cards,"
                  " offsides, saves, layer)")  # fmt: skip
    world.execute("CREATE TABLE shots (match_id, team_id, minute, result, layer)")
    world.executemany("INSERT INTO player_match_statistics VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      [(1, 100, 0, 0, 0, 0, 1, 2, None, 0, "feed"),
                       (1, 101, 0, 0, 0, 0, 0, 0, None, 0, "feed"),
                       (1, 200, 3, 1, 1, 2, 0, 0, 1, None, "feed")])  # fmt: skip
    world.executemany("INSERT INTO shots VALUES (?,?,?,?,?)",
                      [(1, 20, 10, "saved", "tracked"), (1, 20, 50, "goal", "tracked"),
                       (1, 20, 70, "saved", "tracked"), (1, 20, 80, "saved", "commentary"),
                       (1, 10, 30, "saved", "tracked")])  # fmt: skip
    sheets = {(1, 10): {100: (True, 60, "G"), 101: (False, 30, None), 200: (True, 90, "F")}}
    found = read_counts(world, sheets, {1})
    starter, conceded = found[(1, 100)]
    assert starter[SAVES] == 1 and conceded == 1  # minutes 0-59: the save at 10, the goal at 50
    assert found[(1, 101)][0][SAVES] == 1 and found[(1, 101)][1] == 0  # from 60: the save at 70
    assert starter[COUNTS.index("booked")] == 1  # two yellows: booked
    assert math.isnan(found[(1, 200)][0][SAVES]) and found[(1, 200)][0][0] == 3


def test_a_sides_published_thresholds_are_whole_distributions_of_the_released_keys():
    from vesta.forecast_log.writing import PublishedDistribution
    from vesta.lineups.publication import TeamLineup
    from vesta.players.thresholds_publication import KEYS, ThresholdsOfTheDay

    day = ThresholdsOfTheDay.__new__(ThresholdsOfTheDay)
    day.groups, day.sizes = {None: GROUP, "forward": GROUP}, {("shots", "forward"): 3.0}
    day.histories, day.known_at, day.season, day.now = {}, {}, 2026, "2026-11-01T12:00:00Z"
    day.terms = MatchTerms()
    day.scales = {"P16.offsides": 0.9}  # candidate 2's scale of the season
    minutes = np.zeros(91)
    minutes[0], minutes[90], minutes[25] = 0.2, 0.6, 0.2
    lineup = TeamLineup(10, [PublishedDistribution("L6.minutes", 5, "", {
        "probabilities": minutes.tolist(), "tail_from": 91, "tail": 0.0})], "4-3-3", 0.5, [],
        ((5, "forward", 0.75, 0.8, 70.0), (6, "forward", 0.1, 0.2, 20.0)))  # fmt: skip
    found = day.build_side_distributions(lineup, 1)
    # player 6: no minutes; P14.saves is a goalkeeper's
    assert sorted(item.quantity_key for item in found) == sorted(set(KEYS) - {"P14.saves"})
    del day.scales["P16.offsides"]  # no scale for the season: offsides not published
    assert "P16.offsides" not in {
        item.quantity_key for item in day.build_side_distributions(lineup, 1)
    }
    found = day.build_side_distributions(lineup, 1) + [
        item for item in found if item.quantity_key == "P16.offsides"
    ]
    for item in found:
        assert item.subject_id == 5
        if item.quantity_key == "P9.booked":
            assert 0 < item.values["yes"] < 1
        else:
            total = sum(item.values["probabilities"]) + item.values["tail"]
            assert abs(total - 1) < 1e-9
