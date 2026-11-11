"""The baseline B0 of the spine: the Dixon-Coles family, the world reader and the walk-forward,
on a small world built here with the world's own schema."""

import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

from vesta.spine.dixon_coles import (
    DixonColesFamily,
    DixonColesParameters,
    find_week_start,
    fit_goals,
    parse_instant_to_seconds,
)
from vesta.spine.match_forecast import average_log_loss
from vesta.spine.point_in_time import Fact, ForecastRequest, replay
from vesta.spine.timing import TimeRules
from vesta.spine.walk_forward import WalkForwardSettings, forecast_with_baseline
from vesta.spine.world import read_league_matches

ROOT = Path(__file__).resolve().parents[3]
SPINE = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))


def test_the_goals_fit_reproduces_the_reference_numbers():
    """Reference values computed on 24 October 2026 on the same inputs (this fit differed by 0.0
    in every parameter)."""
    state = np.random.RandomState(7)
    count, teams = 1200, 20
    home = state.randint(0, teams, count) + 100
    away = (home - 100 + state.randint(1, teams, count)) % teams + 100
    strength = state.normal(0, 0.3, teams)
    home_goals = state.poisson(np.exp(0.3 + strength[home - 100] - strength[away - 100] * 0.5))
    away_goals = state.poisson(np.exp(0.1 + strength[away - 100] - strength[home - 100] * 0.5))
    weights = np.exp(-0.003 * state.uniform(0, 1000, count))
    fit = fit_goals(home, away, home_goals, away_goals, weights, 2.0)
    assert fit.attack[:3].tolist() == pytest.approx(
        [0.14831826928485967, -0.027090208942762137, 0.15923020220652082], abs=1e-6
    )
    assert fit.home_advantage == pytest.approx(0.181061250039581, abs=1e-6)
    assert fit.level == pytest.approx(0.05240647446722888, abs=1e-6)


def test_the_week_starts_on_monday_at_midnight_utc():
    sunday_night = parse_instant_to_seconds("2026-10-25T23:59:59Z")
    assert find_week_start(sunday_night) == parse_instant_to_seconds("2026-10-19T00:00:00Z")
    assert find_week_start(
        parse_instant_to_seconds("2026-10-26T00:00:00Z")
    ) == parse_instant_to_seconds("2026-10-26T00:00:00Z")


def test_the_family_uses_only_the_results_known_at_the_weeks_monday():
    parameters = DixonColesParameters(0.003, 2.0, 1100, minimum_results=3)
    facts = [
        Fact(1, "PL", 2026, 1, 2, 3, 0, "2026-10-05T16:00:00Z"),
        Fact(2, "PL", 2026, 2, 3, 1, 1, "2026-10-12T16:00:00Z"),
        Fact(3, "PL", 2026, 3, 1, 0, 2, "2026-10-18T16:00:00Z"),
        Fact(4, "PL", 2026, 1, 3, 9, 0, "2026-10-21T16:00:00Z"),  # in the week: not used
    ]
    friday = ForecastRequest(9, "PL", 2026, 1, 2, "2026-10-23T19:00:00Z")
    before = replay([DixonColesFamily(parameters)], facts[:3], [friday])[0].features
    after = replay([DixonColesFamily(parameters)], facts, [friday])[0].features
    assert before == after
    too_few = replay([DixonColesFamily(parameters)], facts[:2], [friday])[0].features
    assert all(math.isnan(value) for value in too_few.values())


def test_the_reader_takes_the_pitch_score_and_leaves_out_matches_without_a_result(world):
    matches = read_league_matches(world, ["PL", "SA"], 2014, 2017)
    assert len(matches) == 2 * 4 * 90 - 1
    awarded = next(match for match in matches if match.match_id == 5)
    assert (awarded.home_goals, awarded.away_goals) != (0, 3)
    assert all(match.match_id != 6 for match in matches)
    kickoffs = [match.kickoff_utc for match in matches]
    assert kickoffs == sorted(kickoffs)


def test_the_baseline_forecasts_each_season_with_the_seasons_before_it(world):
    settings = WalkForwardSettings(2015, 2, 100, ("PL", "SA"))
    matches = read_league_matches(world, ["PL", "SA"], 2014, 2017)
    forecasts = forecast_with_baseline(
        matches, TimeRules.from_config(SPINE), DixonColesParameters.from_config(SPINE),
        settings, forecast_seasons=(2016, 2017),
    )  # fmt: skip
    assert {forecast.season for forecast in forecasts} == {2016, 2017}
    for forecast in forecasts:
        assert sum(forecast.probabilities) == pytest.approx(1.0, abs=1e-12)
    assert average_log_loss(forecasts) < math.log(3)  # better than knowing nothing
    again = forecast_with_baseline(
        matches, TimeRules.from_config(SPINE), DixonColesParameters.from_config(SPINE),
        settings, forecast_seasons=(2016, 2017),
    )  # fmt: skip
    assert again == forecasts  # reproduced to the bit


def test_the_baseline_never_reads_a_season_after_the_last_it_forecasts(world):
    settings = WalkForwardSettings(2015, 2, 100, ("PL", "SA"))
    matches = read_league_matches(world, ["PL", "SA"], 2014, 2017)
    parameters = DixonColesParameters.from_config(SPINE)
    rules = TimeRules.from_config(SPINE)
    with_future = forecast_with_baseline(matches, rules, parameters, settings, (2016,))
    without = forecast_with_baseline(
        [match for match in matches if match.season <= 2016], rules, parameters, settings, (2016,)
    )
    assert with_future == without
