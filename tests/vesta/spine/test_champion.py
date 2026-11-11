"""The champion of the spine: the player ratings, the blend, the champion's walk-forward."""

import math
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vesta.spine.announced_eleven import (
    SHEET_OUT,
    AnnouncedElevenFamily,
    count_changed_starters,
    find_announced_elevens,
)
from vesta.spine.dixon_coles import DixonColesParameters
from vesta.spine.match_forecast import average_log_loss
from vesta.spine.player_ratings import (
    PlayerRatingsFamily,
    PlayerRatingsParameters,
    RatingsState,
    blend,
)
from vesta.spine.timing import TimeRules
from vesta.spine.walk_forward import (
    MatchFacts,
    WalkForwardSettings,
    forecast_with_champion,
)
from vesta.spine.world import read_league_matches

ROOT = Path(__file__).resolve().parents[3]
SPINE = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
PARAMETERS = PlayerRatingsParameters(0.003, 0.6, 1100, 60, 0.7, 0.5, 0.0)


def synthetic_ratings_state(family: PlayerRatingsFamily) -> RatingsState:
    """300 matches of two leagues of ten clubs, thirteen players a side, the first a goalkeeper;
    the same data the reference fit was given on 24 October 2026."""
    state_rng = np.random.RandomState(3)
    state = RatingsState()
    for index in range(300):
        competition = ("PL", "SA")[index % 2]
        offset = 100 if competition == "PL" else 200
        home, away = state_rng.choice(10, 2, replace=False) + offset
        players = []
        for side, team in ((0, home), (1, away)):
            for shirt in range(13):
                minutes = float(90 if shirt < 11 else state_rng.randint(1, 45))
                players.append((side, int(team * 100 + shirt), minutes, shirt < 11, shirt == 0))
        home_goals, away_goals = int(state_rng.poisson(1.4)), int(state_rng.poisson(1.1))
        home_expected, away_expected = state_rng.gamma(2, 0.7), state_rng.gamma(2, 0.55)
        state.known_at_seconds.append(1_600_000_000 + index * 86_400)
        state.leagues.append(family.leagues.index(competition))
        state.home_targets.append(family._blend_target(home_goals, float(home_expected)))
        state.away_targets.append(family._blend_target(away_goals, float(away_expected)))
        state.players.append(tuple(players))
    return state


def test_the_ratings_fit_reproduces_the_reference_numbers():
    """The reference fit (goalkeeper out of the attack) gave a difference
    of 0.0 in every parameter on these data; its values are frozen here."""
    family = PlayerRatingsFamily(PARAMETERS, ("PL", "SA"))
    state = synthetic_ratings_state(family)
    week = 1_600_000_000 + 400 * 86_400
    decay = np.exp(-0.003 * (week - np.array(state.known_at_seconds)) / 86_400)
    fit = family._fit(state, 0, 300, decay)
    assert len(fit.players) == 260
    assert fit.attack[[5, 17, 40]].tolist() == pytest.approx(
        [0.014594652827311807, -0.030897505289112406, 0.017047415486935398], abs=1e-6
    )
    assert fit.defence[[5, 17]].tolist() == pytest.approx(
        [0.025826075365360874, 0.04460799788257232], abs=1e-6
    )
    assert fit.home_advantage == pytest.approx(0.18617879364625373, abs=1e-6)
    assert fit.level.tolist() == pytest.approx([0.09653549272545617, 0.13751831132845618],
                                               abs=1e-6)  # fmt: skip
    assert fit.attack[0] == pytest.approx(0.0, abs=1e-6)  # a goalkeeper has no attack


def test_the_blend_is_geometric_and_falls_back_to_the_team_model():
    assert blend(2.0, 0.5, 0.5) == pytest.approx(1.0)
    assert blend(2.0, float("nan"), 0.5) == 2.0
    assert blend(1.7, 1.2, 0.0) == pytest.approx(1.7)


def facts_for(matches) -> dict[int, MatchFacts]:
    """Our expected goals near the goals, and the eleven of each side with a goalkeeper."""
    state = np.random.RandomState(11)
    facts = {}
    for match in matches:
        players = tuple(
            (side, team * 100 + shirt, 90.0, True, shirt == 0)
            for side, team in ((0, match.home_team_id), (1, match.away_team_id))
            for shirt in range(11)
        )
        facts[match.match_id] = MatchFacts(
            max(match.home_goals + state.normal(0, 0.6), 0.1),
            max(match.away_goals + state.normal(0, 0.6), 0.1),
            players,
        )
    return facts


def test_the_champion_forecasts_each_season_with_the_seasons_before_it(world):
    matches = read_league_matches(world, ["PL", "SA"], 2014, 2017)
    settings = WalkForwardSettings(2015, 2, 100, ("PL", "SA"))
    arguments = (
        matches, facts_for(matches), TimeRules.from_config(SPINE),
        DixonColesParameters.from_config(SPINE, "dixon_coles_expected_goals"),
        PlayerRatingsParameters.from_config(SPINE), 3, settings, (2016, 2017),
    )  # fmt: skip
    forecasts = forecast_with_champion(*arguments)
    assert {forecast.season for forecast in forecasts} == {2016, 2017}
    for forecast in forecasts:
        assert sum(forecast.probabilities) == pytest.approx(1.0, abs=1e-12)
    assert average_log_loss(forecasts) < math.log(3)
    assert forecast_with_champion(*arguments) == forecasts  # reproduced to the bit


def test_the_champion_parameters_are_the_reference_values():
    team = DixonColesParameters.from_config(SPINE, "dixon_coles_expected_goals")
    assert (team.expected_goals_weight, team.time_decay_per_day, team.ridge) == (0.8, 0.004, 4.0)
    assert (team.unknown_club_attack, team.unknown_club_defence) == (-0.25, -0.25)
    ratings = PlayerRatingsParameters.from_config(SPINE)
    assert (ratings.blend_weight, ratings.goalkeeper_attack_weight) == (0.5, 0.0)
    assert SPINE["champion_calibration"]["seasons"] == 3


def test_the_announced_eleven_candidate_differs_from_the_champion_only_by_the_eleven(world):
    """H4: forecast an hour before kick-off, the candidate reads the match's own starters; where
    they are the last match's (every match here fields the same eleven), it is the champion."""
    matches = read_league_matches(world, ["PL", "SA"], 2014, 2017)
    settings = WalkForwardSettings(2015, 2, 100, ("PL", "SA"))
    facts = facts_for(matches)
    ratings = PlayerRatingsParameters.from_config(SPINE)
    arguments = (
        matches, facts, TimeRules.from_config(SPINE),
        DixonColesParameters.from_config(SPINE, "dixon_coles_expected_goals"),
        ratings, 3, settings, (2016, 2017),
    )  # fmt: skip
    champion = forecast_with_champion(*arguments, earlier=SHEET_OUT)
    at_kickoff = {item.match_id: item for item in forecast_with_champion(*arguments)}
    assert all(item.forecast_time < at_kickoff[item.match_id].forecast_time for item in champion)
    announced = find_announced_elevens(facts)
    same = forecast_with_champion(*arguments, earlier=SHEET_OUT, ratings=AnnouncedElevenFamily(
        ratings, settings.competitions, announced))  # fmt: skip
    assert same == champion
    # the home side announces another eleven (here its opponent's players): only that match moves
    target = champion[-1].match_id
    home, away = announced[target]
    rotated = dict(announced)
    rotated[target] = (away, away)
    moved = {item.match_id: item for item in forecast_with_champion(
        *arguments, earlier=SHEET_OUT, ratings=AnnouncedElevenFamily(
            ratings, settings.competitions, rotated))}  # fmt: skip
    assert moved[target].probabilities != same[-1].probabilities
    assert all(moved[item.match_id] == item for item in same if item.match_id != target)


def test_changed_by_counts_the_starters_new_since_the_last_match():
    first = SimpleNamespace(match_id=1, kickoff_utc="2020-01-01T15:00:00Z", home_team_id=10,
                            away_team_id=20)  # fmt: skip
    second = SimpleNamespace(match_id=2, kickoff_utc="2020-01-08T15:00:00Z", home_team_id=20,
                             away_team_id=10)  # fmt: skip
    eleven = tuple(range(100, 111))
    other = tuple(range(200, 211))
    rotated = (*eleven[:8], 300, 301, 302)
    facts = {1: MatchFacts(1.0, 1.0, tuple((0, p, 90.0, True, False) for p in eleven)
                           + tuple((1, p, 90.0, True, False) for p in other)),
             2: MatchFacts(1.0, 1.0, tuple((0, p, 90.0, True, False) for p in other)
                           + tuple((1, p, 90.0, True, False) for p in rotated))}  # fmt: skip
    assert count_changed_starters(facts, [second, first]) == {(2, 20): 0, (2, 10): 3}
