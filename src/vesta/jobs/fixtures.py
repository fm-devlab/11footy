"""The matches still to play and their forecasts, made now, by the same models the verdicts
judged.

- The leagues: the champion (`vesta.spine.walk_forward.champion_forecasts`), its families
  replayed on every fact known now with the fixtures as requests at this instant, the matrix
  calibrated on the `calibration_seasons` seasons before the fixture's season; each fixture's
  terms are kept for its explanation.
- The UEFA cups: the cups model (`vesta.spine.cups`), every club's rating after the matches
  played, the mapping fitted on the cup matches of the seasons before.
- The national teams: the nations model at this instant (`vesta.season.strength`).
A fixture is forecast once both teams have played their previous match (the forecast log's
choice C: the first publication waits for it).
"""

import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from vesta.expected_goals.dataset import is_dataset, read_rated_shots
from vesta.expected_goals.model import ModelSettings
from vesta.expected_goals.shots import read_shots
from vesta.expected_goals.walk_forward import WalkForwardSettings as ShotWalkForwardSettings
from vesta.expected_goals.walk_forward import (
    rate_current_season,
    rate_every_season,
    sum_expected_goals_by_match,
)
from vesta.jobs.past_cache import KeptPast, hash_inputs, read_kept, write_kept
from vesta.season.strength import build_nations_strength
from vesta.spine.cups import (
    ClubMatch,
    CupsParameters,
    fit_mapping,
    make_rating_lookup,
    rate_matches,
)
from vesta.spine.dixon_coles import DixonColesParameters, ExpectedGoalsDixonColesFamily
from vesta.spine.explanation import EloParts, ForecastParts
from vesta.spine.nations import NationsParameters
from vesta.spine.player_ratings import PlayerRatingsFamily, PlayerRatingsParameters, blend
from vesta.spine.point_in_time import ForecastRequest, replay
from vesta.spine.score_matrix import score_matrices
from vesta.spine.timing import TimeRules
from vesta.spine.walk_forward import (
    ForecastableMatch,
    MatchFacts,
    WalkForwardSettings,
    _build_events,
    _fit_calibration_or_neutral,
    _keep_forecastable,
)
from vesta.spine.world import (
    RESULT_STATUSES,
    read_club_matches,
    read_league_matches,
    read_match_players,
    read_nation_matches,
)

TO_PLAY = ("scheduled", "postponed")


@dataclass(frozen=True)
class Fixture:
    match_id: int
    competition: str
    kind: str  # league, cup or nation
    season: int
    home_team_id: int
    away_team_id: int
    kickoff_utc: str
    neutral: bool
    stage: str


@dataclass(frozen=True)
class FixtureForecast:
    fixture: Fixture
    matrix: np.ndarray
    home_intensity: float
    away_intensity: float
    parts: ForecastParts | EloParts | None


def read_fixtures(world: sqlite3.Connection, now: str, hours: float, leagues: list[str],
                  cups: list[str]) -> list[Fixture]:  # fmt: skip
    """The matches of the perimeter to play from now to `hours` ahead whose two teams have both
    played their previous match."""
    until = (datetime.fromisoformat(now.replace("Z", "+00:00")) + timedelta(hours=hours))
    until_text = until.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = world.execute(
        """
        SELECT m.match_id, c.code, c.kind, s.season, m.home_team_id, m.away_team_id,
               m.kickoff_utc, m.neutral_venue, COALESCE(m.stage, ''), h.kind, a.kind
        FROM matches m
        JOIN seasons s ON s.season_id = m.season_id
        JOIN competitions c ON c.competition_id = s.competition_id
        JOIN teams h ON h.team_id = m.home_team_id
        JOIN teams a ON a.team_id = m.away_team_id
        WHERE m.status IN ('scheduled', 'postponed') AND m.kickoff_utc > ? AND m.kickoff_utc <= ?
        ORDER BY m.kickoff_utc, m.match_id
        """,
        (now, until_text),
    ).fetchall()
    fixtures = []
    for match_id, code, _, season, home, away, kickoff, neutral, stage, h_kind, a_kind in rows:
        if code in leagues:
            sort = "league"
        elif code in cups:
            sort = "cup"
        elif h_kind == "nation" and a_kind == "nation":
            sort = "nation"
        else:
            continue
        fixture = Fixture(match_id, code, sort, season, home, away, kickoff, bool(neutral), stage)
        if _were_previous_matches_played(world, fixture):
            fixtures.append(fixture)
    return fixtures


def _were_previous_matches_played(world: sqlite3.Connection, fixture: Fixture) -> bool:
    for team in (fixture.home_team_id, fixture.away_team_id):
        row = world.execute(
            "SELECT status FROM matches WHERE (home_team_id = ? OR away_team_id = ?)"
            " AND kickoff_utc < ? ORDER BY kickoff_utc DESC LIMIT 1",
            (team, team, fixture.kickoff_utc),
        ).fetchone()
        if row is not None and row[0] not in RESULT_STATUSES + ("cancelled", "abandoned"):
            return False
    return True


def forecast_league_fixtures(
    world: sqlite3.Connection,
    fixtures: list[Fixture],
    now: str,
    spine: dict,
    expected_goals: dict,
    season: int | None = None,
    kept_in: Path | None = None,
    release: str = "",
    rated: dict | None = None,
) -> tuple[list[FixtureForecast], dict[int, np.ndarray]]:
    """The champion's forecasts of the league fixtures, made at `now`, and its pre-match matrices
    of the league matches played from `[live_chain] first_effects_season` to the season before
    `season` (the live chain's effects are fitted on them), each calibrated as the verdicts
    did. With `kept_in`, the past of the seasons before `season` is read from there when it is of
    this `release` and of today's inputs (`vesta.jobs.past_cache`), else computed and kept.
    With `rated`, it also receives the shots read (`shots`, with `last_season`) and their expected
    goals (`chances`), for the scorers."""
    chosen = [fixture for fixture in fixtures if fixture.kind == "league"]
    season = season or max((fixture.season for fixture in chosen), default=int(now[:4]))
    last_season = max([fixture.season for fixture in chosen] + [season])
    settings = WalkForwardSettings.from_config(spine)
    ratings_parameters = PlayerRatingsParameters.from_config(spine)
    matches = [match for match in read_league_matches(
        world, list(settings.competitions), settings.first_training_season - 1, last_season)
        if match.kickoff_utc < now]  # fmt: skip
    shot_section = expected_goals["shots"]
    on_dataset = is_dataset(world)  # our xG read from `shots.xg`, never fitted (S3a)
    if on_dataset:
        shots, given = read_rated_shots(world, shot_section["competitions"],
                                        shot_section["first_season"], last_season)  # fmt: skip
    else:
        shots = read_shots(world, shot_section["competitions"], shot_section["first_season"],
                           last_season, shot_section["layers"])  # fmt: skip
    players = read_match_players(world, {match.match_id for match in matches})
    model, shot_settings = (ModelSettings.from_config(expected_goals),
                            ShotWalkForwardSettings.from_config(expected_goals))  # fmt: skip
    closed_shots = shots.season < season
    inputs = hash_inputs([match for match in matches if match.season < season], players,
                         shots.select(closed_shots))  # fmt: skip
    kept = read_kept(kept_in, release, season, inputs)
    if on_dataset:
        chances = given
    elif kept is None:
        chances, _ = rate_every_season(shots, model, shot_settings)
    else:
        chances = np.full(len(shots), np.nan)
        chances[closed_shots] = kept.chances
        chances[~closed_shots] = rate_current_season(shots, model, shot_settings, season)
    if rated is not None:
        rated.update(shots=shots, chances=chances, last_season=last_season)
    by_match = sum_expected_goals_by_match(shots, chances)
    facts_of = {match.match_id: MatchFacts(*by_match.get(match.match_id, (math.nan, math.nan)),
                                           players.get(match.match_id, ()))
                for match in matches}  # fmt: skip
    time_rules = TimeRules.from_config(spine)
    facts, requests = _build_events(matches, time_rules, settings, facts_of)
    facts = [fact for fact in facts if fact.known_at <= now]
    if kept is not None:  # the past's forecasts are kept: ask only for this season's matches
        requests = [request for request in requests if request.season >= season]
    requests += [ForecastRequest(fixture.match_id, fixture.competition, fixture.season,
                                 fixture.home_team_id, fixture.away_team_id, now)
                 for fixture in chosen]  # fmt: skip
    families = [ExpectedGoalsDixonColesFamily(
        DixonColesParameters.from_config(spine, "dixon_coles_expected_goals")),
        PlayerRatingsFamily(ratings_parameters, settings.competitions)]  # fmt: skip
    rows = replay(families, facts, requests,
                  parts_for=frozenset(fixture.match_id for fixture in chosen))  # fmt: skip
    weight = ratings_parameters.blend_weight

    def blend_intensities(features):
        return (blend(features["dixon_coles_expected_goals_home_intensity"],
                      features["player_ratings_home_intensity"], weight),
                blend(features["dixon_coles_expected_goals_away_intensity"],
                      features["player_ratings_away_intensity"], weight))  # fmt: skip

    fixture_ids = {fixture.match_id: fixture for fixture in chosen}
    past = _keep_forecastable([row for row in rows if row.request.match_id not in fixture_ids],
                              matches, blend_intensities)  # fmt: skip
    by_id = {match.match_id: match for match in matches}
    if kept is None:
        write_kept(kept_in, KeptPast(release, season, inputs, chances[closed_shots], tuple(
            (item.match.match_id, item.forecast_time, item.home_intensity, item.away_intensity)
            for item in past if item.match.season < season)))  # fmt: skip
    else:
        past = [ForecastableMatch(by_id[match_id], at, home, away)
                for match_id, at, home, away in kept.forecasts] + past  # fmt: skip
    calibration_seasons = int(spine["champion_calibration"]["seasons"])
    calibrations = {}

    def fit_calibration(target: int):
        if target not in calibrations:
            first = max(target - calibration_seasons, settings.first_training_season)
            calibrations[target] = _fit_calibration_or_neutral(
                [item for item in past if first <= item.match.season < target],
                minimum_matches=1, fit_spread=True, fit_dispersion=True)  # fmt: skip
        return calibrations[target]

    first_effects = int(spine["live_chain"]["first_effects_season"])
    past_matrices = {}
    # the current season's played matches too: the player thresholds read a side's recent
    # expected goals (`vesta.players.thresholds.MatchTerms.attack`)
    for target in range(first_effects, season + 1):
        items = [item for item in past if item.match.season == target]
        if items:
            matrices = score_matrices(np.array([item.home_intensity for item in items]),
                                      np.array([item.away_intensity for item in items]),
                                      fit_calibration(target))  # fmt: skip
            past_matrices.update(zip((item.match.match_id for item in items), matrices,
                                     strict=True))  # fmt: skip
    forecasts = []
    for row in rows:
        fixture = fixture_ids.get(row.request.match_id)
        if fixture is None:
            continue
        home, away = blend_intensities(row.features)
        if not (math.isfinite(home) and math.isfinite(away)):
            continue
        calibration = fit_calibration(fixture.season)
        matrix = score_matrices(np.array([home]), np.array([away]), calibration)[0]
        team = (row.parts or {}).get("dixon_coles_expected_goals")
        ratings = (row.parts or {}).get("player_ratings")
        parts = ForecastParts(team, ratings, weight, calibration) if team else None
        forecasts.append(FixtureForecast(fixture, matrix, home, away, parts))
    return forecasts, past_matrices


def forecast_cup_fixtures(world: sqlite3.Connection, fixtures: list[Fixture], now: str,
                          spine: dict) -> list[FixtureForecast]:  # fmt: skip
    """The cups model's forecasts of the cup fixtures, made at `now`."""
    chosen = [fixture for fixture in fixtures if fixture.kind == "cup"]
    if not chosen:
        return []
    parameters = CupsParameters.from_config(spine)
    season = max(fixture.season for fixture in chosen)
    matches = [match for match in read_club_matches(
        world, list(parameters.leagues), list(parameters.cups), parameters.first_league_season,
        parameters.first_cup_season, season) if match.kickoff_utc < now]  # fmt: skip
    rated = [row for row in rate_matches(matches, parameters)
             if row.match.competition in parameters.cups]  # fmt: skip
    rating = make_rating_lookup(matches, parameters)
    forecasts = []
    for fixture in chosen:
        training = [row for row in rated
                    if parameters.first_cup_season <= row.match.season < fixture.season]
        mapping = fit_mapping(training)
        stage = ClubMatch(fixture.match_id, fixture.competition, "cup", fixture.season,
                          fixture.stage, fixture.kickoff_utc, fixture.home_team_id,
                          fixture.away_team_id, 0, 0, fixture.neutral, None)  # fmt: skip
        qualifying = 1.0 if stage.qualifying else 0.0
        parts = EloParts(rating(fixture.home_team_id), rating(fixture.away_team_id),
                         mapping.alpha + mapping.qualifying_level * qualifying,
                         mapping.beta + mapping.qualifying_slope * qualifying, mapping.gamma,
                         0.0 if fixture.neutral else 1.0, mapping.calibration)  # fmt: skip
        terms = parts.build_terms()
        matrix = parts.build_matrix(terms)
        home = math.exp(terms["level"] + terms["rating_difference"] + terms["home_ground"])
        away = math.exp(terms["level"] - terms["rating_difference"] - terms["home_ground"])
        forecasts.append(FixtureForecast(fixture, matrix, home, away, parts))
    return forecasts


def forecast_nation_fixtures(world: sqlite3.Connection, fixtures: list[Fixture], now: str,
                             spine: dict) -> list[FixtureForecast]:  # fmt: skip
    """The nations model's forecasts of the national fixtures, made at `now`."""
    chosen = [fixture for fixture in fixtures if fixture.kind == "nation"]
    if not chosen:
        return []
    parameters = NationsParameters.from_config(spine)
    matches = read_nation_matches(world)
    strength = build_nations_strength(matches, parameters, now[:10])
    forecasts = []
    for fixture in chosen:
        matrix, home, away = strength(fixture.home_team_id, fixture.away_team_id, fixture.neutral)
        forecasts.append(FixtureForecast(fixture, matrix, home, away, None))
    return forecasts


def format_now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


