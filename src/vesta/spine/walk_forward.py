"""The spine forecast walk-forward: each season by a model that saw only the seasons before it.

Every match feeds the families at the instant its result became known; every match from
`first_training_season` on is forecast at its earliest kick-off. For a season forecast, each
competition's matrix is calibrated on its own matches of the `calibration_seasons` seasons before
(their forecasts, made before each match, are out of sample); a competition with fewer than
`minimum_calibration_matches` there keeps the neutral matrix. Nothing is fitted on the matches
of the season forecast.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from vesta.spine.calibration import fit_calibration
from vesta.spine.dixon_coles import (
    DixonColesFamily,
    DixonColesParameters,
    ExpectedGoalsDixonColesFamily,
)
from vesta.spine.match_forecast import MatchForecast
from vesta.spine.player_ratings import PlayerRatingsFamily, PlayerRatingsParameters, blend
from vesta.spine.point_in_time import Fact, FeatureRow, ForecastRequest, replay
from vesta.spine.score_matrix import Calibration, score_matrices, sum_result_probabilities
from vesta.spine.timing import TimeRules, find_result_known_at, forecast_time
from vesta.spine.world import LeagueMatch


@dataclass(frozen=True)
class WalkForwardSettings:
    """`[walk_forward]` of `config/vesta/spine.toml`."""

    first_training_season: int
    calibration_seasons: int
    minimum_calibration_matches: int
    competitions: tuple[str, ...]

    @classmethod
    def from_config(cls, spine: dict) -> "WalkForwardSettings":
        section = spine["walk_forward"]
        return cls(
            first_training_season=int(section["first_training_season"]),
            calibration_seasons=int(section["calibration_seasons"]),
            minimum_calibration_matches=int(section["minimum_calibration_matches"]),
            competitions=tuple(section["competitions"]),
        )


@dataclass(frozen=True)
class MatchFacts:
    """What the champion learns of a finished match besides its score."""

    home_expected_goals: float
    away_expected_goals: float
    players: tuple[tuple[int, int, float, bool, bool], ...]


@dataclass(frozen=True)
class ForecastableMatch:
    """A match with the intensities its family gave before it."""

    match: LeagueMatch
    forecast_time: str
    home_intensity: float
    away_intensity: float


def forecast_with_baseline(
    matches: list[LeagueMatch],
    time_rules: TimeRules,
    parameters: DixonColesParameters,
    settings: WalkForwardSettings,
    forecast_seasons: tuple[int, ...],
) -> list[MatchForecast]:
    """B0's forecast of every match of `forecast_seasons` that its family could forecast (one
    with too little history in its window has none): Dixon-Coles on goals, each competition's
    matrix calibrated on its own two seasons before."""
    known = [match for match in matches if match.season <= max(forecast_seasons)]
    rows = replay([DixonColesFamily(parameters)], *_build_events(known, time_rules, settings))
    forecastable = _keep_forecastable(
        rows,
        known,
        lambda features: (
            features["dixon_coles_home_intensity"], features["dixon_coles_away_intensity"]
        ),
    )
    forecasts = []
    for season in forecast_seasons:
        for competition in settings.competitions:
            first = max(season - settings.calibration_seasons, settings.first_training_season)
            calibration = _fit_calibration_or_neutral(
                [row for row in forecastable if row.match.competition == competition
                 and first <= row.match.season < season],
                minimum_matches=settings.minimum_calibration_matches,
                fit_spread=False,
                fit_dispersion=False,
            )  # fmt: skip
            forecasts += _build_forecasts(
                [row for row in forecastable
                 if row.match.season == season and row.match.competition == competition],
                calibration,
            )  # fmt: skip
    return forecasts


def forecast_with_champion(
    matches: list[LeagueMatch],
    match_facts: dict[int, MatchFacts],
    time_rules: TimeRules,
    team_parameters: DixonColesParameters,
    ratings_parameters: PlayerRatingsParameters,
    calibration_seasons: int,
    settings: WalkForwardSettings,
    forecast_seasons: tuple[int, ...],
    ratings: PlayerRatingsFamily | None = None,
    earlier: timedelta = timedelta(0),
) -> list[MatchForecast]:
    """The champion's forecast (the previous `dc_xg2_cal_pl_pool_com`): Dixon-Coles on our
    expected goals mixed with goals and a prior for promoted clubs, blended with the ratings of
    the last eleven, through one matrix calibration for every league together on the
    `calibration_seasons` seasons before, with the spread and the dispersion. A candidate may
    pass its own `ratings` family, and every forecast may be made `earlier` than kick-off."""
    known = [match for match in matches if match.season <= max(forecast_seasons)]
    families = [
        ExpectedGoalsDixonColesFamily(team_parameters),
        ratings or PlayerRatingsFamily(ratings_parameters, settings.competitions),
    ]
    rows = replay(families, *_build_events(known, time_rules, settings, match_facts, earlier))
    weight = ratings_parameters.blend_weight
    forecastable = _keep_forecastable(
        rows,
        known,
        lambda features: (
            blend(features["dixon_coles_expected_goals_home_intensity"],
                  features["player_ratings_home_intensity"], weight),
            blend(features["dixon_coles_expected_goals_away_intensity"],
                  features["player_ratings_away_intensity"], weight),
        ),
    )  # fmt: skip
    forecasts = []
    for season in forecast_seasons:
        first = max(season - calibration_seasons, settings.first_training_season)
        calibration = _fit_calibration_or_neutral(
            [row for row in forecastable if first <= row.match.season < season],
            minimum_matches=1,
            fit_spread=True,
            fit_dispersion=True,
        )
        forecasts += _build_forecasts(
            [row for row in forecastable if row.match.season == season], calibration
        )
    return forecasts


def _build_events(
    known: list[LeagueMatch],
    time_rules: TimeRules,
    settings: WalkForwardSettings,
    match_facts: dict[int, MatchFacts] | None = None,
    earlier: timedelta = timedelta(0),
) -> tuple[list[Fact], list[ForecastRequest]]:
    """Every match as a fact at the instant its result was known, and every match from the
    first training season as a request at its earliest kick-off (less `earlier`)."""
    facts = []
    for match in known:
        extra = (match_facts or {}).get(match.match_id)
        facts.append(
            Fact(
                match.match_id, match.competition, match.season, match.home_team_id,
                match.away_team_id, match.home_goals, match.away_goals,
                find_result_known_at(match.kickoff_utc, match.kickoff_precision, match.local_date,
                                     time_rules),
                *((extra.home_expected_goals, extra.away_expected_goals, extra.players)
                  if extra else ()),
            )
        )  # fmt: skip
    requests = [
        ForecastRequest(match.match_id, match.competition, match.season, match.home_team_id,
                        match.away_team_id,
                        _subtract_delay(forecast_time(match.kickoff_utc, match.kickoff_precision,
                                                      match.local_date, time_rules), earlier))
        for match in known
        if match.season >= settings.first_training_season
    ]
    return facts, requests


def _subtract_delay(instant: str, earlier: timedelta) -> str:
    if not earlier:
        return instant
    moment = datetime.strptime(instant, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return (moment - earlier).strftime("%Y-%m-%dT%H:%M:%SZ")


def _keep_forecastable(
    rows: list[FeatureRow],
    known: list[LeagueMatch],
    intensities: Callable[[dict[str, float]], tuple[float, float]],
) -> list[ForecastableMatch]:
    """The requests whose intensities exist (a family with too little history gives none)."""
    by_match = {match.match_id: match for match in known}
    forecastable = []
    for row in rows:
        home, away = intensities(row.features)
        if math.isfinite(home) and math.isfinite(away):
            forecastable.append(
                ForecastableMatch(by_match[row.request.match_id], row.request.forecast_time,
                                  home, away)  # fmt: skip
            )
    return forecastable


def _fit_calibration_or_neutral(
    rows: list[ForecastableMatch], minimum_matches: int, fit_spread: bool, fit_dispersion: bool
) -> Calibration:
    """The matrix calibration fitted on `rows`; the neutral one under `minimum_matches`."""
    if len(rows) < minimum_matches:
        return Calibration()
    return fit_calibration(
        np.array([row.home_intensity for row in rows]),
        np.array([row.away_intensity for row in rows]),
        np.array([row.match.home_goals for row in rows]),
        np.array([row.match.away_goals for row in rows]),
        fit_spread=fit_spread,
        fit_dispersion=fit_dispersion,
    )


def _build_forecasts(
    rows: list[ForecastableMatch], calibration: Calibration
) -> list[MatchForecast]:
    if not rows:
        return []
    matrices = score_matrices(
        np.array([row.home_intensity for row in rows]),
        np.array([row.away_intensity for row in rows]),
        calibration,
    )
    probabilities = sum_result_probabilities(matrices)
    return [
        MatchForecast(
            match_id=row.match.match_id,
            competition=row.match.competition,
            season=row.match.season,
            forecast_time=row.forecast_time,
            home_intensity=row.home_intensity,
            away_intensity=row.away_intensity,
            probabilities=tuple(float(p) for p in probabilities[index]),
            result=_classify_result(row.match),
            calibration=calibration,
        )
        for index, row in enumerate(rows)
    ]


def _classify_result(match: LeagueMatch) -> str:
    if match.home_goals > match.away_goals:
        return "home"
    return "draw" if match.home_goals == match.away_goals else "away"
