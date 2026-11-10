"""The spine for the national teams: World Football Elo, then a Poisson on its difference.

The reference walk-forward of 2012-2021: 0.8870 on 9,033 matches. For each side,
goals ~ Poisson with

    log intensity = alpha + beta * (own Elo - opponent's Elo) / 400 + gamma * venue

(venue +1 at home, 0 on neutral ground, -1 for the away side), fitted on the matches of the
`training_years` calendar years before the year forecast, then the spine's score matrix with its
calibration (low scores, draws, the two scales) fitted on the same matches. The ratings of a match
are those at the start of its day: every match of the days before, none of the same day.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from vesta.determinism import run_on_one_thread
from vesta.spine.calibration import fit_calibration
from vesta.spine.elo import ELO_SCALE, compute_rating_change
from vesta.spine.match_forecast import RESULTS, MatchForecast
from vesta.spine.score_matrix import Calibration, score_matrices, sum_result_probabilities

QUALIFIER_KINDS = ("nations qualifying", "nations league")


@dataclass(frozen=True)
class NationsParameters:
    """`[nations]` of `config/vesta/spine.toml`."""

    first_rating: float
    home_bonus: float
    world_cup_finals: tuple[str, ...]
    world_cup_finals_weight: float
    continental_finals: tuple[str, ...]
    continental_finals_weight: float
    qualifiers_and_leagues_weight: float
    friendlies: tuple[str, ...]
    friendlies_weight: float
    other_tournaments_weight: float
    training_years: int
    minimum_matches_played: int

    @classmethod
    def from_config(cls, spine: dict) -> "NationsParameters":
        section = spine["nations"]
        return cls(**{
            name: tuple(section[name]) if isinstance(section[name], list) else section[name]
            for name in cls.__dataclass_fields__
        })  # fmt: skip

    def weight(self, competition: str, competition_kind: str) -> float:
        """K of World Football Elo by the kind of match."""
        if competition in self.world_cup_finals:
            return self.world_cup_finals_weight
        if competition_kind in QUALIFIER_KINDS:
            return self.qualifiers_and_leagues_weight
        if competition in self.continental_finals:
            return self.continental_finals_weight
        if competition in self.friendlies:
            return self.friendlies_weight
        return self.other_tournaments_weight


@dataclass(frozen=True)
class NationMatch:
    match_id: int
    competition: str
    competition_kind: str
    local_date: str
    home_team_id: int
    away_team_id: int
    home_goals: int
    away_goals: int
    neutral: bool


@dataclass(frozen=True)
class RatedNationMatch:
    match: NationMatch
    home_rating: float
    away_rating: float
    home_played: int
    away_played: int

    @property
    def year(self) -> int:
        return int(self.match.local_date[:4])


def rate_matches(
    matches: list[NationMatch], parameters: NationsParameters
) -> list[RatedNationMatch]:
    """Every match with the two ratings and the matches played at the start of its day."""
    return _rate(matches, parameters)[0]


def find_current_ratings(
    matches: list[NationMatch], parameters: NationsParameters
) -> tuple[dict[int, float], dict[int, int]]:
    """Every team's rating and matches played after all of `matches`."""
    _, rating, played = _rate(matches, parameters)
    return rating, played


def _rate(matches, parameters):
    ordered = sorted(matches, key=lambda match: (match.local_date, match.match_id))
    rating: dict[int, float] = {}
    played: dict[int, int] = {}
    rated: list[RatedNationMatch] = []
    start = 0
    while start < len(ordered):
        end = start
        while end < len(ordered) and ordered[end].local_date == ordered[start].local_date:
            end += 1
        day = [
            RatedNationMatch(
                match,
                rating.get(match.home_team_id, parameters.first_rating),
                rating.get(match.away_team_id, parameters.first_rating),
                played.get(match.home_team_id, 0),
                played.get(match.away_team_id, 0),
            )
            for match in ordered[start:end]
        ]
        for row in day:  # every match of a day moves the ratings of its start
            match = row.match
            change = compute_rating_change(
                row.home_rating, row.away_rating, match.home_goals, match.away_goals,
                parameters.weight(match.competition, match.competition_kind),
                0.0 if match.neutral else parameters.home_bonus,
            )  # fmt: skip
            for team, sign in ((match.home_team_id, 1.0), (match.away_team_id, -1.0)):
                rating[team] = rating.get(team, parameters.first_rating) + sign * change
                played[team] = played.get(team, 0) + 1
        rated += day
        start = end
    return rated, rating, played


@dataclass(frozen=True)
class NationsMapping:
    alpha: float
    beta: float
    gamma: float
    calibration: Calibration

    def predict_intensities(self, rows: list[RatedNationMatch]) -> tuple[np.ndarray, np.ndarray]:
        difference, venue = _build_design(rows)
        home = np.exp(self.alpha + self.beta * difference + self.gamma * venue)
        away = np.exp(self.alpha - self.beta * difference - self.gamma * venue)
        return home, away


@run_on_one_thread
def fit_mapping(rows: list[RatedNationMatch]) -> NationsMapping:
    """The Poisson of both sides' goals on the Elo difference and the venue, then the matrix's
    calibration on the same matches."""
    difference, venue = _build_design(rows)
    home_goals = np.array([row.match.home_goals for row in rows], dtype="float64")
    away_goals = np.array([row.match.away_goals for row in rows], dtype="float64")
    differences = np.concatenate([difference, -difference])
    venues = np.concatenate([venue, -venue])
    goals = np.concatenate([home_goals, away_goals])

    def measure_negative_log_likelihood(theta: np.ndarray) -> float:
        linear = theta[0] + theta[1] * differences + theta[2] * venues
        return float((np.exp(linear) - goals * linear + gammaln(goals + 1)).sum())

    fitted = minimize(measure_negative_log_likelihood, np.array([0.2, 0.5, 0.1]), method="L-BFGS-B")
    alpha, beta, gamma = (float(value) for value in fitted.x)
    uncalibrated = NationsMapping(alpha, beta, gamma, Calibration())
    home, away = uncalibrated.predict_intensities(rows)
    calibration = fit_calibration(home, away, home_goals, away_goals, False, False)
    return NationsMapping(alpha, beta, gamma, calibration)


def forecast_nations(
    matches: list[NationMatch], parameters: NationsParameters, years: tuple[int, ...]
) -> list[MatchForecast]:
    """Each year forecast by the mapping fitted on the years before it, the matches between two
    teams with a reliable rating only."""
    rated = rate_matches(matches, parameters)
    minimum = parameters.minimum_matches_played
    reliable = [row for row in rated
                if row.home_played >= minimum and row.away_played >= minimum]  # fmt: skip
    forecasts = []
    for year in years:
        training = [row for row in reliable
                    if year - parameters.training_years <= row.year < year]  # fmt: skip
        forecast = [row for row in reliable if row.year == year]
        if not training or not forecast:
            continue
        mapping = fit_mapping(training)
        home, away = mapping.predict_intensities(forecast)
        probabilities = sum_result_probabilities(score_matrices(home, away, mapping.calibration))
        for index, row in enumerate(forecast):
            forecasts.append(MatchForecast(
                match_id=row.match.match_id, competition=row.match.competition, season=year,
                forecast_time=row.match.local_date, home_intensity=float(home[index]),
                away_intensity=float(away[index]),
                probabilities=tuple(float(p) for p in probabilities[index]),
                result=_classify_result(row.match), calibration=mapping.calibration,
            ))  # fmt: skip
    return forecasts


def _build_design(rows: list[RatedNationMatch]) -> tuple[np.ndarray, np.ndarray]:
    difference = np.array([(row.home_rating - row.away_rating) / ELO_SCALE for row in rows])
    venue = np.array([0.0 if row.match.neutral else 1.0 for row in rows])
    return difference, venue


def _classify_result(match: NationMatch) -> str:
    if match.home_goals > match.away_goals:
        return RESULTS[0]
    return RESULTS[1] if match.home_goals == match.away_goals else RESULTS[2]

