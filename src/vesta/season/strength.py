"""The strength of any pair of national teams, for the matches a tournament has not yet drawn.

The spine's nations model (`vesta.spine.nations`) as it stands at a date: every team's World
Football Elo after the matches before that date, and the Poisson on the Elo difference and the
venue fitted on the reliable matches of the `training_years` calendar years before the date's
year, with its calibration; the ratings stay frozen through the simulated tournament. A host of
the tournament plays at home on neutral ground (its venue +1, its opponent's -1).
"""

import numpy as np

from vesta.season.knockout import Strength
from vesta.spine.elo import ELO_SCALE
from vesta.spine.nations import (
    NationMatch,
    NationsMapping,
    NationsParameters,
    find_current_ratings,
    fit_mapping,
    rate_matches,
)
from vesta.spine.score_matrix import score_matrices


def build_nations_strength(
    matches: list[NationMatch],
    parameters: NationsParameters,
    as_of: str,
    hosts: frozenset[int] = frozenset(),
) -> Strength:
    """The strength function of the nations model at `as_of` (an ISO date: the matches before it
    count)."""
    before = [match for match in matches if match.local_date < as_of]
    rating, _ = find_current_ratings(before, parameters)
    year = int(as_of[:4])
    minimum = parameters.minimum_matches_played
    training = [row for row in rate_matches(before, parameters)
                if year - parameters.training_years <= row.year < year
                and row.home_played >= minimum and row.away_played >= minimum]  # fmt: skip
    if not training:
        raise ValueError(f"no reliable nations match in the {parameters.training_years} years"
                         f" before {as_of}")  # fmt: skip
    mapping = fit_mapping(training)
    cache: dict[tuple[int, int, bool], tuple[np.ndarray, float, float]] = {}

    def forecast_match(home: int, away: int, neutral: bool) -> tuple[np.ndarray, float, float]:
        key = (home, away, neutral)
        if key not in cache:
            cache[key] = _forecast_from_ratings(
                mapping, rating, parameters, home, away, neutral, hosts
            )
        return cache[key]

    return forecast_match


def _forecast_from_ratings(mapping: NationsMapping, rating, parameters, home, away, neutral, hosts):
    difference = (rating.get(home, parameters.first_rating)
                  - rating.get(away, parameters.first_rating)) / ELO_SCALE  # fmt: skip
    venue = 0.0 if neutral else 1.0
    if neutral and home in hosts:
        venue = 1.0
    elif neutral and away in hosts:
        venue = -1.0
    home_rate = float(np.exp(mapping.alpha + mapping.beta * difference + mapping.gamma * venue))
    away_rate = float(np.exp(mapping.alpha - mapping.beta * difference - mapping.gamma * venue))
    matrix = score_matrices(np.array([home_rate]), np.array([away_rate]), mapping.calibration)[0]
    return matrix, home_rate, away_rate
