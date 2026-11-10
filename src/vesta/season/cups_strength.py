"""The strength of any pair of clubs in a UEFA cup, for the matches a cup has not yet played.

- **The model:** the spine's cups model (`vesta.spine.cups`) as it stands at an instant: every
  club's Elo after the league and cup matches that kicked off before it, the Poisson on the Elo
  difference, the venue and the qualifying flag fitted on the cup matches of the seasons before
  the one forecast (as the spine's walk-forward), with its calibration. The ratings stay frozen
  through the simulated season, as the nations' favourites (`vesta.season.strength`).
- **The reference, equal strength:** every pair given the mean score of the UEFA cups' matches of
  the seasons before (main stage, or qualifying rounds): the matrix of the scores' shares at home,
  its mirror averaged on neutral ground, the mean goals of each side for extra time. The three
  cups are pooled, so a cup's first season (the Conference League's 2021-22) has its reference.
"""

from dataclasses import dataclass

import numpy as np

from vesta.season.knockout import Strength
from vesta.spine.cups import (
    ClubMatch,
    CupsMapping,
    CupsParameters,
    fit_mapping,
    make_rating_lookup,
    rate_matches,
)
from vesta.spine.elo import ELO_SCALE
from vesta.spine.score_matrix import SIZE, score_matrices

MAIN_CUPS = ("CL", "EL", "ECL")
QUALIFYING_CUPS = ("CL", "EL", "ECL", "CLQ", "ELQ", "ECLQ")


def fit_cups_mapping(
    matches: list[ClubMatch], parameters: CupsParameters, season: int
) -> CupsMapping:
    """The cups model's Poisson fitted on the cup matches of the seasons before `season`."""
    earlier = [match for match in matches if match.season < season]
    training = [row for row in rate_matches(earlier, parameters)
                if row.match.competition in parameters.cups
                and parameters.first_cup_season <= row.match.season]  # fmt: skip
    if not training:
        raise ValueError(f"no cup match before {season} to fit the cups model on")
    return fit_mapping(training)


def build_cups_strength(matches: list[ClubMatch], parameters: CupsParameters, mapping: CupsMapping,
                        instant: str, qualifying: bool = False) -> Strength:  # fmt: skip
    """The strength of the cups model at `instant` (the matches that kicked off before it count),
    for main-stage matches or, with `qualifying`, for qualifying rounds."""
    rating = make_rating_lookup([match for match in matches if match.kickoff_utc < instant],
                                parameters)  # fmt: skip
    flag = 1.0 if qualifying else 0.0
    slope = mapping.beta + mapping.qualifying_slope * flag
    common = mapping.alpha + mapping.qualifying_level * flag
    cache: dict[tuple[int, int, bool], tuple[np.ndarray, float, float]] = {}

    def forecast_match(home: int, away: int, neutral: bool) -> tuple[np.ndarray, float, float]:
        key = (home, away, neutral)
        if key not in cache:
            difference = (rating(home) - rating(away)) / ELO_SCALE
            venue = 0.0 if neutral else 1.0
            home_rate = float(np.exp(common + slope * difference + mapping.gamma * venue))
            away_rate = float(np.exp(common - slope * difference - mapping.gamma * venue))
            matrix = score_matrices(np.array([home_rate]), np.array([away_rate]),
                                    mapping.calibration)[0]  # fmt: skip
            cache[key] = (matrix, home_rate, away_rate)
        return cache[key]

    return forecast_match


@dataclass(frozen=True)
class MeanScore:
    """The scores' shares of the cups' matches at a home ground, (11, 11), and each side's mean
    goals."""

    matrix: np.ndarray
    home_goals: float
    away_goals: float
    matches: int


def compute_mean_score(
    matches: list[ClubMatch], season: int, qualifying: bool = False
) -> MeanScore:
    """The mean score of the UEFA cups' matches not on neutral ground in the seasons before
    `season`: their main stage, or with `qualifying` their qualifying rounds."""
    cups = QUALIFYING_CUPS if qualifying else MAIN_CUPS
    chosen = [match for match in matches
              if match.competition in cups and match.season < season and not match.neutral
              and match.qualifying == qualifying]  # fmt: skip
    if not chosen:
        raise ValueError(f"no cup match before {season} for the mean score")
    counts = np.zeros((SIZE, SIZE))
    for match in chosen:
        counts[min(match.home_goals, SIZE - 1), min(match.away_goals, SIZE - 1)] += 1
    return MeanScore(counts / counts.sum(),
                     float(np.mean([match.home_goals for match in chosen])),
                     float(np.mean([match.away_goals for match in chosen])),
                     len(chosen))  # fmt: skip


def build_equal_strength(mean: MeanScore) -> Strength:
    """Every pair the same: the mean score at home, its mirror averaged on neutral ground."""
    neutral_matrix = (mean.matrix + mean.matrix.T) / 2
    neutral_goals = (mean.home_goals + mean.away_goals) / 2

    def forecast_match(_home: int, _away: int, neutral: bool) -> tuple[np.ndarray, float, float]:
        if neutral:
            return neutral_matrix, neutral_goals, neutral_goals
        return mean.matrix, mean.home_goals, mean.away_goals

    return forecast_match
