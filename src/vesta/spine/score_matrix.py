"""The score matrix: the one place where two intensities become the probabilities of every score.

For a match with intensities (expected goals before correction) of the home and the away
side, the probability of the final score i-j, for i, j from 0 to `MAXIMUM_GOALS`, is

    P(i, j) proportional to tau(i, j) * f(i; home) * f(j; away) * (1 + draw_factor * [i = j])

where f is the goals of one side (a Poisson, raised to the power `dispersion` and renormalised:
above 1 fewer blanks and fewer blowouts than a Poisson) and tau the Dixon-Coles correction of the
four low scores. The Poisson's mass beyond `MAXIMUM_GOALS` is folded into the last cell of each
side, so the last row and column mean "ten or more" and the matrix sums to one. Every number of
M1-M12 is read from this matrix (`vesta.forecast_reading`), so the numbers of a match agree by
construction.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

MAXIMUM_GOALS = 10
SIZE = MAXIMUM_GOALS + 1
GOALS = np.arange(SIZE)
# tau stays positive: a correction that would make a cell negative is held at this floor
TAU_FLOOR = 1e-9


@dataclass(frozen=True)
class Calibration:
    """The numbers that correct the matrix; the default leaves a plain Poisson untouched.

    - `low_score_correlation` (Dixon-Coles rho) moves probability among 0-0, 1-0, 0-1 and 1-1;
    - `draw_factor` adds a share to every draw;
    - `home_scale`, `away_scale` multiply the two intensities;
    - `spread` stretches the gap between the two intensities around their geometric mean (a
      positive spread corrects strengths shrunk too much towards the mean);
    - `dispersion` is the power of each side's Poisson (Conway-Maxwell-Poisson); 1 is a Poisson.
    """

    low_score_correlation: float = 0.0
    draw_factor: float = 0.0
    home_scale: float = 1.0
    away_scale: float = 1.0
    spread: float = 0.0
    dispersion: float = 1.0


def calibrate_intensities(
    home_intensity: np.ndarray, away_intensity: np.ndarray, calibration: Calibration
) -> tuple[np.ndarray, np.ndarray]:
    """The intensities after the scales and the spread (the correlation and the draw factor act
    inside the matrix)."""
    if calibration.spread == 0.0:  # the plain product, to the bit, as in the reference
        return home_intensity * calibration.home_scale, away_intensity * calibration.away_scale
    log_home, log_away = np.log(home_intensity), np.log(away_intensity)
    half_gap = 0.5 * (log_home - log_away)
    return (
        np.exp(log_home + calibration.spread * half_gap) * calibration.home_scale,
        np.exp(log_away - calibration.spread * half_gap) * calibration.away_scale,
    )


def score_matrices(
    home_intensity: np.ndarray, away_intensity: np.ndarray, calibration: Calibration
) -> np.ndarray:
    """The matrices of n matches, shape (n, 11, 11): `[match, home goals, away goals]`."""
    home, away = calibrate_intensities(
        np.atleast_1d(np.asarray(home_intensity, dtype="float64")),
        np.atleast_1d(np.asarray(away_intensity, dtype="float64")),
        calibration,
    )
    rho = calibration.low_score_correlation
    matrix = (
        _build_goal_distribution(home, calibration.dispersion)[:, :, None]
        * _build_goal_distribution(away, calibration.dispersion)[:, None, :]
    )
    tau = np.ones_like(matrix)
    tau[:, 0, 0] = 1.0 - home * away * rho
    tau[:, 0, 1] = 1.0 + home * rho
    tau[:, 1, 0] = 1.0 + away * rho
    tau[:, 1, 1] = 1.0 - rho
    matrix = matrix * np.maximum(tau, TAU_FLOOR)
    matrix[:, GOALS, GOALS] *= 1.0 + calibration.draw_factor
    return matrix / matrix.sum(axis=(1, 2), keepdims=True)


def sum_result_probabilities(matrices: np.ndarray) -> np.ndarray:
    """(n, 3): home win, draw, away win, the sums below, on and above the diagonal."""
    home_win = np.tril(np.ones((SIZE, SIZE)), -1)
    away_win = np.triu(np.ones((SIZE, SIZE)), 1)
    draw = np.eye(SIZE)
    return np.stack(
        [(matrices * mask).sum(axis=(1, 2)) for mask in (home_win, draw, away_win)], axis=1
    )


def find_score_probability(
    matrices: np.ndarray, home_goals: np.ndarray, away_goals: np.ndarray
) -> np.ndarray:
    """The probability each match gave to its own score; ten goals or more fall in the last cell."""
    home = np.minimum(np.asarray(home_goals, dtype="int64"), MAXIMUM_GOALS)
    away = np.minimum(np.asarray(away_goals, dtype="int64"), MAXIMUM_GOALS)
    return matrices[np.arange(len(home)), home, away]


def format_published_matrix(matrix: np.ndarray) -> dict:
    """One match's matrix in the forecast log's `joint_count` shape (`docs/FORECAST_LOG.md` §6):
    the tail is folded into the last row and column, so the published tail is zero."""
    return {"matrix": matrix.tolist(), "tail": 0.0}


def _build_goal_distribution(intensity: np.ndarray, dispersion: float) -> np.ndarray:
    """(n, 11): the Poisson of each intensity with its tail folded into the last cell, raised to
    `dispersion` and renormalised when the dispersion is not 1."""
    log_probability = GOALS * np.log(intensity[:, None]) - intensity[:, None] - gammaln(GOALS + 1)
    probability = np.exp(log_probability)
    probability[:, -1] = np.maximum(1.0 - probability[:, :-1].sum(axis=1), 0.0)
    if dispersion == 1.0:
        return probability
    probability = probability**dispersion
    return probability / probability.sum(axis=1, keepdims=True)
