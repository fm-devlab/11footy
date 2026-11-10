"""Fitting the calibration of the score matrix, on matches the intensities never saw.

The loss is the mean log loss of the result plus the mean log loss of the exact score, so the
matrix is right both on who wins and on by how much. It is the only calibration of the spine:
nothing is recalibrated after the matrix.
"""

import numpy as np
from scipy.optimize import minimize

from vesta.determinism import run_on_one_thread
from vesta.spine.score_matrix import (
    Calibration,
    find_score_probability,
    score_matrices,
    sum_result_probabilities,
)

# Bounds wide enough to be inert on real data, tight enough to keep every tau positive for
# intensities up to about five goals.
LOW_SCORE_CORRELATION_BOUNDS = (-0.15, 0.15)
DRAW_FACTOR_BOUNDS = (0.0, 0.5)
LOG_SCALE_BOUNDS = (-0.4, 0.4)
SPREAD_BOUNDS = (-0.5, 1.0)
DISPERSION_BOUNDS = (0.7, 1.5)
LOG_FLOOR = 1e-15


@run_on_one_thread
def fit_calibration(
    home_intensity: np.ndarray,
    away_intensity: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    fit_spread: bool,
    fit_dispersion: bool,
) -> Calibration:
    """The calibration that minimises the loss on these matches, by bounded quasi-Newton from the
    neutral point. The intensities must be out of sample: each computed without its match."""
    home_intensity = np.asarray(home_intensity, dtype="float64")
    away_intensity = np.asarray(away_intensity, dtype="float64")
    home_goals, away_goals = np.asarray(home_goals), np.asarray(away_goals)
    result_index = np.where(home_goals > away_goals, 0, np.where(home_goals == away_goals, 1, 2))
    start = [0.0, 0.0, 0.0, 0.0] + ([0.0] if fit_spread else []) + ([1.0] if fit_dispersion else [])
    bounds = [LOW_SCORE_CORRELATION_BOUNDS, DRAW_FACTOR_BOUNDS, LOG_SCALE_BOUNDS, LOG_SCALE_BOUNDS]
    bounds += [SPREAD_BOUNDS] if fit_spread else []
    bounds += [DISPERSION_BOUNDS] if fit_dispersion else []

    def measure_loss(parameters: np.ndarray) -> float:
        calibration = _unpack_calibration(parameters, fit_spread, fit_dispersion)
        matrices = score_matrices(home_intensity, away_intensity, calibration)
        result = sum_result_probabilities(matrices)[np.arange(len(result_index)), result_index]
        score = find_score_probability(matrices, home_goals, away_goals)
        return float(
            -np.log(np.clip(result, LOG_FLOOR, 1.0)).mean()
            - np.log(np.clip(score, LOG_FLOOR, 1.0)).mean()
        )

    fitted = minimize(measure_loss, x0=np.array(start), method="L-BFGS-B", bounds=bounds)
    return _unpack_calibration(fitted.x, fit_spread, fit_dispersion)


def _unpack_calibration(
    parameters: np.ndarray, fit_spread: bool, fit_dispersion: bool
) -> Calibration:
    correlation, draw_factor, log_home_scale, log_away_scale, *rest = (float(p) for p in parameters)
    spread = rest.pop(0) if fit_spread else 0.0
    dispersion = rest.pop(0) if fit_dispersion else 1.0
    return Calibration(
        low_score_correlation=correlation,
        draw_factor=draw_factor,
        home_scale=float(np.exp(log_home_scale)),
        away_scale=float(np.exp(log_away_scale)),
        spread=spread,
        dispersion=dispersion,
    )
