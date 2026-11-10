"""One match forecast of the spine's result, and its log loss: what the walk-forwards of the
leagues, the cups and the national teams return and the reproductions measure."""

import math
from dataclasses import dataclass

import numpy as np

from vesta.spine.score_matrix import Calibration, score_matrices

RESULTS = ("home", "draw", "away")


@dataclass(frozen=True)
class MatchForecast:
    match_id: int
    competition: str
    season: int
    forecast_time: str
    home_intensity: float
    away_intensity: float
    probabilities: tuple[float, float, float]  # home win, draw, away win
    result: str  # what happened on the pitch: home, draw or away
    calibration: Calibration = Calibration()  # the matrix's, so the matrix can be rebuilt

    @property
    def matrix(self) -> np.ndarray:
        """The score matrix the forecast was read from."""
        return score_matrices(
            np.array([self.home_intensity]), np.array([self.away_intensity]), self.calibration
        )[0]

    @property
    def log_loss(self) -> float:
        return -math.log(self.probabilities[RESULTS.index(self.result)])


def average_log_loss(forecasts: list[MatchForecast]) -> float:
    return math.fsum(forecast.log_loss for forecast in forecasts) / len(forecasts)
