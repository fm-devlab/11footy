"""Every shot rated by a model that never saw its match.

From `first_forward_season` on, the model of season S is fitted on the seasons before S - 1, with
S - 1 choosing the number of trees and then joining the fit, and rates the shots of S. The seasons
before are rated by cross-fitting over the two first seasons: `folds` models, each fitted on the
shots of the other folds (a fold is a stable hash of the match), rate the shots of their own fold.
"""

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from vesta.determinism import run_on_one_thread
from vesta.expected_goals.model import FittedModel, ModelSettings, fit
from vesta.expected_goals.shots import ShotTable

LOG_FLOOR = 1e-6  # the check clips each chance to [1e-6, 1 - 1e-6], as the reference did


@dataclass(frozen=True)
class WalkForwardSettings:
    """`[walk_forward]` of `config/vesta/expected_goals.toml`."""

    first_forward_season: int
    folds: int

    @classmethod
    def from_config(cls, expected_goals: dict) -> "WalkForwardSettings":
        section = expected_goals["walk_forward"]
        return cls(int(section["first_forward_season"]), int(section["folds"]))


def assign_fold(match_id: int, folds: int) -> int:
    """The cross-fitting fold of a match: a stable hash of its id."""
    return int(hashlib.sha1(str(match_id).encode()).hexdigest(), 16) % folds


@run_on_one_thread
def rate_every_season(
    shots: ShotTable, model: ModelSettings, settings: WalkForwardSettings
) -> tuple[np.ndarray, dict[str, FittedModel]]:
    """The expected goals of every shot, and the models that gave them (by name)."""
    chances = np.full(len(shots), np.nan)
    models: dict[str, FittedModel] = {}
    first = settings.first_forward_season
    early = shots.season < first
    fold = np.array([assign_fold(int(match), settings.folds) for match in shots.match_id])
    for index in range(settings.folds):
        outside = early & (fold != index)
        fitted = fit(
            shots.select(outside & (shots.season == first - 2)),
            shots.select(outside & (shots.season == first - 1)),
            model,
        )
        models[f"fold {index}"] = fitted
        inside = early & (fold == index)
        chances[inside] = fitted.rate(shots.select(inside))
    for season in sorted({int(season) for season in shots.season if season >= first}):
        fitted = _fit_season_model(shots, model, season)
        models[f"season {season}"] = fitted
        chosen = shots.season == season
        chances[chosen] = fitted.rate(shots.select(chosen))
    return chances, models


def rate_current_season(
    shots: ShotTable, model: ModelSettings, settings: WalkForwardSettings, season: int
) -> np.ndarray:
    """The chances of the shots of `season` and after, as `rate_every_season` gives them (each
    season by the model fitted on the seasons before it), without rating the seasons before."""
    chances = np.full(int((shots.season >= season).sum()), np.nan)
    later = shots.select(shots.season >= season)
    if season < settings.first_forward_season:
        raise ValueError(f"season {season} is inside the folds: rate every season instead")
    for each in sorted({int(value) for value in later.season}):
        chosen = later.season == each
        chances[chosen] = _fit_season_model(shots, model, each).rate(later.select(chosen))
    return chances


def _fit_season_model(shots: ShotTable, model: ModelSettings, season: int) -> FittedModel:
    return fit(shots.select(shots.season < season - 1), shots.select(shots.season == season - 1),
               model)  # fmt: skip


@dataclass(frozen=True)
class Check:
    """Expected goals against goals, and the per-shot scores of the chances."""

    shots: int
    goals: int
    expected_goals: float
    log_loss: float
    brier: float


def check(chances: np.ndarray, goal: np.ndarray) -> Check:
    clipped = np.clip(chances, LOG_FLOOR, 1 - LOG_FLOOR)
    log_loss = -(goal * np.log(clipped) + (1 - goal) * np.log(1 - clipped)).mean()
    return Check(
        shots=len(goal),
        goals=int(goal.sum()),
        expected_goals=math.fsum(chances.tolist()),
        log_loss=float(log_loss),
        brier=float(((chances - goal) ** 2).mean()),
    )


def sum_expected_goals_by_match(
    shots: ShotTable, chances: np.ndarray
) -> dict[int, tuple[float, float]]:
    """Per match with shots, the expected goals of the home and the away side (penalties
    included): the sums of the chances of their shots, zero for a side that never shot."""
    totals: dict[int, list[float]] = {}
    for match_id, home, chance in zip(shots.match_id, shots.home, chances, strict=True):
        sides = totals.setdefault(int(match_id), [0.0, 0.0])
        sides[0 if home else 1] += float(chance)
    return {match_id: (sides[0], sides[1]) for match_id, sides in totals.items()}
