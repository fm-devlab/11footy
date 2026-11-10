"""Ties (M14, step 3b): extra time, the shoot-out and who goes through, from the score matrix.

A knockout match is a single match (a final, a round of a national tournament) or one leg of two.
Everything is computed exactly from the matrices, with no simulation:
- the score after 90 minutes is the match's matrix;
- extra time, when the tie is level, is two independent Poissons over `extra_time_minutes`, each
  side's intensity the match's calibrated one times the minutes' share and `extra_time_pace`;
- a shoot-out, when still level, goes to the home side with `shootout_home_share`;
- a two-legged tie adds the first leg's goals to the second's, and where the competition's rules
  of that season say so (`config/vesta/tie-rules.csv`, never a constant: UEFA abolished the rule
  from 2021-22), the away goals decide a level aggregate, in extra time too when they say so.
A forecast of a first leg reads who goes through from the matrix of the second leg as well.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import poisson

from vesta.spine.score_matrix import SIZE

TIE_RULES_FILE = Path(__file__).resolve().parents[3] / "config" / "vesta" / "tie-rules.csv"
REGULAR_MINUTES = 90.0
WAYS = ("in_regular_time", "in_extra_time", "on_penalties")


@dataclass(frozen=True)
class TieParameters:
    """`[ties]` of `config/vesta/spine.toml`."""

    extra_time_minutes: float
    extra_time_pace: float
    shootout_home_share: float

    @classmethod
    def from_config(cls, spine: dict) -> "TieParameters":
        return cls(**{name: float(spine["ties"][name]) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class TieRule:
    away_goals: bool
    away_goals_in_extra_time: bool


def read_tie_rule(competition: str, season: int, path: Path = TIE_RULES_FILE) -> TieRule:
    """The rules of a two-legged tie of `competition` in `season`; refused when the table has
    none, so no rule is ever assumed."""
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if (row["competition"] == competition
                    and int(row["first_season"]) <= season <= int(row["last_season"])):  # fmt: skip
                return TieRule(away_goals=row["away_goals"] == "yes",
                               away_goals_in_extra_time=row["away_goals_in_extra_time"] == "yes")
    raise ValueError(
        f"no rule of two-legged ties for {competition} in {season}: write it in {path.name}"
    )


@dataclass(frozen=True)
class Outcomes:
    """Per side of the match forecast (home, away): the chance of going through in regular time,
    in extra time, on penalties; the chance of extra time, of a shoot-out; and the goals of extra
    time (a count, 0 when none is played)."""

    through: dict[str, dict[str, float]]
    extra_time: float
    shootout: float
    extra_time_goals: np.ndarray


def resolve_single_match(
    matrix: np.ndarray, home_intensity: float, away_intensity: float, parameters: TieParameters
) -> Outcomes:
    """A knockout decided in one match: level after 90 minutes, extra time, then penalties."""
    def measure_regular_margin(home: int, away: int) -> int:
        return home - away

    def measure_extra_margin(_home: int, _away: int, extra_home: int, extra_away: int) -> int:
        return extra_home - extra_away

    return _resolve(
        matrix,
        home_intensity,
        away_intensity,
        parameters,
        measure_regular_margin,
        measure_extra_margin,
    )


def resolve_second_leg(
    matrix: np.ndarray,
    home_intensity: float,
    away_intensity: float,
    first_leg: tuple[int, int],
    rule: TieRule,
    parameters: TieParameters,
) -> Outcomes:
    """The second leg of a tie whose first leg ended `first_leg` (goals of this match's away
    side, who were at home then, and of this match's home side)."""
    away_first, home_first = first_leg

    def measure_regular_margin(home, away):
        aggregate = (home_first + home) - (away_first + away)
        if aggregate != 0 or not rule.away_goals:
            return aggregate
        return home_first - away  # away goals: this match's home side scored away in the first

    def measure_extra_margin(_home, _away, extra_home, extra_away):
        aggregate = extra_home - extra_away
        if aggregate != 0 or not rule.away_goals_in_extra_time:
            return aggregate
        return -extra_away  # level in extra time: the visitors' extra-time goals are away goals

    return _resolve(
        matrix,
        home_intensity,
        away_intensity,
        parameters,
        measure_regular_margin,
        measure_extra_margin,
    )


def forecast_first_leg_through(
    first_matrix: np.ndarray,
    second_matrix: np.ndarray,
    second_intensities: tuple[float, float],
    rule: TieRule,
    parameters: TieParameters,
) -> dict[str, float]:
    """At the first leg, the chance each of its sides goes through: every first-leg score with the
    second leg's outcomes given it (the second leg's matrix, its home side the first leg's away)."""
    home_through = 0.0
    for first_home in range(SIZE):
        for first_away in range(SIZE):
            chance = first_matrix[first_home, first_away]
            if chance == 0.0:
                continue
            outcomes = resolve_second_leg(
                second_matrix, *second_intensities, (first_home, first_away), rule, parameters
            )
            home_through += chance * sum(outcomes.through["away"].values())
    return {"home": home_through, "away": 1.0 - home_through}




def _resolve(matrix, home_intensity, away_intensity, parameters, regular, extra) -> Outcomes:
    """Walk every score after 90 minutes, then every score of extra time where still level."""
    share = parameters.extra_time_minutes / REGULAR_MINUTES * parameters.extra_time_pace
    goals = np.arange(SIZE)
    extra_home = poisson.pmf(goals, home_intensity * share)
    extra_away = poisson.pmf(goals, away_intensity * share)
    extra_matrix = np.outer(extra_home / extra_home.sum(), extra_away / extra_away.sum())
    through = {side: dict.fromkeys(WAYS, 0.0) for side in ("home", "away")}
    extra_time_goals = np.zeros(2 * SIZE - 1)
    extra_time = shootout = 0.0
    for home in range(SIZE):
        for away in range(SIZE):
            chance = float(matrix[home, away])
            margin = regular(home, away)
            if margin != 0:
                through["home" if margin > 0 else "away"]["in_regular_time"] += chance
                extra_time_goals[0] += chance
                continue
            extra_time += chance
            for extra_h in range(SIZE):
                for extra_a in range(SIZE):
                    extra_chance = chance * extra_matrix[extra_h, extra_a]
                    extra_time_goals[extra_h + extra_a] += extra_chance
                    level = extra(home, away, extra_h, extra_a)
                    if level != 0:
                        through["home" if level > 0 else "away"]["in_extra_time"] += extra_chance
                        continue
                    shootout += extra_chance
                    through["home"]["on_penalties"] += extra_chance * parameters.shootout_home_share
                    through["away"]["on_penalties"] += (
                        extra_chance * (1.0 - parameters.shootout_home_share)
                    )
    return Outcomes(through, extra_time, shootout, extra_time_goals)
