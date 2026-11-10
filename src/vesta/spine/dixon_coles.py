"""The Dixon-Coles model of goals, refitted every Monday on the results known (baseline B0).

The goal part of Dixon-Coles:

    log home intensity = level + home advantage + attack[home] - defence[away]
    log away intensity = level + attack[away] - defence[home]

fitted by weighted Poisson maximum likelihood with an L2 penalty on attack and defence, each
result weighted by exp(-decay * its age in days). The low-score correlation and the draw factor
are not estimated here: the score matrix and its calibration come after, from these intensities.

Point in time: a forecast in the week that starts on Monday 00:00 UTC uses the model fitted on
the results known at that Monday, never one of the week itself. The fit starts from zero, so it
is a pure function of the results and the week.
"""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from scipy.optimize import minimize

from vesta.determinism import run_on_one_thread
from vesta.spine.point_in_time import Fact, Family, ForecastRequest
from vesta.spine.timing import INSTANT_FORMAT

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class DixonColesParameters:
    """A section of `config/vesta/spine.toml`: `[dixon_coles]` for the baseline B0 (goals only,
    an unknown club as the average one), `[dixon_coles_expected_goals]` for the champion."""

    time_decay_per_day: float
    ridge: float
    window_days: int
    minimum_results: int
    # the target is weight * our expected goals + (1 - weight) * goals (0: goals only)
    expected_goals_weight: float = 0.0
    # attack and defence of a club the window has not seen (a promoted club)
    unknown_club_attack: float = 0.0
    unknown_club_defence: float = 0.0

    @classmethod
    def from_config(cls, spine: dict, section_name: str = "dixon_coles") -> "DixonColesParameters":
        section = spine[section_name]
        return cls(
            time_decay_per_day=float(section["time_decay_per_day"]),
            ridge=float(section["ridge"]),
            window_days=int(section["window_days"]),
            minimum_results=int(section["minimum_results"]),
            expected_goals_weight=float(section.get("expected_goals_weight", 0.0)),
            unknown_club_attack=float(section.get("unknown_club_attack", 0.0)),
            unknown_club_defence=float(section.get("unknown_club_defence", 0.0)),
        )

    def blend_target(self, goals: int, expected_goals: float) -> float:
        """What the fit learns from one side of a result: the goals, mixed with our expected
        goals when the weight is above zero and the match has them."""
        if self.expected_goals_weight == 0.0 or not np.isfinite(expected_goals):
            return float(goals)
        weight = self.expected_goals_weight
        return weight * expected_goals + (1.0 - weight) * goals


@dataclass(frozen=True)
class GoalsFit:
    teams: dict[int, int]  # team id -> index in attack and defence
    attack: np.ndarray
    defence: np.ndarray
    home_advantage: float
    level: float
    results: int


@run_on_one_thread
def fit_goals(
    home_teams: np.ndarray,
    away_teams: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> GoalsFit:
    """Weighted Poisson maximum likelihood of attack, defence, home advantage and level."""
    teams = {
        team: index
        for index, team in enumerate(sorted(set(home_teams.tolist()) | set(away_teams.tolist())))
    }
    count = len(teams)
    home_index = np.array([teams[team] for team in home_teams], dtype="int64")
    away_index = np.array([teams[team] for team in away_teams], dtype="int64")
    home_goals = np.asarray(home_goals, dtype="float64")
    away_goals = np.asarray(away_goals, dtype="float64")

    def measure_loss_and_gradient(theta: np.ndarray) -> tuple[float, np.ndarray]:
        attack, defence = theta[:count], theta[count : 2 * count]
        home_advantage, level = theta[2 * count], theta[2 * count + 1]
        home_log = level + home_advantage + attack[home_index] - defence[away_index]
        away_log = level + attack[away_index] - defence[home_index]
        home_intensity, away_intensity = np.exp(home_log), np.exp(away_log)
        loss = float(
            np.sum(weights * (home_intensity - home_goals * home_log))
            + np.sum(weights * (away_intensity - away_goals * away_log))
            + ridge * (attack @ attack + defence @ defence)
        )
        home_residual = weights * (home_intensity - home_goals)
        away_residual = weights * (away_intensity - away_goals)
        gradient = np.empty_like(theta)
        gradient[:count] = (
            np.bincount(home_index, home_residual, count)
            + np.bincount(away_index, away_residual, count)
            + 2 * ridge * attack
        )
        gradient[count : 2 * count] = (
            -np.bincount(away_index, home_residual, count)
            - np.bincount(home_index, away_residual, count)
            + 2 * ridge * defence
        )
        gradient[2 * count] = home_residual.sum()
        gradient[2 * count + 1] = home_residual.sum() + away_residual.sum()
        return loss, gradient

    start = np.zeros(2 * count + 2)
    mean_goals = float((home_goals @ weights + away_goals @ weights) / (2 * weights.sum()))
    start[2 * count + 1] = np.log(max(mean_goals, 0.1))
    result = minimize(
        measure_loss_and_gradient,
        start,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    theta = result.x
    return GoalsFit(
        teams=teams,
        attack=theta[:count],
        defence=theta[count : 2 * count],
        home_advantage=float(theta[2 * count]),
        level=float(theta[2 * count + 1]),
        results=len(home_teams),
    )


class GoalsState:
    """The results a stream has learned, in the order they became known."""

    def __init__(self) -> None:
        self.home_teams: list[int] = []
        self.away_teams: list[int] = []
        self.home_targets: list[float] = []
        self.away_targets: list[float] = []
        self.known_at_seconds: list[int] = []  # never decreasing: facts arrive in time order
        self.fitted_week: tuple[int, GoalsFit | None] | None = None  # derived, not state


class DixonColesFamily(Family):
    """The intensities of the two sides from the attack and defence of the clubs."""

    name = "dixon_coles"
    columns = ("dixon_coles_home_intensity", "dixon_coles_away_intensity")

    def __init__(self, parameters: DixonColesParameters) -> None:
        self.parameters = parameters

    def make_empty_state(self) -> GoalsState:
        return GoalsState()

    def learn(self, state: GoalsState, fact: Fact) -> None:
        state.home_teams.append(fact.home_team_id)
        state.away_teams.append(fact.away_team_id)
        state.home_targets.append(
            self.parameters.blend_target(fact.home_goals, fact.home_expected_goals)
        )
        state.away_targets.append(
            self.parameters.blend_target(fact.away_goals, fact.away_expected_goals)
        )
        state.known_at_seconds.append(parse_instant_to_seconds(fact.known_at))

    def build_features(self, state: GoalsState, request: ForecastRequest) -> dict[str, float]:
        parts = self.build_log_terms(state, request)
        if parts is None:
            return dict.fromkeys(self.columns, float("nan"))
        home_column, away_column = self.columns
        return {
            home_column: float(np.exp(parts["level"] + parts["home_advantage"]
                                      + parts["home_attack"] - parts["away_defence"])),
            away_column: float(np.exp(parts["level"] + parts["away_attack"]
                                      - parts["home_defence"])),
        }  # fmt: skip

    def build_log_terms(self, state: GoalsState, request: ForecastRequest) -> dict | None:
        """The log terms the intensities add up from (the explanation reads them): the level,
        the home advantage, each club's attack and defence; `*_known` False for a club the fit
        does not hold, given the prior of an unknown club."""
        fit = self._fit_of_week(
            state, find_week_start(parse_instant_to_seconds(request.forecast_time))
        )
        if fit is None:
            return None
        home, away = fit.teams.get(request.home_team_id), fit.teams.get(request.away_team_id)
        unknown_attack = self.parameters.unknown_club_attack
        unknown_defence = self.parameters.unknown_club_defence
        return {
            "level": float(fit.level),
            "home_advantage": float(fit.home_advantage),
            "home_attack": float(fit.attack[home]) if home is not None else unknown_attack,
            "home_defence": float(fit.defence[home]) if home is not None else unknown_defence,
            "away_attack": float(fit.attack[away]) if away is not None else unknown_attack,
            "away_defence": float(fit.defence[away]) if away is not None else unknown_defence,
            "home_known": home is not None,
            "away_known": away is not None,
        }

    def _fit_of_week(self, state: GoalsState, week_seconds: int) -> GoalsFit | None:
        """The model of the week starting at `week_seconds`: the results known at that instant."""
        if state.fitted_week is not None and state.fitted_week[0] == week_seconds:
            return state.fitted_week[1]
        window = self.parameters.window_days * SECONDS_PER_DAY
        last = bisect_right(state.known_at_seconds, week_seconds)
        first = bisect_left(state.known_at_seconds, week_seconds - window)
        fit = None
        if last - first >= self.parameters.minimum_results:
            known = np.array(state.known_at_seconds[first:last], dtype="int64")
            age_days = (week_seconds - known) / SECONDS_PER_DAY
            fit = fit_goals(
                np.array(state.home_teams[first:last], dtype="int64"),
                np.array(state.away_teams[first:last], dtype="int64"),
                np.array(state.home_targets[first:last], dtype="float64"),
                np.array(state.away_targets[first:last], dtype="float64"),
                np.exp(-self.parameters.time_decay_per_day * age_days),
                self.parameters.ridge,
            )
        state.fitted_week = (week_seconds, fit)
        return fit



class ExpectedGoalsDixonColesFamily(DixonColesFamily):
    """The champion's Dixon-Coles: fitted on our expected goals mixed with goals, a promoted
    club given the attack and defence of the prior."""

    name = "dixon_coles_expected_goals"
    columns = (
        "dixon_coles_expected_goals_home_intensity",
        "dixon_coles_expected_goals_away_intensity",
    )

def parse_instant_to_seconds(instant: str) -> int:
    """An ISO 8601 UTC instant as whole seconds since 1970-01-01."""
    return int(datetime.strptime(instant, INSTANT_FORMAT).replace(tzinfo=UTC).timestamp())


def find_week_start(seconds: int) -> int:
    """Monday 00:00 UTC of the ISO week holding `seconds`."""
    moment = datetime.fromtimestamp(seconds, tz=UTC)
    monday = (moment - timedelta(days=moment.weekday())).replace(hour=0, minute=0, second=0)
    return int(monday.timestamp())
