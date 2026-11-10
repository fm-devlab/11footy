"""The strength of a team from the players it fields.

Every player gets an attack and a defence rating, refitted every Monday on the results known then,
in the five leagues together, so a rating follows the player across clubs and leagues:

    log E[target of a side] = level[league] + home * [side is home]
                              + sum of the side's attack ratings weighted by minutes
                              - sum of the opponents' defence ratings weighted by minutes

where the target is weight * our expected goals + (1 - weight) * goals and the weights are each
player's share of his side's minutes. Weighted Poisson quasi-likelihood with the time decay and a
small L2 penalty (a player weighs about one eleventh of a team). A goalkeeper (a player who kept
goal in more than half his matches of the window) weighs `goalkeeper_attack_weight` in the attack
side, zero in the champion: his attack rating is only the level of his club.

For a match to forecast, each team is the mean rating of the eleven that started its last match
known before the forecast (never the sheet of the match itself; `eleven_of`, which a candidate
reading the announced sheet replaces, `vesta.spine.announced_eleven`).
"""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from vesta.determinism import run_on_one_thread
from vesta.spine.dixon_coles import SECONDS_PER_DAY, find_week_start, parse_instant_to_seconds
from vesta.spine.point_in_time import Fact, Family, ForecastRequest

SIDES = 2


@dataclass(frozen=True)
class PlayerRatingsParameters:
    """`[player_ratings]` of `config/vesta/spine.toml`."""

    time_decay_per_day: float
    ridge: float
    window_days: int
    minimum_results: int
    expected_goals_weight: float
    blend_weight: float
    goalkeeper_attack_weight: float

    @classmethod
    def from_config(cls, spine: dict) -> "PlayerRatingsParameters":
        section = spine["player_ratings"]
        return cls(**{name: section[name] for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class RatingsFit:
    level: np.ndarray  # per league, in the order of `leagues`
    home_advantage: float
    attack: np.ndarray
    defence: np.ndarray
    players: dict[int, int]  # player id -> index in attack and defence
    attack_weight: np.ndarray  # per player: 1, or the goalkeeper's weight


class RatingsState:
    def __init__(self) -> None:
        self.known_at_seconds: list[int] = []
        self.leagues: list[int] = []
        self.home_targets: list[float] = []
        self.away_targets: list[float] = []
        self.players: list[tuple[tuple[int, int, float, bool, bool], ...]] = []
        self.last_eleven: dict[int, tuple[int, ...]] = {}  # team -> starters of its last match
        self.fitted_week: tuple[int, RatingsFit | None] | None = None  # derived, not state


class PlayerRatingsFamily(Family):
    name = "player_ratings"
    scope = "all"
    columns = ("player_ratings_home_intensity", "player_ratings_away_intensity")

    def __init__(self, parameters: PlayerRatingsParameters, leagues: tuple[str, ...]) -> None:
        self.parameters = parameters
        self.leagues = tuple(sorted(leagues))

    def make_empty_state(self) -> RatingsState:
        return RatingsState()

    def learn(self, state: RatingsState, fact: Fact) -> None:
        players = tuple(row for row in fact.players if row[2] > 0)
        state.known_at_seconds.append(parse_instant_to_seconds(fact.known_at))
        state.leagues.append(self.leagues.index(fact.competition))
        state.home_targets.append(self._blend_target(fact.home_goals, fact.home_expected_goals))
        state.away_targets.append(self._blend_target(fact.away_goals, fact.away_expected_goals))
        state.players.append(players)
        for side, team in ((0, fact.home_team_id), (1, fact.away_team_id)):
            starters = tuple(
                player for s, player, _, started, _ in players if s == side and started
            )
            if starters:
                state.last_eleven[team] = starters

    def build_features(self, state: RatingsState, request: ForecastRequest) -> dict[str, float]:
        parts = self.build_log_terms(state, request)
        if parts is None:
            return dict.fromkeys(self.columns, float("nan"))
        return {
            "player_ratings_home_intensity": float(np.exp(
                parts["level"] + parts["home_advantage"] + parts["home_attack"]
                - parts["away_defence"])),
            "player_ratings_away_intensity": float(np.exp(
                parts["level"] + parts["away_attack"] - parts["home_defence"])),
        }  # fmt: skip

    def build_log_terms(self, state: RatingsState, request: ForecastRequest) -> dict | None:
        """The log terms the intensities add up from (the explanation reads them): the league's
        level, the home advantage, the attack and defence of each side's last eleven; beside
        them, what was read (the forecast log's `publication_inputs`): each side's eleven and
        the week of the fit, its Monday as seconds."""
        week = find_week_start(parse_instant_to_seconds(request.forecast_time))
        fit = self._fit_of_week(state, week)
        if fit is None:
            return None
        home_eleven = self.choose_eleven(state, request, request.home_team_id)
        away_eleven = self.choose_eleven(state, request, request.away_team_id)
        home = self._rate_eleven(fit, home_eleven)
        away = self._rate_eleven(fit, away_eleven)
        if home is None or away is None:
            return None
        return {
            "level": float(fit.level[self.leagues.index(request.competition)]),
            "home_advantage": float(fit.home_advantage),
            "home_attack": home[0], "home_defence": home[1],
            "away_attack": away[0], "away_defence": away[1],
            "home_eleven": home_eleven,
            "away_eleven": away_eleven,
            "fit_week": week,
        }  # fmt: skip

    def _blend_target(self, goals: int, expected_goals: float) -> float:
        weight = self.parameters.expected_goals_weight
        if not np.isfinite(expected_goals):
            return float(goals)
        return weight * expected_goals + (1 - weight) * goals

    def choose_eleven(self, state: RatingsState, _request: ForecastRequest, team: int) -> tuple:
        """The eleven a team is rated by: the starters of its last match known."""
        return state.last_eleven.get(team, ())

    def _rate_eleven(self, fit: RatingsFit, players: tuple) -> tuple[float, float] | None:
        """(attack, defence) of an eleven: the mean of the players the fit knows, the attack
        weighted by each player's attack weight."""
        eleven = [fit.players[p] for p in players if p in fit.players]
        if not eleven:
            return None
        weights = fit.attack_weight[eleven]
        attack = (
            float(fit.attack[eleven] @ weights / weights.sum())
            if weights.sum() > 0
            else float(fit.attack[eleven].mean())
        )
        return attack, float(fit.defence[eleven].mean())

    def _fit_of_week(self, state: RatingsState, week_seconds: int) -> RatingsFit | None:
        if state.fitted_week is not None and state.fitted_week[0] == week_seconds:
            return state.fitted_week[1]
        window = self.parameters.window_days * SECONDS_PER_DAY
        last = bisect_right(state.known_at_seconds, week_seconds)
        first = bisect_left(state.known_at_seconds, week_seconds - window)
        fit = None
        if last - first >= self.parameters.minimum_results:
            known = np.array(state.known_at_seconds[first:last], dtype="int64")
            decay = np.exp(-self.parameters.time_decay_per_day * (week_seconds - known)
                           / SECONDS_PER_DAY)  # fmt: skip
            fit = self._fit(state, first, last, decay)
        state.fitted_week = (week_seconds, fit)
        return fit

    @run_on_one_thread
    def _fit(self, state: RatingsState, first: int, last: int, decay: np.ndarray) -> RatingsFit:
        count = last - first
        rows = [(match, side, player, minutes, keeper)
                for match in range(count)
                for side, player, minutes, _, keeper in state.players[first + match]]  # fmt: skip
        players = {player: index for index, player in enumerate(sorted({r[2] for r in rows}))}
        size = len(players)
        match = np.array([r[0] for r in rows], dtype="int64")
        side = np.array([r[1] for r in rows], dtype="int64")
        player = np.array([players[r[2]] for r in rows], dtype="int64")
        minutes = np.array([r[3] for r in rows], dtype="float64")
        kept_goal = np.array([float(r[4]) for r in rows], dtype="float64")
        side_index = match * SIDES + side
        share = minutes / np.bincount(side_index, minutes, SIDES * count)[side_index]
        goalkeeper = np.bincount(player, kept_goal, size) > 0.5 * np.bincount(player, None, size)
        attack_weight = np.where(goalkeeper, self.parameters.goalkeeper_attack_weight, 1.0)
        weighted = minutes * attack_weight[player]
        side_total = np.bincount(side_index, weighted, SIDES * count)[side_index]
        attack_share = weighted / np.maximum(side_total, 1e-12)
        own_row = np.where(side == 0, match, match + count)
        opponent_row = np.where(side == 0, match + count, match)
        own = sparse.csr_matrix((attack_share, (own_row, player)), shape=(2 * count, size))
        opponents = sparse.csr_matrix((share, (opponent_row, player)), shape=(2 * count, size))
        target = np.concatenate([state.home_targets[first:last], state.away_targets[first:last]])
        weights = np.concatenate([decay, decay])
        league = np.array(state.leagues[first:last] * 2, dtype="int64")
        at_home = np.concatenate([np.ones(count), np.zeros(count)])
        leagues = len(self.leagues)
        ridge = self.parameters.ridge

        def measure_loss_and_gradient(theta: np.ndarray) -> tuple[float, np.ndarray]:
            level, home = theta[:leagues], theta[leagues]
            attack = theta[leagues + 1 : leagues + 1 + size]
            defence = theta[leagues + 1 + size :]
            log_intensity = level[league] + home * at_home + own @ attack - opponents @ defence
            intensity = np.exp(log_intensity)
            residual = weights * (intensity - target)
            gradient = np.empty_like(theta)
            gradient[:leagues] = np.bincount(league, residual, leagues)
            gradient[leagues] = residual @ at_home
            gradient[leagues + 1 : leagues + 1 + size] = own.T @ residual + 2 * ridge * attack
            gradient[leagues + 1 + size :] = -(opponents.T @ residual) + 2 * ridge * defence
            loss = float(np.sum(weights * (intensity - target * log_intensity))
                         + ridge * (attack @ attack + defence @ defence))  # fmt: skip
            return loss, gradient

        start = np.zeros(leagues + 1 + 2 * size)
        start[:leagues] = np.log(max(float(target @ weights / weights.sum()), 0.1))
        result = minimize(measure_loss_and_gradient, start, jac=True, method="L-BFGS-B",
                          options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-7})  # fmt: skip
        theta = result.x
        return RatingsFit(
            level=theta[:leagues],
            home_advantage=float(theta[leagues]),
            attack=theta[leagues + 1 : leagues + 1 + size],
            defence=theta[leagues + 1 + size :],
            players=players,
            attack_weight=attack_weight,
        )


def blend(team: float, ratings: float, weight: float) -> float:
    """The champion's intensity of one side: the geometric blend of the team model and the
    player ratings (weight on the ratings), the team model alone when the ratings are missing."""
    if not np.isfinite(ratings):
        return team
    return float(np.exp((1 - weight) * np.log(team) + weight * np.log(ratings)))
