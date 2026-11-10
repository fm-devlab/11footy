"""The team counts' model: each side's corners or cards, negative binomial around what it makes and
what its opponent allows.

For a league, every week: `log mu = level + home (the home side) + make(side) - allow(opponent)`,
fitted by Poisson maximum likelihood with a ridge on the teams' terms, on the matches known before
the week starts within `window_days`, each weighted by a half-life back from the week's start; the
over-dispersion of the league's counts in the same window gives the negative binomial's size. The
two sides are independent: the joint distribution is the product of their distributions, each
folded at `size - 1` (that many or more), and the match's total and each side's count are read
from it, so the three keys of a count always agree.
"""

import math
import sqlite3
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom, poisson

from vesta.data_format import FEED, STATISTICS_LAYERS
from vesta.forecast_log.writing import PublishedDistribution
from vesta.spine.dixon_coles import find_week_start
from vesta.spine.dixon_coles import parse_instant_to_seconds as _seconds_of
from vesta.spine.timing import TimeRules, find_result_known_at, forecast_time

SECONDS_PER_DAY = 86_400
KEYS = {"corners": ("M15.corners", "M15.corners.team", "M15.corners.sides"),
        "cards": ("M17.cards", "M17.cards.team", "M17.cards.sides")}  # fmt: skip
# family A's counts, judged by their plan of 31 October 2026 after the corners and
# the cards (released with 2.6.0); offsides have no sides key (the catalogue asks none)
FAMILY_A_KEYS = {"shots": ("M20.shots", "M20.shots.team", "M20.shots.sides"),
                 "shots_on_target": ("M20.shots_on_target", "M20.shots_on_target.team",
                                     "M20.shots_on_target.sides"),
                 "fouls": ("M21.fouls", "M21.fouls.team", "M21.fouls.sides"),
                 "offsides": ("M19.offsides", "M19.offsides.team", None)}  # fmt: skip
ALL_KEYS = KEYS | FAMILY_A_KEYS
LAYER_ORDER = STATISTICS_LAYERS  # the match report first, then the match feed (decision C3)
# a count's column in match_team_statistics (fouls: also the player thresholds' match terms)
COLUMNS = {"corners": "corners", "cards": "yellow_cards + red_cards", "fouls": "fouls",
           "shots": "shots", "shots_on_target": "shots_on_target",
           "offsides": "offsides"}  # fmt: skip
seconds_of = cache(_seconds_of)  # every week reads the same instants again


def is_stub(sides: list[tuple[int | None, int | None]]) -> bool:
    """A feed stub (plan §2): `sides`, each feed row's (fouls, corners) of one match, both 0 on
    both sides; only the goals and the cards were recorded."""
    return len(sides) == 2 and all(fouls == 0 and corners == 0 for fouls, corners in sides)


@dataclass(frozen=True)
class CountMatch:
    match_id: int
    competition: str
    season: int
    home_team_id: int
    away_team_id: int
    forecast_time: str
    known_at: str
    home_count: int | None  # None: no provider counts the match
    away_count: int | None
    layer: str | None


@dataclass(frozen=True)
class FitParameters:
    window_days: int
    half_life_days: float
    ridge: float
    minimum_matches: int
    size: int

    @classmethod
    def from_config(cls, config: dict, count: str) -> "FitParameters":
        fit = config["fit"]
        return cls(int(fit["window_days"]), float(fit["half_life_days"]), float(fit["ridge"]),
                   int(fit["minimum_matches"]), int(config["counts"][count]["size"]))  # fmt: skip


@dataclass(frozen=True)
class LeagueFit:
    teams: dict[int, int]
    level: float
    home: float
    make: np.ndarray
    allow: np.ndarray
    size_nb: float  # the negative binomial's size; inf: Poisson

    def predict_means(self, home: int, away: int) -> tuple[float, float]:
        def look_up_term(values: np.ndarray, team: int) -> float:
            return float(values[self.teams[team]]) if team in self.teams else 0.0

        return (
            math.exp(
                self.level
                + self.home
                + look_up_term(self.make, home)
                - look_up_term(self.allow, away)
            ),
            math.exp(self.level + look_up_term(self.make, away) - look_up_term(self.allow, home)),
        )


def read_count_matches(world: sqlite3.Connection, competitions: list[str], first_season: int,
                       last_season: int, count: str,
                       rules: TimeRules) -> list[CountMatch]:  # fmt: skip
    """The league matches played of `competitions`, each side's count from the provider of record
    (the report, else the feed; a feed stub's counts unknown but for the cards, as
    `outcomes.match_counts`), in the order of their kick-off."""
    column = COLUMNS[count]
    marks = ", ".join("?" for _ in competitions)
    counts: dict[tuple[int, int], dict[str, int]] = defaultdict(dict)
    feed_rows: dict[int, list[tuple[int | None, int | None]]] = defaultdict(list)
    for match_id, team_id, layer, value, fouls, corners in world.execute(
        f"SELECT t.match_id, t.team_id, t.layer, t.{column}, t.fouls,"
        f" t.corners"
        f" FROM match_team_statistics t"
        f" JOIN matches m ON m.match_id = t.match_id JOIN seasons s ON s.season_id ="
        f" m.season_id JOIN competitions c ON c.competition_id = s.competition_id"
        f" WHERE c.code IN ({marks}) AND s.season BETWEEN ? AND ?",
        (*competitions, first_season, last_season),
    ):
        if value is not None:
            counts[(match_id, team_id)][layer] = int(value)
        if layer == FEED:
            feed_rows[match_id].append((fouls, corners))
    found = []
    for (match_id, code, season, home, away, kickoff, precision, local_date) in world.execute(
            f"SELECT m.match_id, c.code, s.season, m.home_team_id, m.away_team_id, m.kickoff_utc,"
            f" m.kickoff_precision, m.local_date FROM matches m JOIN seasons s ON s.season_id ="
            f" m.season_id JOIN competitions c ON c.competition_id = s.competition_id"
            f" WHERE c.code IN ({marks}) AND s.season BETWEEN ? AND ? AND m.status = 'played'"
            f" AND COALESCE(m.stage, '') <> 'play-off' ORDER BY m.kickoff_utc, m.match_id",
            (*competitions, first_season, last_season)):  # fmt: skip
        home_counts = counts.get((match_id, home), {})
        away_counts = counts.get((match_id, away), {})
        layer = next((name for name in LAYER_ORDER if name in home_counts and name in away_counts),
                      None)  # fmt: skip
        if layer == FEED and count != "cards" and is_stub(feed_rows[match_id]):
            layer = None
        found.append(CountMatch(
            match_id, code, season, home, away,
            forecast_time(kickoff, precision, local_date, rules),
            find_result_known_at(kickoff, precision, local_date, rules),
            home_counts[layer] if layer else None, away_counts[layer] if layer else None,
            layer))  # fmt: skip
    return found


def fit_week(known: list[CountMatch], week_seconds: int, parameters: FitParameters
             ) -> LeagueFit | None:  # fmt: skip
    """The league's fit for the week starting at `week_seconds`, from `known` (one league's
    matches with counts, each known before the week starts)."""
    first = week_seconds - parameters.window_days * SECONDS_PER_DAY
    rows = [match for match in known if first <= seconds_of(match.known_at) < week_seconds]
    if len(rows) < parameters.minimum_matches:
        return None
    teams = {team: index for index, team in enumerate(sorted(
        {match.home_team_id for match in rows} | {match.away_team_id for match in rows}))}
    team_count = len(teams)
    age = np.array([(week_seconds - seconds_of(match.known_at)) / SECONDS_PER_DAY
                    for match in rows])  # fmt: skip
    weight = np.repeat(0.5 ** (age / parameters.half_life_days), 2)
    # two observations a match: the home side's count, then the away side's
    y = np.array([[match.home_count, match.away_count] for match in rows], float).ravel()
    columns = 2 + 2 * team_count
    design = np.zeros((len(y), columns))
    design[:, 0] = 1.0
    design[0::2, 1] = 1.0
    for index, match in enumerate(rows):
        home, away = teams[match.home_team_id], teams[match.away_team_id]
        design[2 * index, 2 + home], design[2 * index, 2 + team_count + away] = 1.0, -1.0
        design[2 * index + 1, 2 + away], design[2 * index + 1, 2 + team_count + home] = 1.0, -1.0
    penalty = np.r_[0.0, 0.0, np.full(2 * team_count, parameters.ridge)]
    beta = np.zeros(columns)
    beta[0] = math.log(max(np.average(y, weights=weight), 1e-6))
    for _ in range(50):  # Newton's method on the penalised Poisson likelihood
        mu = np.exp(design @ beta)
        gradient = design.T @ (weight * (y - mu)) - penalty * beta
        hessian = (design * (weight * mu)[:, None]).T @ design + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    mu = np.exp(design @ beta)
    excess = float(np.sum(weight * ((y - mu) ** 2 - mu)) / np.sum(weight * mu ** 2))
    return LeagueFit(teams, float(beta[0]), float(beta[1]), beta[2:2 + team_count].copy(),
                     beta[2 + team_count:].copy(),
                     1.0 / excess if excess > 1e-6 else math.inf)  # fmt: skip


def make_side_distribution(mean: float, size_nb: float, size: int) -> np.ndarray:
    """A side's count 0 .. size - 1, the last cell holding that many or more."""
    counts = np.arange(size - 1)
    if math.isinf(size_nb):
        body = poisson.pmf(counts, mean)
    else:
        body = nbinom.pmf(counts, size_nb, size_nb / (size_nb + mean))
    return np.r_[body, max(1.0 - float(body.sum()), 0.0)]


def make_joint_distribution(means: tuple[float, float], size_nb: float, size: int,
                            tilt: float = 0.0) -> np.ndarray:
    """The two sides' counts together: the product of their distributions, tilted by
    `exp(tilt (i - m_home) (j - m_away))` and renormalised (candidate 2; 0 keeps them
    independent)."""
    home = make_side_distribution(means[0], size_nb, size)
    away = make_side_distribution(means[1], size_nb, size)
    joint = np.outer(home, away)
    if tilt:
        counts = np.arange(size)
        joint = joint * np.exp(tilt * np.outer(counts - means[0], counts - means[1]))
        joint /= joint.sum()
    return joint


def count_distributions(count: str, match_id: int, home_team_id: int, away_team_id: int,
                        means: tuple[float, float], size_nb: float, size: int,
                        version: str,
                        tilt: float = 0.0) -> list[PublishedDistribution]:  # fmt: skip
    """The keys of `count` for one match, read from one joint distribution."""
    total_key, team_key, sides_key = ALL_KEYS[count]
    joint = make_joint_distribution(means, size_nb, size, tilt)
    home, away = joint.sum(axis=1), joint.sum(axis=0)
    total = np.array([np.trace(np.fliplr(joint), offset=size - 1 - goals)
                      for goals in range(2 * size - 1)])  # fmt: skip

    def format_as_count(probabilities: np.ndarray) -> dict:
        return {"probabilities": probabilities.tolist(), "tail_from": len(probabilities),
                "tail": 0.0}  # fmt: skip

    found = [
        PublishedDistribution(total_key, match_id, version, format_as_count(total)),
        PublishedDistribution(team_key, home_team_id, version, format_as_count(home)),
        PublishedDistribution(team_key, away_team_id, version, format_as_count(away)),
    ]
    if sides_key is not None:
        found.append(PublishedDistribution(sides_key, match_id, version,
                                           {"matrix": joint.tolist(), "tail": 0.0}))  # fmt: skip
    return found


Forecast = tuple[CountMatch, tuple[float, float], float]


@dataclass(frozen=True)
class Calibration:
    """Candidate 2's corrections, fitted on the forecasts of the seasons before: per side
    `log mu' = intercept + slope log mu`, and the tilt of the two sides' dependence."""

    home: tuple[float, float] = (0.0, 1.0)
    away: tuple[float, float] = (0.0, 1.0)
    tilt: float = 0.0
    # family A's candidate 2: after the sides' slopes, the match's total `T` scaled to
    # `exp(intercept + slope log T)`, the two sides keeping their shares (identity: 2.2.0)
    total: tuple[float, float] = (0.0, 1.0)

    def predict_means(self, means: tuple[float, float]) -> tuple[float, float]:
        home = math.exp(self.home[0] + self.home[1] * math.log(means[0]))
        away = math.exp(self.away[0] + self.away[1] * math.log(means[1]))
        if self.total == (0.0, 1.0):
            return home, away
        both = home + away
        scale = math.exp(self.total[0] + self.total[1] * math.log(both)) / both
        return home * scale, away * scale


TILTS = np.linspace(-0.08, 0.08, 33)


def fit_size(forecasts: list[Forecast], calibration: Calibration) -> float:
    """Family A's candidate 2: the negative binomial's size by maximum likelihood of both sides'
    counts around their corrected means on `forecasts` (out of sample: the seasons before the one
    forecast), so a side's spread is the forecast's error, not the weekly fit's in-sample one;
    inf when the counts are no wider than Poisson."""
    means = np.array([calibration.predict_means(item[1]) for item in forecasts]).ravel()
    counts = np.array([(match.home_count, match.away_count) for match, _, _ in forecasts],
                      dtype=float).ravel()  # fmt: skip
    if np.mean((counts - means) ** 2) <= np.mean(means):
        return math.inf

    def score_size(log_size: float) -> float:
        size_nb = math.exp(log_size)
        return -float(np.sum(nbinom.logpmf(counts, size_nb, size_nb / (size_nb + means))))

    found = minimize_scalar(score_size, bounds=(math.log(0.5), math.log(1e4)), method="bounded")
    return math.exp(float(found.x))


def _fit_poisson_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """`log E[y] = a + b x` by Poisson maximum likelihood."""
    beta = np.array([0.0, 1.0])
    design = np.c_[np.ones_like(x), x]
    for _ in range(50):
        mu = np.exp(design @ beta)
        step = np.linalg.solve((design * mu[:, None]).T @ design, design.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def fit_candidate_two(forecasts: list[Forecast], size: int,
                      group_of: Callable[[object], str]
                      ) -> tuple[Calibration, dict[str, float]]:  # fmt: skip
    """Family A's candidate 2 (its plan, §13), on
    `forecasts` of the seasons before the one forecast, in order: the sides' slopes (as 2.2.0);
    the slope of the match's total on the corrected total; the negative binomial's size per
    `group_of(match)` around the corrected means; then the tilt with those sizes. Returns the
    corrections and the sizes, which replace the weekly fits' own."""
    sides = Calibration(*(_fit_poisson_slope(np.log([means[side] for _, means, _ in forecasts]),
                                             np.array([(match.home_count, match.away_count)[side]
                                                       for match, _, _ in forecasts], dtype=float))
                          for side in (0, 1)))  # fmt: skip
    corrected = [sides.predict_means(item[1]) for item in forecasts]
    total = _fit_poisson_slope(np.log([sum(means) for means in corrected]),
                               np.array([match.home_count + match.away_count
                                         for match, _, _ in forecasts], dtype=float))  # fmt: skip
    calibration = Calibration(sides.home, sides.away, 0.0, total)
    groups: dict[str, list[Forecast]] = defaultdict(list)
    for item in forecasts:
        groups[group_of(item[0])].append(item)
    sizes = {name: fit_size(items, calibration) for name, items in groups.items()}
    resized = [(match, means, sizes[group_of(match)]) for match, means, _ in forecasts]
    return Calibration(sides.home, sides.away, _fit_tilt(resized, calibration, size), total), sizes


def fit_calibration(forecasts: list[Forecast], size: int) -> Calibration:
    """The slopes by Poisson maximum likelihood on `forecasts` (out of sample, the seasons before
    the one forecast), then the tilt by maximum likelihood of the joint counts on the grid."""
    sides = []
    for side in (0, 1):
        x = np.log([means[side] for _, means, _ in forecasts])
        y = np.array([(match.home_count, match.away_count)[side] for match, _, _ in forecasts],
                     dtype=float)  # fmt: skip
        sides.append(_fit_poisson_slope(x, y))
    calibration = Calibration(sides[0], sides[1])
    return Calibration(sides[0], sides[1], _fit_tilt(forecasts, calibration, size))


def _fit_tilt(forecasts: list[Forecast], calibration: Calibration, size: int) -> float:
    """The tilt of the two sides' dependence on the grid, by maximum likelihood of the joint
    counts around `calibration`'s means, each forecast with its own size."""
    best, best_loss = 0.0, math.inf
    cells = [(min(match.home_count, size - 1), min(match.away_count, size - 1))
             for match, _, _ in forecasts]  # fmt: skip
    joints = [make_joint_distribution(calibration.predict_means(means), size_nb, size)
              for _, means, size_nb in forecasts]  # fmt: skip
    counts = np.arange(size)
    for tilt in TILTS:
        loss = 0.0
        for (home, away), joint, (_, means, _) in zip(cells, joints, forecasts, strict=True):
            mean = calibration.predict_means(means)
            tilted = joint * np.exp(tilt * np.outer(counts - mean[0], counts - mean[1]))
            loss -= math.log(max(tilted[home, away] / tilted.sum(), 1e-300))
        if loss < best_loss:
            best, best_loss = float(tilt), loss
    return best


def walk_forward(matches: list[CountMatch], parameters: FitParameters, seasons: set[int],
                 audit: Callable[[CountMatch, int], None] | None = None) -> list[Forecast]:
    """Every match of `seasons` with counts, forecast at its kick-off by the fit of its week on
    its league's matches known before the week: (match, the two means, the size). `audit`
    receives each forecast match with the latest instant (seconds) one of its fit's matches was
    known at."""
    by_league: dict[str, list[CountMatch]] = defaultdict(list)
    for match in matches:
        if match.home_count is not None:
            by_league[match.competition].append(match)
    forecasts = []
    for league_matches in by_league.values():
        fits: dict[int, LeagueFit | None] = {}
        latest: dict[int, int] = {}
        for match in league_matches:
            if match.season not in seasons:
                continue
            week = find_week_start(seconds_of(match.forecast_time))
            if week not in fits:
                known = [other for other in league_matches
                         if seconds_of(other.known_at) < week]  # fmt: skip
                fits[week] = fit_week(known, week, parameters)
                latest[week] = max((seconds_of(other.known_at) for other in known), default=0)
            fit = fits[week]
            if fit is not None:
                if audit is not None:
                    audit(match, latest[week])
                forecasts.append((match, fit.predict_means(match.home_team_id, match.away_team_id),
                                  fit.size_nb))  # fmt: skip
    return forecasts
