"""The team counts of the UEFA cups: one pooled fit over the five leagues and the cups
(family A's plan, §4).

A cup match needs the terms of clubs the leagues never meet. For each cup week, one fit on every
match of the leagues and the cups known before the week starts, within `window_days`, the feed's
counts only (the cups have no other layer, so the fit and the outcome read the same one). The
feed's stubs (both sides' fouls and corners 0: only goals and cards recorded) count as unknown,
but for the cards, and so do the offsides of a season that did not record them (plan §2):

    log mu = level(group) + home(kind) * venue + make(side) - allow(opponent)
             + [a cup row] elo * (rating(side) - rating(opponent)) / 400

`group` is the league's code, or CL, EL, ECL, or Q for a qualifying round (the spine's rule:
`ClubMatch.qualifying`); `kind` is league or cup; `venue` 1 at home, 0 on neutral ground; the
ratings are the spine's cups Elo before the kick-off (`vesta.spine.cups.rate_matches`). The ridge
shrinks `make` and `allow` only; the negative binomial's size comes from the window's cup rows.
"""

import math
import sqlite3
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse

from vesta.data_format import FEED
from vesta.spine.cups import CupsParameters, _find_club_countries, _find_league_clubs, rate_matches
from vesta.spine.dixon_coles import find_week_start
from vesta.spine.elo import ELO_SCALE
from vesta.spine.timing import TimeRules, find_result_known_at, forecast_time
from vesta.spine.world import read_club_matches
from vesta.team_counts.model import COLUMNS, SECONDS_PER_DAY, FitParameters, is_stub, seconds_of
from vesta.team_counts.outcomes import are_offsides_unrecorded

QUALIFYING = "Q"
MAIN_CUPS = ("CL", "EL", "ECL")


@dataclass(frozen=True)
class CupCountMatch:
    match_id: int
    competition: str
    group: str  # the level's group: a league's code, CL, EL, ECL, or Q (a qualifying round)
    is_cup: bool
    season: int
    home_team_id: int
    away_team_id: int
    forecast_time: str
    known_at: str
    neutral: bool
    home_rating: float | None  # the spine's cups Elo before the kick-off; None in a league
    away_rating: float | None
    home_count: int | None  # None: the feed holds no counts of the match
    away_count: int | None
    layer: str | None

    @property
    def qualifying(self) -> bool:
        return self.group == QUALIFYING


def read_cup_count_matches(world: sqlite3.Connection, spine: dict, first_season: int,
                           last_season: int, count: str) -> list[CupCountMatch]:  # fmt: skip
    """The matches of the five leagues and the cups of `first_season` .. `last_season`, each
    side's feed count of `count`, the cup matches with the two Elo ratings before the kick-off,
    in the order of their kick-off."""
    cups = CupsParameters.from_config(spine)
    rules = TimeRules.from_config(spine)
    club_matches = read_club_matches(world, list(cups.leagues), list(cups.cups),
                                     cups.first_league_season, cups.first_cup_season,
                                     last_season)  # fmt: skip
    ratings = {rated.match.match_id: (rated.home_rating, rated.away_rating, rated.match.qualifying)
               for rated in rate_matches(club_matches, cups)}  # fmt: skip
    codes = (*cups.leagues, *cups.cups)
    marks = ", ".join("?" for _ in codes)
    counts: dict[tuple[int, int], int] = {}
    # (fouls, corners) of each feed row
    feed_rows: dict[int, list[tuple[int | None, int | None]]] = defaultdict(list)
    for match_id, team_id, value, fouls, corners in world.execute(
            f"SELECT t.match_id, t.team_id, t.{COLUMNS[count]}, t.fouls, t.corners"
            f" FROM match_team_statistics t"
            f" JOIN matches m ON m.match_id = t.match_id JOIN seasons s ON s.season_id ="
            f" m.season_id JOIN competitions c ON c.competition_id = s.competition_id"
            f" WHERE t.layer = '{FEED}' AND c.code IN ({marks})"
            f" AND s.season BETWEEN ? AND ?",
            (*codes, first_season, last_season)):  # fmt: skip
        feed_rows[match_id].append((fouls, corners))
        if value is not None:
            counts[(match_id, team_id)] = int(value)
    stubs = {match_id for match_id, sides in feed_rows.items() if is_stub(sides)}
    found = []
    for (match_id, code, season, stage, home, away, kickoff, precision, local_date,
         neutral) in world.execute(
            f"SELECT m.match_id, c.code, s.season, COALESCE(m.stage, ''),"
            f" m.home_team_id, m.away_team_id, m.kickoff_utc, m.kickoff_precision, m.local_date,"
            f" m.neutral_venue FROM matches m JOIN seasons s ON s.season_id = m.season_id"
            f" JOIN competitions c ON c.competition_id = s.competition_id"
            f" WHERE c.code IN ({marks}) AND s.season BETWEEN ? AND ? AND m.status = 'played'"
            f" ORDER BY m.kickoff_utc, m.match_id",
            (*codes, first_season, last_season)):  # fmt: skip
        is_cup = code in cups.cups
        if not is_cup and stage == "play-off":  # a league's play-off, as the leagues' reader
            continue
        if is_cup and match_id not in ratings:
            continue  # no result on the pitch: the spine cannot rate it
        home_rating, away_rating, qualifying = ratings.get(match_id, (None, None, False))
        group = QUALIFYING if qualifying else code
        home_count, away_count = counts.get((match_id, home)), counts.get((match_id, away))
        both = (home_count is not None and away_count is not None
                and (count == "cards" or match_id not in stubs))  # fmt: skip
        found.append(CupCountMatch(
            match_id, code, group, is_cup, season, home, away,
            forecast_time(kickoff, precision, local_date, rules),
            find_result_known_at(kickoff, precision, local_date, rules), bool(neutral),
            home_rating, away_rating, home_count if both else None, away_count if both else None,
            FEED if both else None))  # fmt: skip
    if count == "offsides":
        found = _drop_unrecorded_offsides(found)
    return found


def _drop_unrecorded_offsides(matches: list[CupCountMatch]) -> list[CupCountMatch]:
    """A competition's season that did not record its offsides (`outcomes.offsides_unrecorded`)
    has them unknown (plan §2: the feed wrote 0 for none recorded)."""
    seasons: dict[tuple[str, int], list[bool]] = defaultdict(list)
    for match in matches:
        if match.home_count is not None:
            seasons[(match.competition, match.season)].append(
                match.home_count == 0 and match.away_count == 0)
    unrecorded = {key for key, zeros in seasons.items()
                  if are_offsides_unrecorded(sum(zeros), len(zeros)) is not False}  # fmt: skip
    return [replace(match, home_count=None, away_count=None, layer=None)
            if (match.competition, match.season) in unrecorded else match
            for match in matches]  # fmt: skip


@dataclass(frozen=True)
class PooledFit:
    teams: dict[int, int]
    levels: dict[str, float]
    home: dict[bool, float]  # by is_cup
    elo: float
    make: np.ndarray
    allow: np.ndarray
    size_nb: float  # the negative binomial's size, from the cup rows; inf: Poisson

    def choose_level(self, group: str) -> float:
        """A group the window has not met (the ECL's first weeks, the qualifiers' before their
        counts) takes the EL's level, else the mean of the cups' levels."""
        if group in self.levels:
            return self.levels[group]
        if "EL" in self.levels:
            return self.levels["EL"]
        cups = [self.levels[name] for name in (*MAIN_CUPS, QUALIFYING) if name in self.levels]
        return float(np.mean(cups)) if cups else float(np.mean(list(self.levels.values())))

    def predict_means(self, match: CupCountMatch) -> tuple[float, float]:
        def look_up_term(values: np.ndarray, team: int) -> float:
            return float(values[self.teams[team]]) if team in self.teams else 0.0

        base = self.choose_level(match.group)
        venue = 0.0 if match.neutral else self.home[match.is_cup]
        gap = 0.0
        if match.is_cup:
            gap = self.elo * (match.home_rating - match.away_rating) / ELO_SCALE
        return (math.exp(base + venue + look_up_term(self.make, match.home_team_id)
                         - look_up_term(self.allow, match.away_team_id) + gap),
                math.exp(base + look_up_term(self.make, match.away_team_id)
                         - look_up_term(self.allow, match.home_team_id) - gap))  # fmt: skip


def fit_pooled(rows: list[CupCountMatch], weights: np.ndarray, ridge: float,
               iterations: int = 60) -> PooledFit | None:  # fmt: skip
    """The pooled fit on `rows` (matches with counts), each weighted by `weights`."""
    if not any(row.is_cup for row in rows):
        return None
    groups = sorted({row.group for row in rows})
    teams = {team: index for index, team in enumerate(sorted(
        {row.home_team_id for row in rows} | {row.away_team_id for row in rows}))}
    group_count, team_count = len(groups), len(teams)
    group_of = {name: index for index, name in enumerate(groups)}
    # columns: the groups' levels, home (league), home (cup), elo, make (n), allow (n)
    home_league, home_cup, elo = group_count, group_count + 1, group_count + 2
    columns = group_count + 3 + 2 * team_count
    cells, rows_of, cols_of = [], [], []
    y = np.empty(2 * len(rows))
    for index, row in enumerate(rows):
        gap = ((row.home_rating - row.away_rating) / ELO_SCALE) if row.is_cup else 0.0
        home, away = teams[row.home_team_id], teams[row.away_team_id]
        for side, (attack, defend, sign) in enumerate(((home, away, 1.0), (away, home, -1.0))):
            row_index = 2 * index + side
            entries = [(group_of[row.group], 1.0), (group_count + 3 + attack, 1.0),
                       (group_count + 3 + team_count + defend, -1.0)]  # fmt: skip
            if side == 0 and not row.neutral:
                entries.append((home_cup if row.is_cup else home_league, 1.0))
            if gap:
                entries.append((elo, sign * gap))
            for column, value in entries:
                rows_of.append(row_index)
                cols_of.append(column)
                cells.append(value)
        y[2 * index], y[2 * index + 1] = row.home_count, row.away_count
    design = sparse.csr_matrix((cells, (rows_of, cols_of)), shape=(len(y), columns))
    weight = np.repeat(weights, 2)
    is_cup = np.repeat([row.is_cup for row in rows], 2)
    penalty = np.r_[np.zeros(group_count + 3), np.full(2 * team_count, ridge)]
    beta = np.zeros(columns)
    for name, index in group_of.items():
        chosen = np.repeat([row.group == name for row in rows], 2)
        beta[index] = math.log(max(np.average(y[chosen], weights=weight[chosen]), 1e-6))
    for _ in range(iterations):  # Newton's method on the penalised Poisson likelihood
        mu = np.exp(design @ beta)
        gradient = design.T @ (weight * (y - mu)) - penalty * beta
        hessian = (design.T @ design.multiply((weight * mu)[:, None])).toarray()
        step = np.linalg.solve(hessian + np.diag(penalty) + 1e-9 * np.eye(columns), gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    mu = np.exp(design @ beta)
    cup_weights, observed, expected = weight[is_cup], y[is_cup], mu[is_cup]
    excess = float(np.sum(cup_weights * ((observed - expected) ** 2 - expected))
                   / np.sum(cup_weights * expected ** 2))
    return PooledFit(teams, {name: float(beta[index]) for name, index in group_of.items()},
                     {False: float(beta[home_league]), True: float(beta[home_cup])},
                     float(beta[elo]), beta[group_count + 3:group_count + 3 + team_count].copy(),
                     beta[group_count + 3 + team_count:].copy(),
                     1.0 / excess if excess > 1e-6 else math.inf)  # fmt: skip


def fit_week(matches: list[CupCountMatch], week_seconds: int,
             parameters: FitParameters) -> PooledFit | None:  # fmt: skip
    """The pooled fit for the week starting at `week_seconds`: the matches with counts known
    before it, within the window, weighted by the half-life back from the week's start."""
    first = week_seconds - parameters.window_days * SECONDS_PER_DAY
    rows = [match for match in matches if match.home_count is not None
            and first <= seconds_of(match.known_at) < week_seconds]  # fmt: skip
    if len(rows) < parameters.minimum_matches:
        return None
    age = np.array([(week_seconds - seconds_of(row.known_at)) / SECONDS_PER_DAY for row in rows])
    return fit_pooled(rows, 0.5 ** (age / parameters.half_life_days), parameters.ridge)


Forecast = tuple[CupCountMatch, tuple[float, float], float]


def walk_forward(matches: list[CupCountMatch], parameters: FitParameters, seasons: set[int],
                 audit: Callable[[CupCountMatch, int], None] | None = None) -> list[Forecast]:
    """Every cup match of `seasons` with counts, forecast at its kick-off by the pooled fit of
    its week: (match, the two means, the size). `audit` receives each forecast match with the
    latest instant (seconds) one of its fit's matches was known at."""
    fits: dict[int, PooledFit | None] = {}
    latest: dict[int, int] = {}
    known = [match for match in matches if match.home_count is not None]
    forecasts = []
    for match in matches:
        if not match.is_cup or match.home_count is None or match.season not in seasons:
            continue
        week = find_week_start(seconds_of(match.forecast_time))
        if week not in fits:
            fits[week] = fit_week(known, week, parameters)
            first = week - parameters.window_days * SECONDS_PER_DAY
            latest[week] = max((seconds_of(row.known_at) for row in known
                                if first <= seconds_of(row.known_at) < week), default=0)
        fit = fits[week]
        if fit is not None:
            if audit is not None:
                audit(match, latest[week])
            forecasts.append((match, fit.predict_means(match), fit.size_nb))
    return forecasts


def read_elo_inputs_known(world: sqlite3.Connection, spine: dict,
                          last_season: int) -> dict[int, int]:  # fmt: skip
    """For each cup match, the latest instant (seconds) a result feeding its two ratings was
    known at: each club's previous match, and for a club new to the rating the results its
    peers' ratings held when the spine took their mean (`vesta.spine.cups`, 97a02c84: the
    ratings as of the results it counts as known by the kick-off). Each result's instant here is
    the world's `result_known_at`, so the check is independent of the spine's own clock."""
    cups = CupsParameters.from_config(spine)
    rules = TimeRules.from_config(spine)
    matches = read_club_matches(world, list(cups.leagues), list(cups.cups),
                                cups.first_league_season, cups.first_cup_season,
                                last_season)  # fmt: skip
    known_at = {match_id: seconds_of(find_result_known_at(kickoff, precision, local_date, rules))
                for match_id, kickoff, precision, local_date in world.execute(
                    "SELECT match_id, kickoff_utc, kickoff_precision, local_date FROM matches")}
    ordered = sorted(matches, key=lambda match: (match.kickoff_utc, match.is_cup, match.match_id))
    league_of, _ = _find_league_clubs(ordered, cups)
    country = _find_club_countries(ordered)
    delay = cups.result_known_after_minutes * 60
    last: dict[int, int] = {}  # each club's latest result, as the club's own rating holds it
    held: dict[int, int] = {}  # the latest result each club's known rating holds
    pending: deque[tuple[int, int, int]] = deque()  # (the spine's known instant, club, result's)
    found = {}
    for match in ordered:
        kickoff = seconds_of(match.kickoff_utc)
        while pending and pending[0][0] <= kickoff:
            _, club, instant = pending.popleft()
            held[club] = instant
        latest = 0
        for team in (match.home_team_id, match.away_team_id):
            if team in last:
                latest = max(latest, last[team])
                continue
            peers = ([club for club in held if league_of.get(club) == league_of[team]]
                     if team in league_of else
                     [club for club in held if club not in league_of
                      and country.get(club) == country.get(team)])  # fmt: skip
            latest = max([latest, *(held[club] for club in peers)])
            held.setdefault(team, 0)
        if match.is_cup:
            found[match.match_id] = latest
        last[match.home_team_id] = last[match.away_team_id] = known_at[match.match_id]
        for team in (match.home_team_id, match.away_team_id):
            pending.append((kickoff + delay, team, known_at[match.match_id]))
    return found


def forecast_in_hindsight(matches: list[CupCountMatch], parameters: FitParameters,
                          season: int) -> list[Forecast]:  # fmt: skip
    """The hindsight of the alarm (plan §8, §11): one pooled fit on every match of `season` with
    counts, unweighted, and its own cup matches forecast by it."""
    rows = [match for match in matches if match.season == season and match.home_count is not None]
    fit = fit_pooled(rows, np.ones(len(rows)), parameters.ridge)
    if fit is None:
        return []
    return [(match, fit.predict_means(match), fit.size_nb) for match in rows if match.is_cup]

