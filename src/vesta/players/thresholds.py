"""The player thresholds (their plan, 5f4c62c): a
player's shots, shots on target, key passes, fouls committed and drawn, booking, saves and offsides
in a match if he plays, each a whole distribution.

A player's rate of each count per 90 minutes comes from his league matches of this season and the
one before, each match back weighing `FADE` less, shrunk to his position group's rate by `STRENGTH`
nineties (the scorers' construction). A keeper's saves are his saves per goal conceded, shrunk the
same way, times the opponent's expected goals. The rate is scaled by the match (`MatchTerms`): the
side's expected goals against its own recent level for the attacking counts, the opponent's and
the venue's terms of team counts' weekly fits for fouls and cards. His count is a negative
binomial mixed over his minutes if he plays (the lineups' distribution, in `MINUTE_BINS`), with
one dispersion per count and position group fitted on the training seasons.

What happened: the feed's counts of the player (`player_match_statistics`); a keeper's saves are
the opponent's tracked shots saved while he was on the pitch (the feed's keeper saves are broken
in 2020-2022), and the goals in the same window are his goals conceded.
"""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.stats import nbinom, poisson

from vesta.data_format import build_player_rows_filter
from vesta.lineups.model import classify_position
from vesta.spine.dixon_coles import find_week_start
from vesta.team_counts.model import CountMatch, FitParameters, LeagueFit, fit_week, seconds_of

FADE = 0.97  # each league match back
STRENGTH = 5.0  # nineties of the group's rate
REFERENCE_MATCHES = 20  # the reference: his last matches played
REFERENCE_STRENGTH = 5.0  # matches
SIDE_FADE = 0.97  # the side's own recent level of expected goals, each match back

# the counts, in the order of an appearance's array; a count's cells run 0 to cells - 1, the last
# holding that many or more
COUNTS = ("shots", "shots_on_target", "key_passes", "fouls", "fouls_drawn", "booked", "saves",
          "offsides")  # fmt: skip
SAVES = COUNTS.index("saves")


@dataclass(frozen=True)
class Threshold:
    count: str
    kind: str  # count or binary
    cells: int
    lines: tuple[int, ...]  # at_least
    term: str  # the match term scaling his rate
    who: str  # everyone, keepers, outfield


KEYS = {
    "P1.shots": Threshold("shots", "count", 10, (1, 2, 3, 4), "attack", "everyone"),
    "P2.shots_on_target": Threshold("shots_on_target", "count", 7, (1, 2, 3), "attack",
                                    "everyone"),  # fmt: skip
    "P7.key_passes": Threshold("key_passes", "count", 7, (1, 2, 3), "attack", "everyone"),
    "P8.fouls": Threshold("fouls", "count", 7, (1, 2, 3), "fouls", "everyone"),
    "P8.fouls_drawn": Threshold("fouls_drawn", "count", 7, (1, 2, 3), "fouls_drawn", "everyone"),
    "P9.booked": Threshold("booked", "binary", 2, (1,), "cards", "everyone"),
    "P14.saves": Threshold("saves", "count", 11, (1, 2, 3, 4, 5), "saves", "keepers"),
    "P16.offsides": Threshold("offsides", "count", 5, (1, 2), "attack", "outfield"),
}
# the minutes if he plays, in bins (first and last minute): his count mixes over them
MINUTE_BINS = ((1, 44), (45, 59), (60, 89), (90, 90))
SIZES = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0, 100.0, math.inf)


@dataclass(frozen=True, slots=True)
class Appearance:
    """A player's league match: his minutes and counts (nan where the source has none)."""

    known_at: str
    season: int
    minutes: float  # 90 or more counts 90
    counts: np.ndarray  # in the order of COUNTS
    conceded: float  # goals in his window (keepers); nan without tracked shots


FEED_ROWS = ("SELECT match_id, player_id, shots, shots_on_target, key_passes, fouls_committed,"
             " fouls_suffered, yellow_cards, offsides, saves IS NOT NULL"
             " FROM player_match_statistics WHERE {rows}")  # fmt: skip


def read_counts(world: sqlite3.Connection, sheets: dict, match_ids: set[int]
                ) -> dict[tuple[int, int], tuple[np.ndarray, float]]:  # fmt: skip
    """(match, player) -> his counts and, for a keeper, the goals conceded in his window."""
    found: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
    keepers = []
    for (match_id, player, shots, on_target, key_passes, fouls, drawn, yellow, offsides,
         keeper) in _read_by_match(world, FEED_ROWS.format(rows=build_player_rows_filter(world)),
                                   match_ids):  # fmt: skip
        values = [shots, on_target, key_passes, fouls, drawn,
                  None if yellow is None else min(yellow, 1), None, offsides]  # fmt: skip
        found[(match_id, player)] = (np.array([math.nan if value is None else float(value)
                                               for value in values]), math.nan)  # fmt: skip
        if keeper:
            keepers.append((match_id, player))
    shots_of: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for match_id, team_id, minute, result in _read_by_match(
            world, "SELECT match_id, team_id, COALESCE(minute, 0), result FROM shots"
                   " WHERE layer = 'tracked'", match_ids):  # fmt: skip
        shots_of[match_id].append((team_id, minute, result))
    lines = {(match_id, player): (team_id, started, minutes)
             for (match_id, team_id), sheet in sheets.items() if match_id in match_ids
             for player, (started, minutes, _) in sheet.items()}  # fmt: skip
    for match_id, player in keepers:
        shots = shots_of.get(match_id)
        line = lines.get((match_id, player))
        if not shots or line is None:
            continue
        team_id, started, minutes = line
        saves = conceded = 0
        for team, minute, result in shots:
            if team == team_id:
                continue
            on = (minute < minutes or minutes >= 90) if started else minute >= 90 - minutes
            if on:
                saves += result == "saved"
                conceded += result == "goal"
        found[(match_id, player)][0][SAVES] = saves
        found[(match_id, player)] = (found[(match_id, player)][0], float(conceded))
    return found


def _read_by_match(world: sqlite3.Connection, query: str, match_ids: set[int]):
    chosen = sorted(match_ids)
    for start in range(0, len(chosen), 500):
        part = chosen[start:start + 500]
        yield from world.execute(query + f" AND match_id IN ({', '.join('?' * len(part))})",
                                 part)  # fmt: skip


def build_histories(world, by_id, counted) -> tuple[dict[int, list[Appearance]], dict]:
    """Each player's league appearances in time order, and the position group of each (his
    position if he started, else the last group he started in)."""
    histories: dict[int, list[Appearance]] = defaultdict(list)
    groups_of: dict[tuple[int, int], str | None] = {}
    last_group: dict[int, str | None] = {}
    seen: set[int] = set()
    items = []
    for played in world.team_matches.values():
        for match in played:
            if match.match_id in by_id and match.match_id not in seen:
                seen.add(match.match_id)
                items.append(match)
    items.sort(key=lambda item: (item.known_at, item.match_id))
    missing = np.full(len(COUNTS), math.nan)
    for match in items:
        league_match = by_id[match.match_id]
        for side in (league_match.home_team_id, league_match.away_team_id):
            for player, (started, minutes, code) in world.sheets.get(
                    (match.match_id, side), {}).items():  # fmt: skip
                if not (started or minutes > 0):
                    continue
                counts, conceded = counted.get((match.match_id, player), (missing, math.nan))
                group = classify_position(code) if started and code else None
                group = group or last_group.get(player)
                if started and code:
                    last_group[player] = group
                groups_of[(player, len(histories[player]))] = group
                histories[player].append(Appearance(match.known_at, match.season,
                                                    min(max(minutes, 1), 90), counts, conceded))
    return dict(histories), groups_of


@dataclass(frozen=True)
class GroupRates:
    """A position group's rates from the seasons before: per 90, per match played (the
    reference's), and a keeper's saves per goal conceded."""

    per_ninety: np.ndarray
    per_match: np.ndarray
    saves_per_goal: float
    conceded_per_ninety: float


def group_rates(appearances: list[tuple[str | None, Appearance]]) -> dict[str | None, GroupRates]:
    sums: dict[str | None, np.ndarray] = defaultdict(lambda: np.zeros((4, len(COUNTS))))
    keeper: dict[str | None, np.ndarray] = defaultdict(lambda: np.zeros(3))
    for group, item in appearances:
        known = ~np.isnan(item.counts)
        sums[group][0] += np.where(known, item.counts, 0.0)
        sums[group][1] += known * item.minutes / 90
        sums[group][2] += known
        if not math.isnan(item.conceded) and known[SAVES]:
            keeper[group] += (item.counts[SAVES], item.conceded, item.minutes / 90)
    everyone = sum(sums.values(), np.zeros((4, len(COUNTS))))
    every_keeper = sum(keeper.values(), np.zeros(3))
    found = {}
    for group, totals in [*sums.items(), (None, everyone)]:
        saves, conceded, nineties = keeper.get(group, every_keeper) if group is not None \
            else every_keeper  # fmt: skip
        if conceded <= 0:
            saves, conceded, nineties = every_keeper
        found[group] = GroupRates(totals[0] / np.maximum(totals[1], 1e-9),
                                  totals[0] / np.maximum(totals[2], 1e-9),
                                  float(saves / max(conceded, 1e-9)),
                                  float(conceded / max(nineties, 1e-9)))  # fmt: skip
    return found


def estimate_own_rates(history: list[Appearance], season: int, rates: GroupRates
                       ) -> tuple[np.ndarray, float]:  # fmt: skip
    """His rates per 90 from his appearances of `season` and the one before (newest last), each
    shrunk to his group's by `STRENGTH` nineties; and a keeper's saves per goal conceded."""
    chosen = [item for item in history if item.season >= season - 1]
    if not chosen:
        return rates.per_ninety.copy(), rates.saves_per_goal
    weights = FADE ** np.arange(len(chosen))[::-1]
    counts = np.array([item.counts for item in chosen])
    known = ~np.isnan(counts)
    nineties = np.array([item.minutes / 90 for item in chosen])
    totals = weights @ np.where(known, counts, 0.0)
    exposure = weights @ (known * nineties[:, None])
    per_ninety = (totals + STRENGTH * rates.per_ninety) / (exposure + STRENGTH)
    conceded = np.array([item.conceded for item in chosen])
    seen = ~np.isnan(conceded) & known[:, SAVES]
    saves = float(weights[seen] @ counts[seen, SAVES]) if seen.any() else 0.0
    goals = float(weights[seen] @ conceded[seen]) if seen.any() else 0.0
    prior_goals = STRENGTH * rates.conceded_per_ninety
    ratio = (saves + prior_goals * rates.saves_per_goal) / (goals + prior_goals)
    return per_ninety, ratio


def estimate_reference_rates(history: list[Appearance], rates: GroupRates) -> np.ndarray:
    """The reference: his counts per match played over his last matches, shrunk to his
    group's by `REFERENCE_STRENGTH` matches."""
    last = history[-REFERENCE_MATCHES:]
    if not last:
        return rates.per_match.copy()
    counts = np.array([item.counts for item in last])
    known = ~np.isnan(counts)
    totals = np.where(known, counts, 0.0).sum(axis=0)
    return (totals + REFERENCE_STRENGTH * rates.per_match) / (known.sum(axis=0)
                                                              + REFERENCE_STRENGTH)


def split_into_minute_bins(minutes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """His minutes if he plays (the lineups' cells 0 to 90), as the weights and mean minutes of
    `MINUTE_BINS`."""
    on = minutes[1:]
    total = float(on.sum())
    if total <= 0:
        return np.array([0.0, 0.0, 0.0, 1.0]), np.array([22.0, 52.0, 75.0, 90.0])
    weights, means = [], []
    for first, last in MINUTE_BINS:
        mass = minutes[first:last + 1]
        weights.append(float(mass.sum()) / total)
        span = np.arange(first, last + 1)
        means.append(float(span @ mass / mass.sum()) if mass.sum() > 0 else (first + last) / 2)
    return np.array(weights), np.array(means)


def count_cells(means: np.ndarray, weights: np.ndarray, size_nb: float, cells: int
                ) -> np.ndarray:  # fmt: skip
    """The mixture over the minute bins (`means` per bin, `weights`) of a negative binomial
    (Poisson when `size_nb` is infinite), cells 0 to `cells - 1`, the last that many or more."""
    counts = np.arange(cells - 1)[None, :]
    means = np.maximum(means, 1e-9)[:, None]
    if math.isinf(size_nb):
        body = poisson.pmf(counts, means)
    else:
        body = nbinom.pmf(counts, size_nb, size_nb / (size_nb + means))
    mixed = weights @ body
    return np.r_[mixed, max(1.0 - float(mixed.sum()), 0.0)]


def make_poisson_cells(mean: float, cells: int) -> np.ndarray:
    body = poisson.pmf(np.arange(cells - 1), max(mean, 1e-9))
    return np.r_[body, max(1.0 - float(body.sum()), 0.0)]


def fit_size(means: np.ndarray, counts: np.ndarray) -> float:
    """The negative binomial's size of `SIZES` with the largest likelihood of `counts` around
    `means` (inf: Poisson)."""
    means = np.maximum(means, 1e-9)
    best, best_loss = math.inf, math.inf
    for size in SIZES:
        if math.isinf(size):
            loss = -float(poisson.logpmf(counts, means).sum())
        else:
            loss = -float(nbinom.logpmf(counts, size, size / (size + means)).sum())
        if loss < best_loss:
            best, best_loss = size, loss
    return best


def iterate_training_rows(
    histories: dict[int, list[Appearance]],
    groups_of: dict,
    seasons: set[int],
    rates: dict[str | None, GroupRates],
):
    """Each appearance of `seasons`: (group, his rates per 90 known before it, the appearance),
    for the dispersions."""
    for player, items in histories.items():
        for index, item in enumerate(items):
            if item.season not in seasons:
                continue
            group = groups_of[(player, index)]
            per_ninety, ratio = estimate_own_rates(items[:index], item.season,
                                                   rates.get(group, rates[None]))  # fmt: skip
            yield group, per_ninety, ratio, item


def fit_sizes(rows, rates: dict[str | None, GroupRates]) -> dict[tuple[str, str | None], float]:
    """One size per count and position group, from the training appearances: his rate known
    before it times the minutes he played (the keeper's saves: his saves per goal times the
    group's goals conceded per 90)."""
    gathered: dict[tuple[str, str | None], tuple[list, list]] = defaultdict(lambda: ([], []))
    for group, per_ninety, ratio, item in rows:
        means = per_ninety * item.minutes / 90
        keeper_rates = rates.get(group, rates[None])
        means[SAVES] = ratio * keeper_rates.conceded_per_ninety * item.minutes / 90
        for index, count in enumerate(COUNTS):
            if not math.isnan(item.counts[index]):
                gathered[(count, group)][0].append(means[index])
                gathered[(count, group)][1].append(item.counts[index])
    return {key: fit_size(np.array(means), np.array(counts))
            for key, (means, counts) in gathered.items() if len(counts) >= 200}  # fmt: skip


class MatchTerms:
    """Per side of a league match: the factors scaling a player's rates (the match against his
    side's usual), from the spine's expected goals and team counts' weekly fits."""

    def __init__(self) -> None:
        self.factors: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
        self.expected_goals: dict[tuple[int, int], float] = {}  # the opponent's, against the side
        self.means: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)  # team counts'

    def set_attack_factors(self, ordered_forecasts) -> None:
        """`ordered_forecasts`: (match, home team, away team, home goals, away goals) in time
        order; the factor is the side's expected goals against its faded mean of the matches
        before (1 for its first)."""
        level: dict[int, tuple[float, float]] = {}
        for match_id, home, away, home_goals, away_goals in ordered_forecasts:
            for team, own, other in ((home, home_goals, away_goals), (away, away_goals,
                                                                       home_goals)):
                total, weight = level.get(team, (0.0, 0.0))
                usual = total / weight if weight > 0 else own
                self.factors[(match_id, team)]["attack"] = own / max(usual, 1e-9)
                self.expected_goals[(match_id, team)] = other
                level[team] = (total * SIDE_FADE + own, weight * SIDE_FADE + 1)

    def fit_count_factors(
        self,
        count: str,
        matches: list[CountMatch],
        parameters: FitParameters,
        seasons: set[int],
        fixtures: tuple[CountMatch, ...] = (),
    ) -> None:
        """Team counts' weekly fits of `count` (fouls, cards): for the side, the venue (half the
        home term, either way) and the opponent's allow term; for fouls drawn, the opponent's
        make term and its venue. The matches of `seasons` with counts get them (the verdict),
        and so do `fixtures`, matches to play (the morning job)."""
        by_league: dict[str, list[CountMatch]] = defaultdict(list)
        for match in matches:
            if match.home_count is not None:
                by_league[match.competition].append(match)
        targets: dict[str, list[CountMatch]] = defaultdict(list)
        for league, league_matches in by_league.items():
            targets[league] += [match for match in league_matches if match.season in seasons]
        for fixture in fixtures:
            targets[fixture.competition].append(fixture)
        for league, chosen in targets.items():
            league_matches = by_league.get(league, [])
            fits: dict[int, LeagueFit | None] = {}
            for match in chosen:
                week = find_week_start(seconds_of(match.forecast_time))
                if week not in fits:
                    fits[week] = fit_week([other for other in league_matches
                                           if seconds_of(other.known_at) < week], week,
                                          parameters)  # fmt: skip
                fit = fits[week]
                if fit is None:
                    continue
                for team, mean in zip((match.home_team_id, match.away_team_id),
                                      fit.predict_means(match.home_team_id, match.away_team_id),
                                      strict=True):  # fmt: skip
                    self.means[(match.match_id, team)][count] = mean
                for team, opponent, venue in ((match.home_team_id, match.away_team_id, 0.5),
                                              (match.away_team_id, match.home_team_id, -0.5)):
                    allow = _look_up_term(fit, fit.allow, opponent)
                    make = _look_up_term(fit, fit.make, opponent)
                    self.factors[(match.match_id, team)][count] = math.exp(
                        venue * fit.home - allow)  # fmt: skip
                    if count == "fouls":
                        self.factors[(match.match_id, team)]["fouls_drawn"] = math.exp(
                            make - venue * fit.home)  # fmt: skip

    def look_up_factor(self, match_id: int, team_id: int, term: str) -> float:
        return self.factors.get((match_id, team_id), {}).get(term, 1.0)


def applies_to_player(who: str, goalkeeper: bool) -> bool:
    """A key's players: everyone, the keepers only, or the outfield players only."""
    return who == "everyone" or (who == "keepers") == goalkeeper


def _look_up_term(fit: LeagueFit, values: np.ndarray, team: int) -> float:
    return float(values[fit.teams[team]]) if team in fit.teams else 0.0


def estimate_player_means(per_ninety: np.ndarray, ratio: float, terms: MatchTerms, match_id: int,
                          team_id: int, bin_minutes: np.ndarray) -> np.ndarray:  # fmt: skip
    """His expected count of each key in each minute bin (keys in the order of `KEYS`)."""
    found = np.zeros((len(KEYS), len(bin_minutes)))
    for index, threshold in enumerate(KEYS.values()):
        position = COUNTS.index(threshold.count)
        if threshold.term == "saves":
            per_match = ratio * terms.expected_goals.get((match_id, team_id), 1.35)
        else:
            per_match = per_ninety[position] * terms.look_up_factor(
                match_id, team_id, threshold.term
            )
        found[index] = per_match * bin_minutes / 90
    return found
