"""The scorers (their plan, 8cec816): a player's goals and assists
in a match if he plays, as his share of his side's expected goals.

A player's rates per 90 minutes (our expected goals without penalties, penalties taken, the
expected goals of the shots he set up) come from his league matches of this season and the one
before, each match back weighing `FADE` less, shrunk to his position group's rate by `STRENGTH`
nineties. Times his minutes if he plays (the lineups' distribution), they give his raw weight;
over the sum of the raw weights of his team's pool, each weighted by the chance to play, they
give his share of the side's goals, and the side's goals are the spine's (its score matrix's
mean, less the opponent's own goals at the league's rate). His goals are Poisson on that.

What happened: his goals of record (`match_events` goal and penalty goal, an own goal is nobody's),
his assists (the scored shot's assister, the goal's related player where the shot names none,
the feed's count where it is higher), and who scored the match's first goal.
"""

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from vesta.data_format import build_player_rows_filter
from vesta.expected_goals.dataset import is_dataset, read_rated_shot_players
from vesta.expected_goals.shots import SHOT_ORDER
from vesta.lineups.model import classify_position

FADE = 0.97  # each league match back
STRENGTH = 5.0  # nineties of the group's rate
REFERENCE_MATCHES = 20  # the reference: his last matches played
REFERENCE_STRENGTH = 5.0  # matches (goals per match) or team goals (the first scorer's share)


@dataclass(frozen=True, slots=True)
class Appearance:
    """A player's league match: what he did, for his rates and the reference."""

    known_at: str
    season: int
    minutes: float  # 90 or more counts 90
    expected_goals: float  # ours, without penalties
    penalties: int  # taken
    expected_assists: float  # the chances of the shots he set up
    goals: int
    assists: int
    team_goals: int
    covered: bool = True  # the match has the shots of the expected goals' source


@dataclass(frozen=True, slots=True)
class Happened:
    goals: int
    assists: int
    first: bool  # scored the match's first goal


def read_shot_players(world: sqlite3.Connection, competitions: list[str], first_season: int,
                      last_season: int, layers: list[str]) -> list[tuple]:  # fmt: skip
    """(shooter, assister, is penalty, team) of every shot `read_shots` reads, in its order; on
    a dataset, of every shot with `xg` (`vesta.expected_goals.dataset`)."""
    if is_dataset(world):
        return read_rated_shot_players(world, competitions, first_season, last_season)
    competition_marks = ", ".join("?" for _ in competitions)
    layer_marks = ", ".join("?" for _ in layers)
    return world.execute(
        f"""
        SELECT sh.player_id, sh.assist_player_id, COALESCE(sh.is_penalty, 0), sh.team_id
        FROM shots sh
        JOIN matches m ON m.match_id = sh.match_id
        JOIN seasons s ON s.season_id = m.season_id
        JOIN competitions c ON c.competition_id = s.competition_id
        WHERE c.code IN ({competition_marks}) AND s.season BETWEEN ? AND ?
          AND sh.layer IN ({layer_marks}) AND sh.x IS NOT NULL AND sh.y IS NOT NULL
        ORDER BY {SHOT_ORDER}
        """,
        (*competitions, first_season, last_season, *layers),
    ).fetchall()


def sum_shots_by_player(match_ids, chances, players) -> dict[tuple[int, int], list[float]]:
    """(match, player) -> [expected goals without penalties, penalties taken, expected
    assists]."""
    sums: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for match_id, chance, (shooter, assister, penalty, _) in zip(match_ids, chances, players,
                                                                strict=True):  # fmt: skip
        if shooter is not None:
            if penalty:
                sums[(int(match_id), shooter)][1] += 1
            else:
                sums[(int(match_id), shooter)][0] += float(chance)
        if assister is not None and not penalty:
            sums[(int(match_id), assister)][2] += float(chance)
    return sums


def read_happened(world: sqlite3.Connection, match_ids: set[int]
                  ) -> tuple[dict[tuple[int, int], Happened], dict[int, int]]:  # fmt: skip
    """(match, player) -> his goals, assists and whether he scored first; and per match the
    league's own goals (match -> count)."""
    goals: dict[tuple[int, int], int] = defaultdict(int)
    assists: dict[tuple[int, int], int] = defaultdict(int)
    first: dict[int, tuple] = {}
    own_goals: dict[int, int] = defaultdict(int)
    seen = set()
    for (match_id, kind, player, related, period, minute, added, sequence,
         assister) in _read_goal_rows(world, match_ids):  # fmt: skip
        if match_id not in match_ids:
            continue
        # one goal told by two sources is one goal
        mark = (match_id, kind == "own_goal", player, period, minute)
        if mark in seen:
            continue
        seen.add(mark)
        order = (period or 0, minute or 0, added, sequence)
        if match_id not in first or order < first[match_id][0]:
            first[match_id] = (order, None if kind == "own_goal" else player)
        if kind == "own_goal":
            own_goals[match_id] += 1
            continue
        if player is not None:
            goals[(match_id, player)] += 1
        maker = assister if assister is not None else (related if kind == "goal" else None)
        if maker is not None and maker != player:
            assists[(match_id, maker)] += 1
    for match_id, player, counted in _read_counted_assists(world, match_ids):
        if match_id in match_ids and counted > assists.get((match_id, player), 0):
            assists[(match_id, player)] = counted
    happened = {}
    for key in set(goals) | set(assists) | {(match, player) for match, (_, player)
                                            in first.items() if player is not None}:
        happened[key] = Happened(goals.get(key, 0), assists.get(key, 0),
                                 first.get(key[0], (None, None))[1] == key[1])  # fmt: skip
    return happened, dict(own_goals)


FEW_MATCHES = 2000  # up to this many, the goals are read by match (the check job's round)
GOAL_ROWS = ("SELECT e.match_id, e.kind, e.player_id, e.related_player_id, e.period, e.minute,"
             " COALESCE(e.added_minute, 0), COALESCE(e.sequence, 0), sh.assist_player_id"
             " FROM match_events e LEFT JOIN shots sh ON sh.shot_id = e.shot_id"
             " WHERE e.kind IN ('goal', 'penalty_goal', 'own_goal')")  # fmt: skip
GOAL_ORDER = " ORDER BY e.match_id, e.sequence, e.event_id"  # the match's order, not the ids'
ASSIST_ROWS = ("SELECT match_id, player_id, assists FROM player_match_statistics"
               " WHERE {rows} AND assists > 0")


def _read_goal_rows(world: sqlite3.Connection, match_ids: set[int]):
    """The goals of record: every one in the world (the verdicts' seasons), or by match in event
    order when the matches are few (the same rows, without reading the whole table)."""
    if len(match_ids) > FEW_MATCHES:
        return world.execute(GOAL_ROWS + GOAL_ORDER)
    rows = []
    chosen = sorted(match_ids)
    for start in range(0, len(chosen), 500):
        part = chosen[start:start + 500]
        rows += world.execute(GOAL_ROWS + f" AND e.match_id IN ({', '.join('?' * len(part))})"
                              + GOAL_ORDER, part).fetchall()  # fmt: skip
    return rows


def _read_counted_assists(world: sqlite3.Connection, match_ids: set[int]):
    if len(match_ids) > FEW_MATCHES:
        return world.execute(ASSIST_ROWS.format(rows=build_player_rows_filter(world)))
    rows = []
    chosen = sorted(match_ids)
    for start in range(0, len(chosen), 500):
        part = chosen[start:start + 500]
        query = ASSIST_ROWS.format(rows=build_player_rows_filter(world))
        rows += world.execute(query + f" AND match_id IN ({', '.join('?' * len(part))})",
                              part).fetchall()  # fmt: skip
    return rows


@dataclass(frozen=True)
class GroupRates:
    """A position group's rates (per 90, and the reference's per match played and share of the
    team's goals) from the seasons before."""

    expected_goals: float
    penalties: float
    expected_assists: float
    goals_per_match: float
    assists_per_match: float
    share: float


def group_rates(appearances: list[tuple[str | None, Appearance]]) -> dict[str | None, GroupRates]:
    """Each group's rates over the appearances given (group, appearance), every appearance of
    a player who played."""
    totals: dict[str | None, np.ndarray] = defaultdict(lambda: np.zeros(8))
    for group, item in appearances:
        totals[group] += (item.minutes / 90, item.expected_goals, item.penalties,
                          item.expected_assists, 1, item.goals, item.assists, item.team_goals)
    everyone = sum(totals.values(), np.zeros(8))
    found = {}
    for group, (nineties, xg, penalties, xa, matches, goals, assists, team) in [
            *totals.items(), (None, everyone)]:  # fmt: skip
        nineties = max(nineties, 1e-9)
        found[group] = GroupRates(xg / nineties, penalties / nineties, xa / nineties,
                                  goals / max(matches, 1), assists / max(matches, 1),
                                  goals / max(team, 1))  # fmt: skip
    return found


def estimate_own_rates(history: list[Appearance], season: int, rates: GroupRates
                       ) -> tuple[float, float, float]:  # fmt: skip
    """His rates per 90 (expected goals without penalties, penalties taken, expected assists)
    from his appearances of `season` and the one before (newest last), shrunk to his group's; a
    match without the source's shots is left out, not read as played with no chance."""
    chosen = [item for item in history if item.season >= season - 1 and item.covered]
    weights = FADE ** np.arange(len(chosen))[::-1] if chosen else np.zeros(0)
    nineties = float(np.dot(weights, [item.minutes / 90 for item in chosen]))
    xg = float(np.dot(weights, [item.expected_goals for item in chosen]))
    penalties = float(np.dot(weights, [item.penalties for item in chosen]))
    xa = float(np.dot(weights, [item.expected_assists for item in chosen]))
    return ((xg + STRENGTH * rates.expected_goals) / (nineties + STRENGTH),
            (penalties + STRENGTH * rates.penalties) / (nineties + STRENGTH),
            (xa + STRENGTH * rates.expected_assists) / (nineties + STRENGTH))  # fmt: skip


def estimate_reference_rates(
    history: list[Appearance], rates: GroupRates
) -> tuple[float, float, float]:
    """The reference: his goals and assists per match played over his last matches, and his
    share of his team's goals in them, each shrunk to his group's."""
    last = history[-REFERENCE_MATCHES:]
    played = len(last)
    goals = sum(item.goals for item in last)
    assists = sum(item.assists for item in last)
    team = sum(item.team_goals for item in last)
    prior_weight = REFERENCE_STRENGTH
    return ((goals + prior_weight * rates.goals_per_match) / (played + prior_weight),
            (assists + prior_weight * rates.assists_per_match) / (played + prior_weight),
            (goals + prior_weight * rates.share) / (team + prior_weight))  # fmt: skip


def split_first_goal(matrix: np.ndarray) -> tuple[float, float]:
    """The chances the home side and the away side score the match's first goal: given the
    score, each goal of it is equally likely to be the first."""
    home = np.arange(matrix.shape[0])[:, None]
    away = np.arange(matrix.shape[1])[None, :]
    total = home + away
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(total > 0, home / np.maximum(total, 1), 0.0)
    scored = matrix.sum() - matrix[0, 0]
    first_home = float((matrix * share).sum())
    return first_home, float(scored - first_home)


def compute_side_means(matrix: np.ndarray) -> tuple[float, float]:
    """The expected goals of the home and the away side read from the score matrix."""
    return (float(np.arange(matrix.shape[0]) @ matrix.sum(axis=1)),
            float(np.arange(matrix.shape[1]) @ matrix.sum(axis=0)))  # fmt: skip


def normalise_shares(raw: np.ndarray, plays: np.ndarray) -> np.ndarray:
    """Each player's share of his side's goals if he plays: his raw weight over the sum of the
    pool's, each weighted by the chance to play (so the shares weighted add up to one)."""
    total = float(np.dot(plays, raw))
    if total <= 0:
        return np.full(len(raw), 1.0 / max(float(plays.sum()), 1e-9))
    return raw / total


def make_poisson_cells(mean: float, cells: int = 4) -> np.ndarray:
    """0, 1, ... and the last cell `cells - 1` or more."""
    values = [math.exp(-mean) * mean ** k / math.factorial(k) for k in range(cells - 1)]
    return np.array([*values, max(1.0 - sum(values), 0.0)])


@dataclass(frozen=True)
class SideExpectations:
    """A side's players if they play: their expected goals and assists, and the chance each
    scores the match's first goal."""

    goals: np.ndarray
    assists: np.ndarray
    first: np.ndarray
    side_goals: float  # the side's expected goals, own goals of the opponent included
    gap: float  # the coherence test: the pool's goals weighted by the chance to play, relative


def share_out_side_goals(raw_goals: np.ndarray, raw_assists: np.ndarray, plays: np.ndarray,
                         matrix: np.ndarray, home: bool, league: dict,
                         ) -> SideExpectations:  # fmt: skip
    """The side's goals from the score matrix shared by its players' raw weights (each weighted
    by the chance to play), less the opponent's own goals at the league's rate; the assists the
    same way on the league's assists per goal; the first goal as each player's part of the
    side's goals times the chance the side scores first."""
    means = compute_side_means(matrix)
    firsts = split_first_goal(matrix)
    side_goals = means[0] if home else means[1]
    players_goals = side_goals * (1 - league["own_goal"])
    goals = normalise_shares(raw_goals, plays) * players_goals
    assists = normalise_shares(raw_assists, plays) * players_goals * league["assisted"]
    first = goals / max(side_goals, 1e-12) * (firsts[0] if home else firsts[1])
    gap = abs(float(plays @ goals) - players_goals) / max(players_goals, 1e-12)
    return SideExpectations(goals, assists, first, side_goals, gap)


def weigh_player(history: list[Appearance], season: int, rates: GroupRates, conversion: float,
                 expected_minutes: float) -> tuple[float, float]:  # fmt: skip
    """A player's raw weights of goals and assists: his rates per 90 (penalties at the league's
    conversion) times his expected minutes if he plays."""
    xg, penalties, xa = estimate_own_rates(history, season, rates)
    return ((xg + penalties * conversion) * expected_minutes / 90,
            xa * expected_minutes / 90)  # fmt: skip


def estimate_league_rates(
    matches, assists_by_match, own_goals, firsts, conversion
) -> dict[str, dict]:
    """Each league's rates over the matches given (the seasons before): own goals among all
    goals, assists per goal of a player, penalties scored per penalty taken, and how often each
    side's player scores the first goal."""
    totals = defaultdict(lambda: np.zeros(6))
    for match in matches:
        side = firsts.get(match.match_id)
        totals[match.competition] += (match.home_goals + match.away_goals,
                                      own_goals.get(match.match_id, 0),
                                      assists_by_match.get(match.match_id, 0),
                                      side == "home", side == "away", 1)  # fmt: skip
    return {league: {"own_goal": own / max(goals, 1),
                     "assisted": assists / max(goals - own, 1),
                     "conversion": conversion.get(league, 0.78),
                     "home_first": home / max(count, 1),
                     "away_first": away / max(count, 1)}
            for league, (goals, own, assists, home, away, count) in totals.items()}  # fmt: skip


def read_penalty_conversion(connection, leagues: list[str], before: int) -> dict[str, float]:
    """Penalties scored per penalty taken, per league, in the seasons before `before` (fixed
    once: the rate barely moves)."""
    marks = ", ".join("?" for _ in leagues)
    found = {}
    for code, taken, scored in connection.execute(
            f"SELECT c.code, COUNT(*), SUM(sh.result = 'goal') FROM shots sh JOIN matches m ON"
            f" m.match_id = sh.match_id JOIN seasons s ON s.season_id = m.season_id JOIN"
            f" competitions c ON c.competition_id = s.competition_id WHERE c.code IN ({marks})"
            f" AND s.season < ? AND sh.layer = 'tracked' AND sh.is_penalty = 1 GROUP BY 1",
            (*leagues, before)):  # fmt: skip
        found[code] = scored / taken if taken else 0.78
    return found


def build_histories(world, by_id, sums, happened, covered: set[int] | None = None):
    """Each player's league appearances in time order, and the position group of each (his
    position if he started, else the last group he started in); with `covered`, the matches
    with the source's shots, each appearance says whether its match has them."""
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
    for match in items:
        league_match = by_id[match.match_id]
        for side, goals in ((league_match.home_team_id, league_match.home_goals),
                            (league_match.away_team_id, league_match.away_goals)):  # fmt: skip
            for player, (started, minutes, code) in world.sheets.get(
                    (match.match_id, side), {}).items():  # fmt: skip
                if not (started or minutes > 0):
                    continue
                xg, penalties, xa = sums.get((match.match_id, player), (0.0, 0.0, 0.0))
                did = happened.get((match.match_id, player))
                group = classify_position(code) if started and code else None
                group = group or last_group.get(player)
                if started and code:
                    last_group[player] = group
                groups_of[(player, len(histories[player]))] = group
                histories[player].append(Appearance(
                    match.known_at, match.season, min(minutes, 90), xg, int(penalties), xa,
                    did.goals if did else 0, did.assists if did else 0, goals,
                    covered is None or match.match_id in covered))  # fmt: skip
    return dict(histories), groups_of


def find_first_scoring_side(match, happened, world) -> str | None:
    """Which side's player scored the match's first goal (None: no goal, or an own goal)."""
    for side, name in ((match.home_team_id, "home"), (match.away_team_id, "away")):
        for player in world.sheets.get((match.match_id, side), {}):
            did = happened.get((match.match_id, player))
            if did and did.first:
                return name
    return None
