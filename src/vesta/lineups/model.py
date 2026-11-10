"""The lineups' model, candidate 1 (its plan, 3bf2e5d).

For a team's league match, every player of its sheets of the last 365 days (less those whose
latest sheet before the forecast is another team's) gets a chance to start from a logistic
regression on what his recent sheets say, fitted walk-forward on the five seasons before; one
shift of the log-odds for the goalkeepers and one for the others make the chances add up to one
goalkeeper and ten others. His chance to play adds the chance of coming on when he does not start;
his minutes mix his minutes when he starts and when he comes on. The formation and his position
(if he starts) are the recent ones, each match back weighing `FADE` of the one after it, shrunk
to the league's and his position group's frequencies.

A departure from the plan, decided before any run: `squad_memberships` is derived with end dates
written after the fact, so the pool comes from the sheets alone. A player a ban of the rules keeps
out of the match (`suspensions_by_rule`, built from our cards, 2 November 2026) is
left out of its pool: he neither starts nor plays (2 November 2026), in the
leagues `config/vesta/bans.toml` names only, those whose bans passed their check. The days
to the team's next match are left out: the world holds the next match's final kick-off, not the
schedule known at the forecast.
"""

import bisect
import math
import sqlite3
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vesta.spine.dixon_coles import parse_instant_to_seconds
from vesta.spine.timing import TimeRules, find_result_known_at, forecast_time

FADE = 0.8
HISTORY = 10  # team matches read back
POOL_DAYS = 365
OTHER = "other"
MINUTE_CELLS = 91  # 0 to 89 minutes, then 90 or more
FEATURES = ("started_1", "started_3", "started_5", "started_10", "minutes_1", "came_on_1",
            "missed", "back", "goalkeeper", "rest_days", "cup_before", "season_share")  # fmt: skip


def classify_position(code: str | None) -> str | None:
    if code is None or code in ("SUB", ""):
        return None
    if code == "G":
        return "goalkeeper"
    if code in ("D", "SW", "DC") or code.startswith(("CD", "LB", "RB")):
        return "defender"
    if code in ("F", "FW", "LF", "RF", "RCF") or code.startswith("CF"):
        return "forward"
    return "midfielder"


@dataclass(frozen=True, slots=True)
class TeamMatch:
    match_id: int
    kickoff: str
    competition: str
    is_league: bool
    season: int
    forecast_time: str
    known_at: str
    seconds: float  # the kick-off's


@dataclass(frozen=True, slots=True)
class Row:
    """A player of a team's league match: what was known before, and what happened."""

    match_id: int
    team_id: int
    player_id: int
    competition: str
    season: int
    forecast_time: str
    features: tuple[float, ...]
    state: str  # in the team's last match
    goalkeeper: bool
    group: str | None
    came_on_rate: tuple[int, int]  # came on, did not start (in the history)
    start_minutes: tuple[tuple[int, float], ...]  # (minutes, weight) of his recent starts
    bench_minutes: tuple[tuple[int, float], ...]  # (minutes, weight) when he came on
    positions: tuple[tuple[str, float], ...]  # (position, weight) of his recent starts
    last_position: str | None
    started: bool
    minutes: int
    position: str | None  # if he started


@dataclass(frozen=True, slots=True)
class TeamTarget:
    match_id: int
    team_id: int
    competition: str
    season: int
    forecast_time: str
    formations: tuple[tuple[str, float], ...]  # the team's recent formations, weighted
    last_formation: str | None
    formation: str | None  # what happened


@dataclass
class World:
    team_matches: dict[int, list[TeamMatch]]
    sheets: dict[tuple[int, int], dict[int, tuple[bool, int, str | None]]]
    formations: dict[tuple[int, int], str | None]
    appearances: dict[int, list[tuple[str, int]]]  # player -> (kickoff, team), in time order
    # match -> the players a ban of the rules keeps out of it (`suspensions_by_rule`), and
    # whether the world holds the bans at all (an empty table is no source)
    bans: dict[int, frozenset[int]] = field(default_factory=dict)
    bans_read: bool = False


BANS_FILE = Path(__file__).resolve().parents[3] / "config/vesta/bans.toml"


def load_ban_competitions() -> frozenset[str]:
    """The leagues whose lineups apply the bans (`config/vesta/bans.toml`)."""
    return frozenset(tomllib.loads(BANS_FILE.read_text(encoding="utf-8"))["competitions"])


def read_bans(world: sqlite3.Connection, competitions: frozenset[str] | None = None
              ) -> tuple[dict[int, frozenset[int]], bool]:  # fmt: skip
    """match -> the players banned from it by the rules (every certainty: the matches a row names
    are the ban's minimum), and whether the world holds any ban; none without the table. With
    `competitions`, only the bans of those leagues (none at all when it is empty); without, every
    league's (the checks)."""
    if competitions is not None and not competitions:
        return {}, False
    if world.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name ="
                     " 'suspensions_by_rule'").fetchone() is None:  # fmt: skip
        return {}, False
    found: dict[int, set[int]] = defaultdict(set)
    if competitions is None:
        rows = world.execute("SELECT player_id, served_in_match_ids FROM suspensions_by_rule"
                             ).fetchall()  # fmt: skip
    else:
        marks = ", ".join("?" for _ in competitions)
        rows = world.execute(
            "SELECT b.player_id, b.served_in_match_ids FROM suspensions_by_rule b JOIN"
            f" competitions c ON c.competition_id = b.competition_id WHERE c.code IN ({marks})",
            sorted(competitions)).fetchall()  # fmt: skip
    for player_id, served in rows:
        for match_id in (served or "").split(","):
            if match_id.strip():
                found[int(match_id)].add(player_id)
    return {match_id: frozenset(players) for match_id, players in found.items()}, bool(rows)


def read_world(world: sqlite3.Connection, first_season: int, last_season: int,
               rules: TimeRules) -> World:  # fmt: skip
    """Every played match of a club in the seasons, every sheet and formation."""
    team_matches: dict[int, list[TeamMatch]] = defaultdict(list)
    kickoffs: dict[int, str] = {}
    for (match_id, home, away, kickoff, precision, local_date, code, kind,
         season) in world.execute(
            "SELECT m.match_id, m.home_team_id, m.away_team_id, m.kickoff_utc,"
            " m.kickoff_precision, m.local_date, c.code, c.kind, s.season FROM matches m"
            " JOIN seasons s ON s.season_id = m.season_id JOIN competitions c ON"
            " c.competition_id = s.competition_id JOIN teams h ON h.team_id = m.home_team_id"
            " WHERE s.season BETWEEN ? AND ? AND m.status = 'played' AND h.kind = 'club'"
            " ORDER BY m.kickoff_utc, m.match_id", (first_season, last_season)):  # fmt: skip
        item = TeamMatch(match_id, kickoff, code, kind == "league", season,
                         forecast_time(kickoff, precision, local_date, rules),
                         find_result_known_at(kickoff, precision, local_date, rules),
                         parse_instant_to_seconds(kickoff))  # fmt: skip
        team_matches[home].append(item)
        team_matches[away].append(item)
        kickoffs[match_id] = kickoff
    sheets: dict[tuple[int, int], dict[int, tuple[bool, int, str | None]]] = defaultdict(dict)
    appearances: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for match_id, team_id, player_id, started, minutes, code in world.execute(
            "SELECT p.match_id, p.team_id, p.player_id, p.started, p.minutes_played,"
            " p.position_code FROM match_players p JOIN matches m ON m.match_id = p.match_id"
            " JOIN seasons s ON s.season_id = m.season_id WHERE s.season BETWEEN ? AND ?",
            (first_season, last_season)):  # fmt: skip
        if match_id not in kickoffs:
            continue
        sheets[(match_id, team_id)][player_id] = (bool(started), int(minutes or 0),
                                                  code if started else None)  # fmt: skip
        if started or (minutes or 0) > 0:
            appearances[player_id].append((kickoffs[match_id], team_id))
    for found in appearances.values():
        found.sort()
    formations = {(match_id, team_id): formation for match_id, team_id, formation in world.execute(
        "SELECT t.match_id, t.team_id, t.formation FROM match_team_sheets t JOIN matches m ON"
        " m.match_id = t.match_id JOIN seasons s ON s.season_id = m.season_id"
        " WHERE s.season BETWEEN ? AND ?", (first_season, last_season))}  # fmt: skip
    bans, bans_read = read_bans(world, load_ban_competitions())
    return World(dict(team_matches), dict(sheets), formations, dict(appearances), bans, bans_read)


def collect_targets(world: World, leagues: set[str], seasons: set[int], lag: int = 0
                    ) -> tuple[list[Row], list[TeamTarget]]:  # fmt: skip
    """The rows of every league match of `seasons` in `leagues` whose sheet exists; with `lag`,
    the history ends that many team matches earlier (the alarm check's test)."""
    rows: list[Row] = []
    teams: list[TeamTarget] = []
    for team_id, played in world.team_matches.items():
        for index, match in enumerate(played):
            if not (match.is_league and match.competition in leagues
                    and match.season in seasons):  # fmt: skip
                continue
            sheet = world.sheets.get((match.match_id, team_id))
            if not sheet or sum(started for started, _, _ in sheet.values()) != 11:
                continue
            # a year of matches is about 60: the last 120 hold the pool and the history
            before = [item for item in played[max(0, index - 120):index]
                      if item.known_at <= match.forecast_time
                      and (item.match_id, team_id) in world.sheets]  # fmt: skip
            before = before[:len(before) - lag]
            if not before:
                continue
            recent = before[::-1][:HISTORY]  # newest first
            weights = [FADE ** age for age in range(len(recent))]
            formations: dict[str, float] = defaultdict(float)
            for item, weight in zip(recent, weights, strict=False):
                formation = world.formations.get((item.match_id, team_id))
                if formation:
                    formations[formation] += weight
            teams.append(TeamTarget(match.match_id, team_id, match.competition, match.season,
                                    match.forecast_time, tuple(formations.items()),
                                    world.formations.get((recent[0].match_id, team_id)),
                                    world.formations.get((match.match_id, team_id))))  # fmt: skip
            rows += build_player_rows(world, team_id, match, before, recent, weights, sheet)
    return rows, teams


def build_player_rows(world, team_id, match, before, recent, weights, sheet) -> list[Row]:
    window = [item for item in before if match.seconds - item.seconds <= POOL_DAYS * 86_400]
    pool = {player for item in window for player in world.sheets[(item.match_id, team_id)]}
    season_matches = [item for item in before if item.season == match.season]
    rest = (match.seconds - recent[0].seconds) / 86_400
    found = []
    banned = world.bans.get(match.match_id, frozenset())
    for player in pool:
        if player in banned:
            continue  # a ban of the rules keeps him out: he neither starts nor plays
        seen = world.appearances.get(player, [])
        cut = bisect.bisect_left(seen, (match.kickoff, -1))
        if cut and seen[cut - 1][1] != team_id:
            continue  # his latest sheet before this match is another team's
        lines = [world.sheets[(item.match_id, team_id)].get(player) for item in recent]
        started = [line is not None and line[0] for line in lines]
        played = [line is not None and (line[0] or line[1] > 0) for line in lines]
        missed = 0
        for was in played:
            if was:
                break
            missed += 1
        positions: dict[str, float] = defaultdict(float)
        start_minutes, bench_minutes = [], []
        for line, weight in zip(lines, weights, strict=False):
            if line is None:
                continue
            if line[0]:
                start_minutes.append((min(line[1], MINUTE_CELLS - 1), weight))
                if line[2]:
                    positions[line[2]] += weight
            elif line[1] > 0:
                bench_minutes.append((min(line[1], MINUTE_CELLS - 1), weight))
        last_position = next((line[2] for line in lines if line is not None and line[0]
                              and line[2]), None)  # fmt: skip
        group = classify_position(last_position) or _find_usual_group(
            world, player, team_id, before
        )
        share = (sum(world.sheets[(item.match_id, team_id)].get(player, (0, 0, None))[1]
                     for item in season_matches) / (90 * len(season_matches))
                 if season_matches else 0.0)  # fmt: skip
        last = lines[0]
        state = ("started" if last and last[0] else
                 "came_on" if last and last[1] > 0 else "absent")  # fmt: skip
        features = (
            float(started[0]), sum(started[:3]) / 3, sum(started[:5]) / 5,
            sum(started) / HISTORY, (last[1] if last else 0) / 90,
            float(state == "came_on"), float(min(missed, 5)),
            float(missed > 0 and sum(started[missed:missed + 5]) >= 3),
            float(group == "goalkeeper"), math.log1p(max(rest, 0.0)),
            float(not recent[0].is_league), min(share, 1.0))  # fmt: skip
        happened = sheet.get(player, (False, 0, None))
        not_started = sum(1 for was in started if not was)
        came_on = sum(1 for was, play in zip(started, played, strict=True) if play and not was)
        found.append(Row(match.match_id, team_id, player, match.competition, match.season,
                         match.forecast_time, features, state, group == "goalkeeper", group,
                         (came_on, not_started), tuple(start_minutes), tuple(bench_minutes),
                         tuple(positions.items()), last_position, happened[0], happened[1],
                         happened[2] if happened[0] else None))  # fmt: skip
    return found


def _find_usual_group(world, player, team_id, before) -> str | None:
    for item in reversed(before[-60:]):
        line = world.sheets[(item.match_id, team_id)].get(player)
        if line is not None and line[0] and line[2]:
            return classify_position(line[2])
    return None


def fit_logistic(design: np.ndarray, y: np.ndarray, ridge: float = 1.0,
                 iterations: int = 25) -> np.ndarray:  # fmt: skip
    """Weights (intercept first) by Newton's method with a ridge on the non-intercept terms."""
    x = np.column_stack([np.ones(len(design)), design])
    beta = np.zeros(x.shape[1])
    penalty = np.full(x.shape[1], ridge)
    penalty[0] = 0.0
    for _ in range(iterations):
        fitted = 1.0 / (1.0 + np.exp(-(x @ beta)))
        gradient = x.T @ (y - fitted) - penalty * beta
        hessian = (x * (fitted * (1 - fitted))[:, None]).T @ x + np.diag(penalty)
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def scale_to_eleven(log_odds: np.ndarray, goalkeepers: np.ndarray) -> np.ndarray:
    """The chances to start, shifted (one shift for the goalkeepers, one for the others) so that
    they add up to one goalkeeper and ten others; a group too small keeps every chance at 1."""
    chances = np.empty(len(log_odds))
    for mask, wanted in ((goalkeepers, 1.0), (~goalkeepers, 10.0)):
        if not mask.any():
            continue
        odds = log_odds[mask]
        if mask.sum() <= wanted:
            chances[mask] = 1.0
            continue
        low, high = -50.0, 50.0
        for _ in range(100):
            middle = (low + high) / 2
            if (1.0 / (1.0 + np.exp(-(odds + middle)))).sum() > wanted:
                high = middle
            else:
                low = middle
        chances[mask] = 1.0 / (1.0 + np.exp(-(odds + (low + high) / 2)))
    return chances


def build_histogram(pairs, cells: int) -> np.ndarray:
    found = np.zeros(cells)
    for value, weight in pairs:
        found[value] += weight
    return found


def shrink_to_prior(own: np.ndarray, prior: np.ndarray, strength: float) -> np.ndarray:
    """`own` (weighted counts) pulled toward `prior` (a distribution) by `strength` cases."""
    total = own + strength * prior
    return total / total.sum()
