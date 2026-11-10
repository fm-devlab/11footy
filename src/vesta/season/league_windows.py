"""The streaks, the next k matches and the 30 days of the league clubs (T3, T12, T13), published
by the season job from the same simulated seasons as T1.

The verdicts: development and blind, 31 October 2026 (21 keys passed; `T12.clean_sheets.next3`,
`T12.clean_sheets.next5` and `T3.unbeaten_at_home` held, and the cups' scope of T12 waits for the
cups season job). Published every morning from 5 rounds on (31 October 2026), each key only
where the gate allows it.

A window needs the order of the team's matches. A match the world does not list yet (the season
job adds it with a negative id) comes after every listed one, in an order nobody knows; a listed
match whose kick-off has passed without a result (postponed, its new date unknown) could fall
anywhere. So, per team:

- `T12.<count>.next<k>` and the firsts: the team's next k (or ten, or to the season's end when
  fewer remain) matches must all be listed with a kick-off still to come, and none of its matches
  postponed; a window running past the season's end is not published.
- `T13.goals.next_30_days`: none of its matches postponed, and either all of them listed or the
  league's listed kick-offs reaching past the 30 days; its UEFA phase matches of the 30 days are
  drawn from the cups model at the publication.
- T3: every match the team has still to play listed with a kick-off to come, as the runs depend
  on the order; a quantity the matches played already settle (three defeats in a row) is
  published decided whatever the order.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from vesta.forecast_log.writing import PublishedDistribution
from vesta.jobs.fixtures import Fixture
from vesta.quantities import QuantityDefinition
from vesta.season.league import SeasonDraws, _find_decided_lines, _make_certain_total
from vesta.season.windows import (
    OUTCOME_NONE,
    count_window,
    find_first,
    find_longest_run,
    run_at_end,
)
from vesta.spine.score_matrix import SIZE

ROUNDS_FROM = 5
LEAGUE_K = (3, 5)
FIRSTS = 10
WINDOW_DAYS = 30
NEXT_KEYS = tuple(f"T12.{name}.next{k}" for k in LEAGUE_K
                  for name in ("points", "wins", "losses", "goals"))  # fmt: skip
FIRST_KEYS = {"T12.next_loss": "losses", "T12.next_clean_sheet": "clean_sheets"}
T13 = "T13.goals.next_30_days"
STREAK_KEYS = ("T3.longest_winning_streak", "T3.longest_unbeaten_run",
               "T3.loses_three_in_a_row")  # fmt: skip
KEYS = (*NEXT_KEYS, *FIRST_KEYS, T13, *STREAK_KEYS)


@dataclass(frozen=True)
class CupGame:
    """A UEFA phase match of a league club in the 30 days, with the cups model's matrix."""

    kickoff_utc: str
    home: bool  # the club plays at home
    matrix: np.ndarray


def is_publishable(teams: int, played: int) -> bool:
    """Whether the season has played 5 rounds' worth of matches."""
    return played >= ROUNDS_FROM * teams // 2


def find_window_dates(now: str, time_zone: str) -> tuple[str, str]:
    """The end instant of the 30 days and their `window_key` (local dates, both included)."""
    start = datetime.fromisoformat(now.replace("Z", "+00:00"))
    end = start + timedelta(days=WINDOW_DAYS)
    zone = ZoneInfo(time_zone)
    key = f"{start.astimezone(zone).date().isoformat()}/{end.astimezone(zone).date().isoformat()}"
    return end.isoformat().replace("+00:00", "Z"), key


def build_window_distributions(
    team_ids: tuple[int, ...],
    played: tuple[tuple[int, int, int, int], ...],
    fixtures: list[Fixture],
    draws: SeasonDraws,
    now: str,
    time_zone: str,
    versions: dict[str, str],
    definitions: dict[str, QuantityDefinition],
    cup_games: dict[int, list[CupGame]] | None = None,
    generator: np.random.Generator | None = None,
) -> list[PublishedDistribution]:
    """Every allowed key (`versions`: key -> the model version the gate registers) of every team,
    from the simulated seasons `draws` of the season job (its matches: `played` in kick-off order,
    then `fixtures`)."""
    matches = draws.matches
    if matches is None:
        raise ValueError("the simulated seasons carry no matches")
    end, calendar_key = find_window_dates(now, time_zone)
    listed_until = max((f.kickoff_utc for f in fixtures if f.match_id > 0), default="")
    offset = len(played)
    published = []
    for team in team_ids:
        own = sorted((f.kickoff_utc, f.match_id, offset + column, f.home_team_id == team)
                     for column, f in enumerate(fixtures)
                     if team in (f.home_team_id, f.away_team_id))  # fmt: skip
        if not own:
            continue
        dated = [entry for entry in own if entry[1] > 0 and entry[0] > now]
        postponed = any(entry[1] > 0 and entry[0] <= now for entry in own)
        unlisted = any(entry[1] < 0 for entry in own)
        goals_for, goals_against = _take_team_goals(matches, [entry[2:] for entry in dated])
        common = (team, versions, definitions)
        if not postponed:
            published += _build_next_matches(common, len(own), goals_for, goals_against)
            published += _build_firsts(common, len(own), len(dated), goals_for, goals_against)
            if T13 in versions and (not unlisted or listed_until > end):
                league = sum(1 for entry in dated if entry[0] <= end)
                published += _build_thirty_days(common, goals_for[:, :league],
                                                [game for game in (cup_games or {}).get(team, [])
                                                 if now < game.kickoff_utc <= end],
                                                calendar_key, generator)  # fmt: skip
        history = [(home == team, home_goals if home == team else away_goals,
                    away_goals if home == team else home_goals)
                   for home, away, home_goals, away_goals in played if team in (home, away)]
        published += _build_streaks(common, history, len(own), len(dated) == len(own), goals_for,
                                    goals_against)  # fmt: skip
    return published


def _take_team_goals(matches, columns: list[tuple[int, bool]]) -> tuple[np.ndarray, np.ndarray]:
    """(draws, n) the team's goals for and against in the matches of `columns` (column, home)."""
    draws = matches.home_goals.shape[0]
    if not columns:
        return np.zeros((draws, 0), dtype=np.int64), np.zeros((draws, 0), dtype=np.int64)
    home = matches.home_goals[:, [column for column, _ in columns]]
    away = matches.away_goals[:, [column for column, _ in columns]]
    at_home = np.array([side for _, side in columns])
    return np.where(at_home, home, away), np.where(at_home, away, home)


def _count(values: np.ndarray) -> dict:
    probabilities = (np.bincount(values) / len(values)).tolist()
    return {"probabilities": probabilities, "tail_from": len(probabilities), "tail": 0.0}


def _build_next_matches(
    common, remaining: int, goals_for, goals_against
) -> list[PublishedDistribution]:
    team, versions, _ = common
    published = []
    for window in LEAGUE_K:
        if remaining < window or goals_for.shape[1] < window:
            continue  # past the season's end, or not all listed
        counts = count_window(goals_for[:, :window], goals_against[:, :window])
        for name in ("points", "wins", "losses", "goals"):
            key = f"T12.{name}.next{window}"
            if key in versions:
                published.append(PublishedDistribution(
                    key, team, versions[key], _count(counts[name]),
                    window_key=f"next_matches={window}"))
    return published


def _build_firsts(common, remaining: int, listed: int, goals_for,
                  goals_against) -> list[PublishedDistribution]:  # fmt: skip
    team, versions, _ = common
    count = min(FIRSTS, remaining)
    if listed < count:
        return []
    happens = {"losses": goals_for[:, :count] < goals_against[:, :count],
               "clean_sheets": goals_against[:, :count] == 0}  # fmt: skip
    published = []
    for key, what in FIRST_KEYS.items():
        if key not in versions:
            continue
        first = find_first(happens[what])  # count where it never happens: the outcome "none"
        shares = np.bincount(np.minimum(first, count), minlength=count + 1) / len(first)
        names = [*(str(index + 1) for index in range(count)), OUTCOME_NONE]
        values = [[name, float(share)] for name, share in zip(names, shares, strict=True)]
        published.append(PublishedDistribution(key, team, versions[key], values,
                                               window_key=f"next_matches={FIRSTS}"))  # fmt: skip
    return published


def _build_thirty_days(
    common,
    league_for,
    cups: list[CupGame],
    calendar_key: str,
    generator: np.random.Generator | None,
) -> list[PublishedDistribution]:
    team, versions, _ = common
    if not league_for.shape[1] and not cups:
        return []
    goals = league_for.sum(axis=1)
    for game in cups:
        if generator is None:
            raise ValueError("a UEFA match in the 30 days needs a generator to draw it")
        cumulative = np.cumsum(game.matrix.ravel() / game.matrix.sum())
        cell = np.minimum(np.searchsorted(cumulative, generator.random(len(goals)), side="right"),
                          SIZE * SIZE - 1)  # fmt: skip
        goals = goals + (cell // SIZE if game.home else cell % SIZE)
    return [PublishedDistribution(T13, team, versions[T13], _count(goals),
                                  window_key=calendar_key)]


def _build_streaks(common, history: list[tuple[bool, int, int]], remaining: int, all_listed: bool,
                   goals_for, goals_against) -> list[PublishedDistribution]:  # fmt: skip
    team, versions, definitions = common
    played_for = np.array([own for _, own, _ in history], dtype=np.int64)
    played_against = np.array([other for _, _, other in history], dtype=np.int64)
    runs = {"T3.longest_winning_streak": played_for > played_against,
            "T3.longest_unbeaten_run": played_for >= played_against}
    lost = played_for < played_against
    published = []
    for key, so_far in runs.items():
        if key not in versions or not all_listed:
            continue
        drawn = goals_for > goals_against if key.endswith("streak") else goals_for >= goals_against
        values = find_longest_run(so_far, drawn)
        lowest = int(find_longest_run(so_far, np.zeros((1, 0), dtype=bool))[0])
        highest = max(lowest, run_at_end(so_far) + remaining)
        published.append(
            PublishedDistribution(
                key,
                team,
                versions[key],
                _count(values),
                decided_lines=_find_decided_lines(
                    definitions[key],
                    "count",
                    [_make_certain_total(total) for total in range(lowest, highest + 1)],
                ),
            )
        )
    key = "T3.loses_three_in_a_row"
    if key in versions:
        already = int(find_longest_run(lost, np.zeros((1, 0), dtype=bool))[0]) >= 3
        if already:
            values = {"yes": 1.0}
        elif all_listed:
            values = {"yes": float((find_longest_run(lost, goals_for < goals_against) >= 3).mean())}
        else:
            return published
        possible = [{"yes": 1.0}] if already else (
            [{"yes": 0.0}, {"yes": 1.0}] if run_at_end(lost) + remaining >= 3
            else [{"yes": 0.0}])  # fmt: skip
        published.append(PublishedDistribution(
            key, team, versions[key], values,
            decided_lines=_find_decided_lines(definitions[key], "binary", possible)))
    return published
