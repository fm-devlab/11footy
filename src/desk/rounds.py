"""A competition's rounds: every matchday of the season, the Matches screen moving between them.

The screen opens on the current round and shows every match of it, with its last forecast
before kick-off, including matches already started or finished; it never drops a match at its
kick-off. The current round is the one in progress, from its first kick-off until every match of
it is finished, else the next one: once a round is over the next one is current, at once. A
postponed, cancelled or abandoned match stays in its own round, marked, and does not keep the
round going; nor does a match moved far from the others (`CORE_DAYS`). When the clock has passed
the first kick-off of a later round, that round is current even if the dataset (refreshed after
each round, never live) lacks results of an earlier one.

A round is, in this order of preference:
- a league's matchday, or a cup's stage with its matchday ("league phase, matchday 4");
- a cup's stage with its leg ("round of 16, first leg");
- a stage without matchdays or legs (a nations' group stage, a final): its matches grouped by
  their dates, as below, labelled with the stage, and with the dates when the stage makes
  several rounds ("Group stage, Thu 24/09/2026 to Sat 26/09/2026");
- where the dataset gives no matchday and no stage (some leagues, the nations' windows): the
  dates, matches grouped while no gap between two kick-offs is longer than `DATE_GAP` (a
  league weekend, a midweek round) or `WINDOW_GAP` for nations and friendlies (an international
  window), and a new round begun when a team plays a second time (a window's next matchday).

A match without a matchday in a stage (or a league) that has matchdays joins the matchday whose
dates hold it, or the nearest one within `CORE_DAYS`; only one far from every matchday is
grouped by its dates.

A round starts at its first kick-off, leaving aside a match moved far from the others (one
played weeks early or postponed): the start is the first kick-off within `CORE_DAYS` of the
round's middle kick-off, and those core matches are the ones that end it.
"""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from desk.formats import parse_instant

DATE_GAP = timedelta(hours=36)
WINDOW_GAP = timedelta(days=5)
CORE_DAYS = timedelta(days=3)
# a match still unplayed in the dataset this long after its kick-off is shown as played, its
# result not yet in the dataset (the dataset is refreshed after each round, never a live feed)
MATCH_LENGTH = timedelta(hours=2, minutes=15)
LEGS = {1: "first leg", 2: "second leg"}
CALLED_OFF = ("postponed", "cancelled", "abandoned", "suspended")
# the statuses of a match whose result stands: the dataset's words, and other datasets' common
# ones
FINISHED = ("played", "awarded", "finished", "ft", "aet", "pen")


@dataclass
class Round:
    label: str
    matches: list[sqlite3.Row] = field(default_factory=list)  # in kick-off order
    start: datetime | None = None


def read_round_columns(row: sqlite3.Row) -> tuple[str | None, int | None, int | None]:
    """A match's stage, matchday and leg, each None where the dataset has no such column."""
    keys = row.keys()
    return tuple(row[name] if name in keys else None for name in ("stage", "matchday", "leg"))


def make_round_key(row: sqlite3.Row) -> tuple | None:
    """The round a match belongs to by its own columns: its matchday, else its stage's leg;
    None when it has neither (its stage's matchdays or its dates decide)."""
    stage, matchday, leg = read_round_columns(row)
    if matchday is None and leg not in LEGS:
        return None
    return row["season_id"], (stage or "").casefold(), matchday, leg if matchday is None else None


def read_stage(row: sqlite3.Row) -> tuple:
    stage = read_round_columns(row)[0]
    return row["season_id"], (stage or "").casefold()


def label_round(row: sqlite3.Row) -> str:
    """The round in words: "Matchday 10", "League phase, matchday 4", "Final, first leg"."""
    stage, matchday, leg = read_round_columns(row)
    parts = [stage[:1].upper() + stage[1:]] if stage else []
    if matchday is not None:
        parts.append(f"matchday {matchday}" if parts else f"Matchday {matchday}")
    elif leg in LEGS:
        parts.append(LEGS[leg])
    return ", ".join(parts)


def group_rounds(
    rows: list[sqlite3.Row], gap: timedelta, date_label: Callable[[str], str]
) -> list[Round]:
    """The rounds of a competition's matches (rows with `kickoff_utc`), each with its start."""
    by_key: dict[tuple, Round] = {}
    loose: dict[tuple, list[sqlite3.Row]] = {}  # by stage, the matches without a round
    for row in sorted(rows, key=lambda row: (row["kickoff_utc"], row["match_id"])):
        key = make_round_key(row)
        if key is None:
            loose.setdefault(read_stage(row), []).append(row)
            continue
        by_key.setdefault(key, Round(label_round(row))).matches.append(row)
    matchdays: dict[tuple, list[tuple[Round, tuple[datetime, datetime]]]] = {}
    for key, found in by_key.items():
        if key[2] is not None:
            matchdays.setdefault(key[:2], []).append((found, find_window(found.matches)))
    rounds = list(by_key.values())
    for stage, stage_rows in loose.items():
        undated = []
        for row in stage_rows:
            home = find_nearest_matchday(row, matchdays.get(stage, []))
            if home is None:
                undated.append(row)
            else:
                home.matches.append(row)
        groups = group_by_dates(undated, gap)
        named = read_round_columns(undated[0])[0] if undated else None
        for group in groups:
            rounds.append(make_dated_round(group, date_label, named, len(groups) > 1))
    for found in rounds:
        found.matches.sort(key=lambda row: (row["kickoff_utc"], row["match_id"]))
        found.start = find_round_start(found.matches)
    return sorted(rounds, key=lambda found: found.start)


def find_window(rows: list[sqlite3.Row]) -> tuple[datetime, datetime]:
    """The first and last kick-off of a round's core matches."""
    kickoffs = [parse_instant(row["kickoff_utc"]) for row in find_core(rows)]
    return min(kickoffs), max(kickoffs)


def find_nearest_matchday(
    row: sqlite3.Row, candidates: list[tuple[Round, tuple[datetime, datetime]]]
) -> Round | None:
    """The matchday whose dates hold the match, else the nearest within `CORE_DAYS`."""
    kickoff = parse_instant(row["kickoff_utc"])
    best, distance = None, CORE_DAYS
    for found, (first, last) in candidates:
        away = max(first - kickoff, kickoff - last, timedelta(0))
        if away < distance or (best is None and away == distance):
            best, distance = found, away
    return best


def group_by_dates(rows: list[sqlite3.Row], gap: timedelta) -> list[list[sqlite3.Row]]:
    """Matches (in kick-off order) grouped while no gap between two kick-offs is longer than
    `gap` and no team plays twice."""
    groups: list[list[sqlite3.Row]] = []
    group: list[sqlite3.Row] = []
    teams: set = set()
    for row in rows:
        playing = read_teams(row)
        if group:
            apart = parse_instant(row["kickoff_utc"]) - parse_instant(group[-1]["kickoff_utc"])
            if apart > gap or teams & playing:
                groups.append(group)
                group, teams = [], set()
        group.append(row)
        teams |= playing
    if group:
        groups.append(group)
    return groups


def read_teams(row: sqlite3.Row) -> set:
    """The match's two teams; none where the row has no team columns."""
    keys = row.keys()
    return {row[name] for name in ("home_team_id", "away_team_id")
            if name in keys and row[name] is not None}  # fmt: skip


def make_dated_round(
    rows: list[sqlite3.Row],
    date_label: Callable[[str], str],
    stage: str | None = None,
    dated: bool = True,
) -> Round:
    """A round of matches grouped by their dates: its stage, with the dates when the stage makes
    several rounds; the dates alone without a stage."""
    first, last = rows[0]["kickoff_utc"], rows[-1]["kickoff_utc"]
    dates = date_label(first)
    if first[:10] != last[:10]:
        dates += f" to {date_label(last)}"
    if not stage:
        return Round(dates, list(rows))
    named = stage[:1].upper() + stage[1:]
    return Round(f"{named}, {dates}" if dated else named, list(rows))


def find_round_start(rows: list[sqlite3.Row]) -> datetime:
    return min(parse_instant(row["kickoff_utc"]) for row in find_core(rows))


def find_core(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
    """The round's matches within `CORE_DAYS` of its middle kick-off: a match played weeks early
    or moved weeks later stays in the round but neither starts it nor ends it."""
    kickoffs = sorted(parse_instant(row["kickoff_utc"]) for row in rows)
    middle = kickoffs[len(kickoffs) // 2]
    return [row for row in rows if abs(parse_instant(row["kickoff_utc"]) - middle) <= CORE_DAYS]


def find_current_index(rounds: list[Round], now: datetime) -> int | None:
    """The current round's place in `rounds` (in start order); None without rounds. After the
    season's last round, the last one."""
    if not rounds:
        return None
    started = [index for index, found in enumerate(rounds) if found.start <= now]
    first = started[-1] if started else 0
    open_ = (index for index in range(first, len(rounds)) if not is_round_over(rounds[index]))
    return next(open_, len(rounds) - 1)


def is_round_over(found: Round) -> bool:
    """Every core match of the round finished, or postponed, cancelled, abandoned or
    suspended."""
    return all(is_finished(row) or is_called_off(row) for row in find_core(found.matches))


def is_finished(row: sqlite3.Row) -> bool:
    """Whether the dataset holds the match's result: a finished status ("played", "awarded")
    and both goals. A postponed match the dataset stores at 0-0 is not finished."""
    status = (row["status"] or "").casefold()
    return row["home_goals"] is not None and row["away_goals"] is not None and status in FINISHED


def is_called_off(row: sqlite3.Row) -> bool:
    """A match postponed, cancelled, abandoned or suspended: it does not keep its round going."""
    return (row["status"] or "").casefold() in CALLED_OFF


def describe_status(row: sqlite3.Row, now: datetime) -> str:
    """The match's status as the dataset and the clock say it: the dataset is a snapshot, so a
    match it has not seen played is "in play" from its kick-off and "played" after, its result
    in a later dataset. A match called off whose kick-off is still to come was moved there: it
    is to play."""
    status = (row["status"] or "").casefold()
    if is_finished(row):
        return f"Finished {row['home_goals']}-{row['away_goals']}{describe_ending(row)}"
    kickoff = parse_instant(row["kickoff_utc"])
    if now < kickoff:
        return "To play"
    if status in CALLED_OFF:
        return status.capitalize()
    if now < kickoff + MATCH_LENGTH:
        return "In play"
    return "Played, no result yet"


def describe_ending(row: sqlite3.Row) -> str:
    """How a finished match ended past its 90 minutes: " (4-3 pens)" after a shootout, " a.e.t."
    after extra time without one; nothing otherwise, or where the dataset has no such columns."""
    keys = row.keys()
    if "shootout_home_goals" in keys and "shootout_away_goals" in keys:
        home, away = row["shootout_home_goals"], row["shootout_away_goals"]
        if home is not None and away is not None:
            return f" ({home}-{away} pens)"
    if "extra_time_played" in keys and row["extra_time_played"]:
        return " a.e.t."
    return ""
