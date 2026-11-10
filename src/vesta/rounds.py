"""The next round of each competition, forecast whole from the dataset's state (the rule of
8 November).

A round is the matches of one `matchday` where the dataset has the column (ours: the five
leagues, ELC, the cups' league phase). Without it, it is derived from the first match still to
play:
- in a league, the next matches in kick-off order until a team repeats;
- in a cup, the same stage and leg;
- between nations, each nation's next match of the window (matches no more than
  `WINDOW_GAP_DAYS` apart, until a nation repeats).

A round is forecast once it is next: no team of it has another match to play before the round's
first kick-off (a Champions League matchday after a league weekend waits for the dataset that
holds the weekend). Then every match of the round still to play is a fixture, the rescheduled
ones too, whatever its kick-off; one whose team plays another match between the round's start and
its own kick-off is `before_previous_match`: its forecast is made without that result, and says
so.

A round already under way when the dataset was built (a matchday whose first kick-off is before
`now`) is forecast whole, its played matches too, from the facts known at its first kick-off:
none of its results is read (the rule of 8 November). Its fixtures carry that instant
(`as_of`), and those kicked off by `now` are `after_kickoff`: their forecast was made after the
match began, from the facts before the round, and says so (forecast log schema 12). The next
round follows as above.
"""

import sqlite3
from dataclasses import dataclass
from datetime import date

from vesta.data_format import has_column
from vesta.jobs.fixtures import TO_PLAY, Fixture
from vesta.spine.world import RESULT_STATUSES

WINDOW_GAP_DAYS = 4


@dataclass(frozen=True)
class RoundFixture:
    fixture: Fixture
    round_key: str  # "<code> <season> md <n>", "<code> <season> <stage> leg <n>", ...
    before_previous_match: bool
    as_of: str  # the instant of the facts it is forecast from
    after_kickoff: bool = False  # kicked off before the dataset was built


def _read_fixture_rows(connection: sqlite3.Connection, code: str, now: str) -> list[tuple]:
    """The competition's matches to play after `now`, in kick-off order."""
    return _read_rows(connection, "c.code = ? AND m.status IN ({}) AND m.kickoff_utc > ?"
                      .format(", ".join("?" for _ in TO_PLAY)), (code, *TO_PLAY, now))


def _read_matchday_rows(connection: sqlite3.Connection, code: str, season: int,
                        matchday: int) -> list[tuple]:  # fmt: skip
    """Every match of one matchday, played or to play, in kick-off order."""
    statuses = (*RESULT_STATUSES, *TO_PLAY)
    return _read_rows(connection, "c.code = ? AND s.season = ? AND m.matchday = ?"
                      " AND m.status IN ({})".format(", ".join("?" for _ in statuses)),
                      (code, season, matchday, *statuses))  # fmt: skip


def _read_rows(connection: sqlite3.Connection, condition: str, parameters: tuple) -> list[tuple]:
    matchday = "m.matchday" if has_column(connection, "matches", "matchday") else "NULL"
    stage = "COALESCE(m.stage, '')" if has_column(connection, "matches", "stage") else "''"
    leg = "m.leg" if has_column(connection, "matches", "leg") else "NULL"
    return connection.execute(
        f"""
        SELECT m.match_id, c.code, c.kind, s.season, m.home_team_id, m.away_team_id,
               m.kickoff_utc, m.neutral_venue, {stage}, {matchday}, {leg}, h.kind, a.kind
        FROM matches m
        JOIN seasons s ON s.season_id = m.season_id
        JOIN competitions c ON c.competition_id = s.competition_id
        JOIN teams h ON h.team_id = m.home_team_id
        JOIN teams a ON a.team_id = m.away_team_id
        WHERE {condition}
        ORDER BY m.kickoff_utc, m.match_id
        """, parameters).fetchall()  # fmt: skip


def _round_of(rows: list[tuple]) -> tuple[str, list[tuple]]:
    first = rows[0]
    code, season, stage, matchday, leg = first[1], first[3], first[8], first[9], first[10]
    if matchday is not None:
        return (f"{code} {season} md {matchday}",
                [row for row in rows if row[3] == season and row[9] == matchday])  # fmt: skip
    if first[11] == "nation" and first[12] == "nation":
        chosen, last, teams = [], date.fromisoformat(first[6][:10]), set()
        for row in rows:
            day = date.fromisoformat(row[6][:10])
            if (day - last).days > WINDOW_GAP_DAYS or {row[4], row[5]} & teams:
                break
            chosen.append(row)
            last, teams = day, teams | {row[4], row[5]}
        return f"{code} window {first[6][:10]}", chosen
    if first[2] != "league":
        return (f"{code} {season} {stage} leg {leg}",
                [row for row in rows if row[3] == season and row[8] == stage
                 and row[10] == leg])  # fmt: skip
    chosen, teams = [], set()
    for row in rows:
        if row[3] != season or {row[4], row[5]} & teams:
            break
        chosen.append(row)
        teams |= {row[4], row[5]}
    return f"{code} {season} from {first[6][:10]}", chosen


@dataclass(frozen=True)
class Rounds:
    fixtures: list[RoundFixture]
    waiting: dict[str, str]  # round key -> why it waits for a later dataset


def find_next_rounds(connection: sqlite3.Connection, now: str, leagues: list[str],
                     cups: list[str], nations: list[str] = ()) -> Rounds:  # fmt: skip
    """The fixtures of each competition's round under way at `now` (whole, from its first
    kick-off) and of its next round, when that round is next."""
    found, waiting = [], {}
    for code in [*leagues, *cups, *nations]:
        rows = _read_fixture_rows(connection, code, now)
        if not rows:
            continue
        sort = "league" if code in leagues else "cup" if code in cups else "nation"
        key, chosen = _round_of(rows)
        season, matchday = chosen[0][3], chosen[0][9]
        if matchday is not None:
            whole = _read_matchday_rows(connection, code, season, matchday)
            start = whole[0][6]
            if start <= now:  # under way: the whole round from its first kick-off
                fixtures = [_make_fixture(row, code, sort) for row in whole]
                found += [RoundFixture(fixture, key, _has_match_between(
                              connection, fixture, start, fixture.kickoff_utc), start,
                              fixture.kickoff_utc <= now) for fixture in fixtures]  # fmt: skip
                rows = [row for row in rows if not (row[3] == season and row[9] == matchday)]
                if not rows:
                    continue
                key, chosen = _round_of(rows)
        fixtures = [_make_fixture(row, code, sort) for row in chosen]
        start = min(fixture.kickoff_utc for fixture in fixtures)
        blocked = sum(_has_match_between(connection, fixture, now, start) for fixture in fixtures)
        if blocked:
            waiting[key] = (f"{blocked} of its {len(fixtures)} fixtures have a team playing"
                            f" before {start[:10]}: it waits for a dataset holding those"
                            " results")  # fmt: skip
            continue
        found += [RoundFixture(fixture, key, _has_match_between(connection, fixture, start,
                                                                fixture.kickoff_utc), now)
                  for fixture in fixtures]  # fmt: skip
    return Rounds(found, waiting)


def _make_fixture(row: tuple, code: str, sort: str) -> Fixture:
    return Fixture(row[0], code, sort, row[3], row[4], row[5], row[6], bool(row[7]), row[8])


def _has_match_between(connection: sqlite3.Connection, fixture: Fixture, after: str,
                       before: str) -> bool:  # fmt: skip
    """Whether a team of the fixture has another match to play between `after` and `before`."""
    return connection.execute(
        f"SELECT 1 FROM matches WHERE (home_team_id IN (?, ?) OR away_team_id IN (?, ?))"
        f" AND match_id <> ? AND kickoff_utc > ? AND kickoff_utc < ?"
        f" AND status IN ({', '.join('?' for _ in TO_PLAY)}) LIMIT 1",
        (fixture.home_team_id, fixture.away_team_id, fixture.home_team_id,
         fixture.away_team_id, fixture.match_id, after, before, *TO_PLAY),
    ).fetchone() is not None  # fmt: skip
