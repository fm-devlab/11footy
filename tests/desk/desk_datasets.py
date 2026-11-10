"""Someone else's dataset for the window's tests: the format's required columns only, its match
to come numbered as the small world's (`tests/fixtures/small_world.py`), so the sample log
stands for the one Vesta would write beside it."""

import sqlite3
from pathlib import Path

from small_world import ARSENAL, BOURNEMOUTH, TO_COME

THEIR_COMPETITION = 7
# The small world's release, ready (`tests/data/make_desk_fixtures.py` rebuilds it): the
# window's tests run without the pipeline
RELEASE = Path(__file__).resolve().parents[1] / "fixtures" / "desk" / "release"


def build_theirs(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        create table format_version (version text, dataset text, built_at text);
        create table competitions (competition_id integer primary key, code text, kind text,
                                   time_zone text);
        create table seasons (season_id integer primary key, competition_id integer,
                              season integer, status text);
        create table teams (team_id integer primary key, kind text, name text);
        create table matches (match_id integer primary key, season_id integer,
            kickoff_utc text, kickoff_precision text, local_date text, home_team_id integer,
            away_team_id integer, neutral_venue integer, status text, home_goals integer,
            away_goals integer, home_goals_on_pitch integer, away_goals_on_pitch integer);
        """
    )
    connection.execute(
        "insert into format_version values ('VDF 1', 'someone', '2026-10-20T04:00:00Z')"
    )
    connection.execute(
        f"insert into competitions values ({THEIR_COMPETITION}, 'XL', 'league', 'UTC')"
    )
    connection.execute(f"insert into seasons values (1, {THEIR_COMPETITION}, 2026, 'open')")
    connection.executemany(
        "insert into teams values (?, 'club', ?)", [(ARSENAL, "Club A"), (BOURNEMOUTH, "Club B")]
    )
    connection.execute(
        "insert into matches values (?, 1, '2026-10-25T15:30:00Z', 'minute', '2026-10-25', ?, ?,"
        " 0, 'scheduled', null, null, null, null)",
        (TO_COME, ARSENAL, BOURNEMOUTH),
    )
    connection.commit()
    connection.close()
    return path
