"""The user's command (`vesta.run`) and the round snapshot (`vesta.rounds`)."""

import io
import re
import sqlite3

from tests.vesta.test_data_format import _foreign_league

from vesta.capabilities import BASELINE_KEYS
from vesta.rounds import find_next_rounds
from vesta.run import main, run

LINE = re.compile(r"^PROGRESS (\d+) (\d+) (\d+|-) .+$")


def _add(connection, match_id, kickoff, home, away, season_id=3):
    connection.execute(
        "INSERT INTO matches (match_id, season_id, kickoff_utc, kickoff_precision, local_date,"
        " home_team_id, away_team_id, neutral_venue, status, extra_time_played) VALUES"
        " (?, ?, ?, 'minute', ?, ?, ?, 0, 'scheduled', 0)",
        (match_id, season_id, kickoff, kickoff[:10], home, away))  # fmt: skip


def test_the_run_writes_the_next_round_once_with_the_window_lines(tmp_path):
    (tmp_path / "data").mkdir()
    dataset = _foreign_league(tmp_path / "data" / "foreign.db")
    connection = sqlite3.connect(dataset)  # the round: 1-2 (9999) and 3-4 the same weekend
    _add(connection, 10000, "2025-09-07T15:00:00Z", 3, 4)
    connection.commit()
    connection.close()
    out = io.StringIO()
    counts = run([tmp_path / "data"], now="2025-09-01T08:00:00Z", out=out)
    lines = out.getvalue().splitlines()
    assert lines and all(LINE.match(line) for line in lines)
    done, total = (int(part) for part in lines[-1].split()[1:3])
    assert done == total == len(lines)
    assert counts["published"] == 2 and counts["proved"] == []
    from desk.reading import find_log

    log_path = tmp_path / "data" / "foreign.db.forecast-log.db"
    assert log_path == find_log(dataset)  # one file: the log where the window reads it
    log = sqlite3.connect(log_path)
    keys = {row[0] for row in log.execute("SELECT DISTINCT quantity_key FROM distributions")}
    assert keys == set(BASELINE_KEYS)
    assert {row[0] for row in log.execute("SELECT code FROM publication_caveats")} == {
        "not_validated_on_dataset"}
    log.close()
    again = run([tmp_path / "data"], now="2025-09-02T08:00:00Z", out=io.StringIO())
    assert again["published"] == 0  # the round's forecasts stay as they were


def test_a_rescheduled_fixture_of_the_round_carries_the_caveat_and_a_later_round_waits(tmp_path):
    dataset = _foreign_league(tmp_path / "foreign.db")
    connection = sqlite3.connect(dataset)
    connection.execute("ALTER TABLE matches ADD COLUMN matchday INTEGER")
    connection.execute("UPDATE matches SET matchday = 1 WHERE match_id = 9999")  # 1-2, 6 Sept
    _add(connection, 10000, "2025-09-20T15:00:00Z", 3, 4)  # md 1, moved after 3's md 2 match
    _add(connection, 10001, "2025-09-13T15:00:00Z", 3, 1)  # md 2
    _add(connection, 10002, "2025-09-13T15:00:00Z", 4, 2)  # md 2
    connection.execute("UPDATE matches SET matchday = 1 WHERE match_id = 10000")
    connection.execute("UPDATE matches SET matchday = 2 WHERE match_id IN (10001, 10002)")
    connection.commit()
    rounds = find_next_rounds(connection, "2025-09-01T08:00:00Z", ["XL"], [])
    caveats = {item.fixture.match_id: item.before_previous_match for item in rounds.fixtures}
    assert caveats == {9999: False, 10000: True}
    assert rounds.waiting == {}
    # a week on, the dataset not updated: md 2 is the next round
    later = find_next_rounds(connection, "2025-09-07T08:00:00Z", ["XL"], [])
    # md 2 is next by kick-off; md 1's moved match was forecast in md 1's snapshot
    assert {item.fixture.match_id for item in later.fixtures} == {10001, 10002}
    connection.close()


def test_a_missing_path_exits_with_one(tmp_path, capsys):
    assert main([str(tmp_path / "nothing")]) == 1
    assert "error" in capsys.readouterr().err


def test_errors_are_sentences_that_name_the_file(tmp_path, capsys):
    """A file that is not a database, or a dataset missing required columns: the error names the
    file or the table in words, never a Python repr."""
    broken = tmp_path / "broken.db"
    broken.write_bytes(b"this is not SQLite" * 100)
    assert main([str(broken)]) == 1
    said = capsys.readouterr().err
    assert "broken.db is not an SQLite database" in said
    dataset = _foreign_league(tmp_path / "league.db")
    connection = sqlite3.connect(dataset)
    connection.execute("ALTER TABLE competitions DROP COLUMN time_zone")
    connection.commit()
    connection.close()
    assert main([str(dataset)]) == 1
    said = capsys.readouterr().err
    assert "table competitions: missing columns ['time_zone']" in said and "Problem(" not in said


def _round_under_way(path):
    """XL with matchdays: md 1 is 1-2 (9999, 6 September, played 3-0) and 3-4 (7 September, to
    play); md 2 is 3-1 and 4-2 on 13 September."""
    dataset = _foreign_league(path)
    connection = sqlite3.connect(dataset)
    connection.execute("ALTER TABLE matches ADD COLUMN matchday INTEGER")
    _add(connection, 10000, "2025-09-07T15:00:00Z", 3, 4)
    _add(connection, 10001, "2025-09-13T15:00:00Z", 3, 1)
    _add(connection, 10002, "2025-09-13T15:00:00Z", 4, 2)
    connection.execute("UPDATE matches SET matchday = 1 WHERE match_id IN (9999, 10000)")
    connection.execute("UPDATE matches SET matchday = 2 WHERE match_id IN (10001, 10002)")
    connection.commit()
    return connection


def test_a_round_under_way_is_forecast_whole_from_its_first_kickoff(tmp_path):
    connection = _round_under_way(tmp_path / "under-way.db")
    connection.execute("UPDATE matches SET status = 'played', home_goals = 3, away_goals = 0"
                       " WHERE match_id = 9999")  # fmt: skip
    connection.commit()
    rounds = find_next_rounds(connection, "2025-09-06T20:00:00Z", ["XL"], [])
    by_match = {item.fixture.match_id: item for item in rounds.fixtures}
    assert set(by_match) == {9999, 10000}  # md 1 whole, the played match too
    assert {item.as_of for item in by_match.values()} == {"2025-09-06T15:00:00Z"}
    assert by_match[9999].after_kickoff and not by_match[10000].after_kickoff
    assert list(rounds.waiting) == ["XL 2025 md 2"]  # its teams play md 1 first
    connection.close()


def test_a_played_match_of_the_round_is_forecast_without_its_own_result(tmp_path):
    played = tmp_path / "played" / "under-way.db"
    played.parent.mkdir()
    connection = _round_under_way(played)
    connection.execute("UPDATE matches SET status = 'played', home_goals = 3, away_goals = 0"
                       " WHERE match_id = 9999")  # fmt: skip
    connection.commit()
    connection.close()
    before = tmp_path / "before" / "under-way.db"  # the same league, the round not yet begun
    before.parent.mkdir()
    _round_under_way(before).close()
    run([played.parent], now="2025-09-06T20:00:00Z", out=io.StringIO())
    run([before.parent], now="2025-09-06T14:00:00Z", out=io.StringIO())

    def published(folder):
        log = sqlite3.connect(folder / "under-way.db.forecast-log.db")
        rows = dict(log.execute(
            "SELECT p.match_id, d.distribution_values FROM distributions d JOIN publications p"
            " USING (publication_id) WHERE d.quantity_key = 'M1.result'").fetchall())
        triggers = dict(log.execute("SELECT match_id, trigger FROM publications").fetchall())
        caveats = {row[0] for row in log.execute(
            "SELECT DISTINCT p.match_id FROM publication_caveats c JOIN publications p"
            " USING (publication_id) WHERE c.code = 'forecast_after_kickoff'")}
        log.close()
        return rows, triggers, caveats

    after_rows, after_triggers, after_caveats = published(played.parent)
    before_rows, _, _ = published(before.parent)
    assert after_rows[9999] == before_rows[9999]  # the 3-0 is not read: no leakage
    assert after_triggers == {9999: "after_kickoff", 10000: "first"}
    assert after_caveats == {9999}



def _whole_league(path, code: str = "PL", clubs: int = 20, rounds_played: int = 6):
    """A league `code` of `clubs` clubs, a double round robin a season: 2023 and 2024 played with
    seeded scores, 2025 open with its first `rounds_played` rounds played and the rest to play.
    No shots, so B0 serves it."""
    import random
    from datetime import date, timedelta

    from vesta.data_format import VERSION, create_statements

    connection = sqlite3.connect(path)
    for statement in create_statements("B7"):
        connection.execute(statement)
    connection.execute("INSERT INTO format_version VALUES (?, 'someone', '2025-10-01T00:00:00Z')",
                       (VERSION,))  # fmt: skip
    connection.execute("INSERT INTO competitions VALUES (1, ?, 'league', 'Europe/London')", (code,))
    for season_id, season in ((1, 2023), (2, 2024), (3, 2025)):
        connection.execute("INSERT INTO seasons VALUES (?, 1, ?, ?)",
                           (season_id, season, "open" if season == 2025 else "closed"))
    connection.executemany("INSERT INTO teams VALUES (?, 'club', ?)",
                           [(team, f"club {team}") for team in range(1, clubs + 1)])  # fmt: skip
    teams = list(range(1, clubs + 1))
    halves = []
    for _ in range(clubs - 1):  # the circle method: every pairing once a half
        halves.append([(teams[index], teams[-1 - index]) for index in range(clubs // 2)])
        teams = [teams[0], teams[-1], *teams[1:-1]]
    schedule = halves + [[(away, home) for home, away in matches] for matches in halves]
    generator = random.Random(11)
    match_id = 0
    for season_id, season in ((1, 2023), (2, 2024), (3, 2025)):
        for number, matches in enumerate(schedule):
            kickoff = f"{date(season, 8, 2) + timedelta(days=7 * number):%Y-%m-%d}T15:00:00Z"
            for home, away in matches:
                match_id += 1
                played = season < 2025 or number < rounds_played
                connection.execute(
                    "INSERT INTO matches (match_id, season_id, kickoff_utc, kickoff_precision,"
                    " local_date, home_team_id, away_team_id, neutral_venue, status, home_goals,"
                    " away_goals, extra_time_played) VALUES (?, ?, ?, 'minute', ?, ?, ?, 0, ?, ?,"
                    " ?, 0)",
                    (match_id, season_id, kickoff, kickoff[:10], home, away,
                     "played" if played else "scheduled",
                     generator.choice((0, 1, 1, 2, 2, 3)) if played else None,
                     generator.choice((0, 0, 1, 1, 2)) if played else None))  # fmt: skip
    connection.commit()
    connection.close()
    return path


def test_one_league_without_the_shots_xg_gets_its_season_on_the_baseline(tmp_path):
    """A user's file of one league whose shots carry no xg is served by B0: its season is
    simulated on B0's matrices and published with `not_validated_on_dataset`, never left out."""
    dataset = _whole_league(tmp_path / "league.db")
    counts = run([dataset], now="2025-10-01T08:00:00Z", out=io.StringIO())
    assert counts["seasons"] == 1
    log = sqlite3.connect(tmp_path / "league.db.forecast-log.db")
    (season,) = log.execute("SELECT publication_id FROM publications WHERE stage = 'season_day'"
                            ).fetchall()  # fmt: skip
    keys = {row[0] for row in log.execute(
        "SELECT DISTINCT quantity_key FROM distributions WHERE publication_id = ?", season)}
    assert {"T1.final_position", "T1.points"} <= keys
    assert {row[0] for row in log.execute(
        "SELECT code FROM publication_caveats WHERE publication_id = ?", season)} == {
        "not_validated_on_dataset"}
    log.close()
