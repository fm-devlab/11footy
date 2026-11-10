"""The forecast log: schema, append-only guard, publications, records, views."""

import json
import sqlite3
from pathlib import Path

import pytest

from vesta.distributions import DistributionError
from vesta.forecast_log.store import ForecastLogError, open_log
from vesta.forecast_log.writing import (
    Caveat,
    Change,
    ExpectedSlot,
    Explanation,
    Publication,
    PublishedDistribution,
    Quantity,
    SideInputs,
    TrackRecordScore,
    publish,
    record_calibration,
    record_expected_eleven,
    record_explanation,
    record_forecasts_of_record,
    record_line_check,
    record_outcome,
    record_scores,
    record_void,
    record_withdrawal,
    register_model_version,
    register_published_quantity,
    register_quantity,
)
from vesta.forecast_reading import LineError
from vesta.research.calibration import bucket_lines

VERSION = "vesta-2.0.0"
MATCH = 77120
KICKOFF = "2026-10-24T19:00:00Z"
RESULT = [["home", 0.5], ["draw", 0.3], ["away", 0.2]]
GOALS = {"probabilities": [0.1, 0.2, 0.3, 0.2, 0.2], "tail_from": 5, "tail": 0.0}


def pre_match(published_at: str, scheduled: str = KICKOFF, trigger: str = "morning") -> Publication:
    return Publication(
        stage="pre_match",
        trigger=trigger,
        published_at=published_at,
        code_commit="abc1234",
        match_id=MATCH,
        scheduled_kickoff_utc=scheduled,
        world_build="f00d",
    )


def match_distributions(home: float = 0.5) -> list[PublishedDistribution]:
    result = [["home", home], ["draw", 0.3], ["away", round(0.7 - home, 10)]]
    return [
        PublishedDistribution("M1.result", MATCH, VERSION, result),
        PublishedDistribution("M3.goals", MATCH, VERSION, GOALS),
    ]


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "forecast-log.db"


@pytest.fixture
def log(log_path: Path) -> sqlite3.Connection:
    connection = open_log(log_path)
    register_model_version(connection, VERSION, "spine", "2026-10-24T00:00:00Z", "the spine")
    register_quantity(
        connection,
        Quantity("M1.result", "M1", "results", "match", "result", "full_time", "ordered", False,
                 "matches.home_goals_on_pitch or home_goals", ("outcome=home", "outcome=draw",
                 "outcome=away")),  # fmt: skip
    )
    register_quantity(
        connection,
        Quantity("M3.goals", "M3", "goals", "match", "goals", "full_time", "count", False,
                 "matches goals", ("over=2.5", "under=2.5")),  # fmt: skip
    )
    yield connection
    connection.close()


def count(connection: sqlite3.Connection, table: str) -> int:
    return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_a_new_log_is_in_write_ahead_mode_with_its_schema_version(log, log_path):
    assert log.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert log.execute("PRAGMA user_version").fetchone()[0] == 12
    open_log(log_path).close()


def test_a_log_of_another_schema_version_is_refused(log, log_path):
    log.execute("PRAGMA user_version = 13")
    with pytest.raises(ForecastLogError, match="schema version 13"):
        open_log(log_path)


def test_publications_get_growing_ids_and_keep_whole_distributions(log):
    first = publish(log, pre_match("2026-10-22T08:00:00Z", trigger="first"), match_distributions())
    second = publish(log, pre_match("2026-10-23T08:00:00Z"), match_distributions(0.45))
    assert second > first
    assert count(log, "distributions") == 4
    assert count(log, "probabilities") == 0  # lines become rows only at the record (A2)
    expected = log.execute(
        "SELECT expected_value FROM distributions WHERE quantity_key = 'M3.goals' LIMIT 1"
    ).fetchone()[0]
    assert expected == pytest.approx(0.2 + 0.6 + 0.6 + 0.8)


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE publications SET published_at = '2026-01-01T00:00:00Z'",
        "DELETE FROM publications",
        "UPDATE distributions SET distribution_values = '[]'",
        "DELETE FROM distributions",
        "UPDATE model_versions SET what_changed = 'something else'",
    ],
)
def test_nothing_published_is_changed_or_deleted(log, statement):
    publish(log, pre_match("2026-10-22T08:00:00Z"), match_distributions())
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        log.execute(statement)


def test_a_closing_date_is_written_once(log):
    log.execute("UPDATE model_versions SET superseded_from = '2026-12-01T00:00:00Z'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        log.execute("UPDATE model_versions SET superseded_from = '2026-12-02T00:00:00Z'")


def test_a_pre_match_publication_after_the_scheduled_kickoff_is_refused(log):
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        publish(log, pre_match("2026-10-24T19:00:00Z"), match_distributions())
    assert count(log, "publications") == 0


def test_a_bad_distribution_leaves_nothing_of_its_publication(log):
    bad = [*match_distributions(), PublishedDistribution("M1.result", 5, VERSION, [["home", 0.9]])]
    with pytest.raises(DistributionError):
        publish(log, pre_match("2026-10-22T08:00:00Z"), bad)
    assert count(log, "publications") == count(log, "distributions") == 0


def test_a_distribution_that_cannot_answer_its_lines_is_refused(log):
    short = {"probabilities": [0.2, 0.3], "tail_from": 2, "tail": 0.5}
    with pytest.raises(LineError, match="tail"):
        publish(
            log,
            pre_match("2026-10-22T08:00:00Z"),
            [PublishedDistribution("M3.goals", MATCH, VERSION, short)],
        )


def test_an_unknown_quantity_is_refused(log):
    with pytest.raises(ForecastLogError, match="not defined"):
        publish(
            log,
            pre_match("2026-10-22T08:00:00Z"),
            [PublishedDistribution("M99.nothing", MATCH, VERSION, {"yes": 0.5})],
        )






def test_the_forecast_of_record_is_the_last_before_the_real_kickoff(log):
    publish(log, pre_match("2026-10-22T08:00:00Z", trigger="first"), match_distributions(0.4))
    of_record = publish(log, pre_match("2026-10-24T17:58:00Z", trigger="lineup"),
                        match_distributions(0.45))  # fmt: skip
    # the kickoff was brought forward to 18:00: this one, published at 18:30, is late
    late = publish(log, pre_match("2026-10-24T18:30:00Z"), match_distributions(0.6))
    assert record_forecasts_of_record(log, MATCH, "2026-10-24T18:00:00Z", KICKOFF) == 2
    chosen = {row[0] for row in log.execute("SELECT publication_id FROM forecasts_of_record")}
    assert chosen == {of_record}
    lines = dict(
        log.execute(
            "SELECT line_key, probability FROM probabilities WHERE quantity_key = 'M1.result'"
        ).fetchall()
    )
    assert lines["outcome=home"] == pytest.approx(0.45)
    movement = [row[0] for row in log.execute("SELECT publication_id FROM movement")]
    assert late not in movement and of_record in movement
    with pytest.raises(ForecastLogError, match="written"):
        record_forecasts_of_record(log, MATCH, "2026-10-24T18:00:00Z", KICKOFF)




def test_checks_count_void_and_withdrawn_lines(log):
    publication_id = publish(log, pre_match("2026-10-22T08:00:00Z"), match_distributions())
    record_forecasts_of_record(log, MATCH, KICKOFF, KICKOFF)
    states = {row[0] for row in log.execute("SELECT state FROM checks")}
    assert states == {"pending"}
    record_outcome(log, "M3.goals", MATCH, 3, "matches", "f00d", "2026-10-24T21:05:00Z",
                   match_id=MATCH)  # fmt: skip
    record_line_check(log, publication_id, "M3.goals", MATCH, "over=2.5", "checked", True,
                      "2026-10-24T21:05:00Z")  # fmt: skip
    record_line_check(log, publication_id, "M1.result", MATCH, "outcome=home", "void", None,
                      "2026-10-24T21:05:00Z")  # fmt: skip
    by_line = dict(log.execute("SELECT line_key, state FROM checks").fetchall())
    assert by_line["over=2.5"] == "counted" and by_line["outcome=home"] == "void"
    record_withdrawal(log, publication_id, "bug", "the home advantage was counted twice",
                      "1a2b3c4d", "reviewer", "2026-10-25T09:00:00Z")  # fmt: skip
    assert {row[0] for row in log.execute("SELECT state FROM checks")} == {"withdrawn"}
    assert count(log, "withdrawn_publications") == 1


def test_a_withdrawal_needs_the_commit_that_fixes_it(log):
    publication_id = publish(log, pre_match("2026-10-22T08:00:00Z"), match_distributions())
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        record_withdrawal(log, publication_id, "bug", "why", "", "reviewer",
                          "2026-10-25T09:00:00Z")  # fmt: skip


def test_a_void_names_its_match_and_a_known_reason(log):
    record_void(log, "did_not_play", "not in the sheet", "2026-10-24T21:00:00Z", match_id=MATCH,
                quantity_key="M3.goals", subject_id=MATCH)  # fmt: skip
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        record_void(log, "rain", "evidence", "2026-10-24T21:00:00Z", match_id=MATCH)


def test_the_season_evolution_lists_every_morning(log):
    register_quantity(
        log,
        Quantity("T1.final_position", "T1", "seasons", "season_team", "final position",
                 "season", "rank", False, "standings", ("place=1", "top=4")),  # fmt: skip
    )
    for day, title in (("2026-10-23", 0.4), ("2026-10-24", 0.35)):
        morning = Publication(stage="season_day", trigger="morning",
                              published_at=f"{day}T06:00:00Z", code_commit="abc1234",
                              season_id=4411, moved_by=[MATCH])  # fmt: skip
        places = [title, 0.3, 0.2, round(0.5 - title, 10)]
        publish(log, morning, [PublishedDistribution("T1.final_position", 13, VERSION, places)])
    rows = log.execute("SELECT published_at, moved_by FROM season_evolution").fetchall()
    assert [row[0][:10] for row in rows] == ["2026-10-23", "2026-10-24"]
    assert rows[0][1] == f"[{MATCH}]"




def test_an_explanation_may_explain_the_whole_distribution_or_one_line(log):
    publication_id = publish(log, pre_match("2026-10-22T08:00:00Z"), match_distributions())

    def explanation(line_key: str) -> Explanation:
        return Explanation(publication_id, "M3.goals", MATCH, VERSION, "why", [], [], [],
                           line_key=line_key)  # fmt: skip

    record_explanation(log, explanation(""))
    record_explanation(log, explanation("over=2.5"))
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        record_explanation(log, explanation("over=2.5"))
    assert count(log, "explanations") == 2


def test_the_map_of_what_is_published_and_its_closing_date(log):
    register_published_quantity(log, "M1.result", 1, VERSION, "2026-10-24T00:00:00Z", True, True)
    log.execute("UPDATE published_quantities SET published_until = '2026-12-01T00:00:00Z'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        log.execute("UPDATE published_quantities SET live_headline = 0")


def test_calibration_and_scores_are_written_by_round(log):
    lines = [(0.25, True), (0.25, False), (0.25, False), (0.25, False), (0.7, True)]
    record_calibration(
        log, "results", VERSION, 0, "all", bucket_lines(lines), "2026-10-25T06:00:00Z"
    )
    rows = log.execute("SELECT said_from, count FROM calibration_buckets").fetchall()
    assert rows == [(0.2, 4), (0.7, 1)]
    score = TrackRecordScore("results", VERSION, 0, "all", "log_loss", 0.98, "league frequencies",
                             1.07, 5, 1, 0, 0)  # fmt: skip
    record_scores(log, [score], "2026-10-25T06:00:00Z")
    assert count(log, "track_record_scores") == 1


def test_a_log_of_version_eight_takes_the_scores_of_every_version_with_its_rows_kept(log,
                                                                                     log_path):
    """Schema 9 (5 November 2026): `track_record_scores` rebuilt so model_version 'all' (every
    version together) needs no model version, the scores already written kept, still
    append-only."""
    import re

    from vesta.forecast_log import store

    score = TrackRecordScore("results", VERSION, 0, "all", "skill", 0.04, "league frequencies",
                             0.0, 5, 1, 0, 0)  # fmt: skip
    record_scores(log, [score], "2026-10-25T06:00:00Z")
    rows = log.execute("SELECT * FROM track_record_scores").fetchall()
    statement = log.execute("SELECT sql FROM sqlite_master WHERE name = 'track_record_scores'"
                            ).fetchone()[0]  # fmt: skip
    log.execute("ALTER TABLE track_record_scores RENAME TO scores_eight")
    for trigger in ("no_delete", "no_update"):
        log.execute(f"DROP TRIGGER track_record_scores_{trigger}")
    log.execute(re.sub(r"model_version TEXT NOT NULL,",
                       "model_version TEXT NOT NULL REFERENCES model_versions,", statement))
    log.execute("INSERT INTO track_record_scores SELECT * FROM scores_eight")
    log.execute("DROP TABLE scores_eight")
    log.execute("PRAGMA user_version = 8")  # the log as schema version 8 left it
    with pytest.raises(sqlite3.IntegrityError):
        record_scores(log, [TrackRecordScore("results", "all", 0, "all", "skill", 0.04,
                                             "league frequencies", 0.0, 5, 1, 0, 0)],
                      "2026-10-26T06:00:00Z")  # fmt: skip
    log.close()
    upgraded = open_log(log_path)
    assert upgraded.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION == 12
    assert upgraded.execute("SELECT * FROM track_record_scores").fetchall() == rows
    record_scores(upgraded, [TrackRecordScore("results", "all", 0, "all", "skill", 0.04,
                                              "league frequencies", 0.0, 5, 1, 0, 0)],
                  "2026-10-26T06:00:00Z")  # fmt: skip
    assert count(upgraded, "track_record_scores") == 2
    with pytest.raises(sqlite3.DatabaseError):  # still append-only
        upgraded.execute("DELETE FROM track_record_scores")
    upgraded.close()



def test_a_log_of_version_nine_takes_the_dataset_caveat_with_its_rows_kept(log, log_path):
    """Schema 10 (8 November 2026, VDF S1): `publication_caveats` rebuilt with
    `not_validated_on_dataset` in its CHECK, the caveats already written kept, still
    append-only."""
    from vesta.forecast_log import store

    publish(log, pre_match("2026-10-22T08:00:00Z", trigger="first"), match_distributions(),
            [Caveat("results", "injuries_not_included", VERSION)])  # fmt: skip
    rows = log.execute("SELECT * FROM publication_caveats").fetchall()
    statement = log.execute("SELECT sql FROM sqlite_master WHERE name = 'publication_caveats'"
                            ).fetchone()[0]  # fmt: skip
    log.execute("ALTER TABLE publication_caveats RENAME TO caveats_nine")
    for trigger in ("no_delete", "no_update"):
        log.execute(f"DROP TRIGGER publication_caveats_{trigger}")
    log.execute(statement.replace(", 'not_validated_on_dataset'", ""))
    log.execute("INSERT INTO publication_caveats SELECT * FROM caveats_nine")
    log.execute("DROP TABLE caveats_nine")
    log.execute("PRAGMA user_version = 9")  # the log as schema version 9 left it
    with pytest.raises(sqlite3.IntegrityError):
        log.execute("INSERT INTO publication_caveats SELECT publication_id, 'goals',"
                    " 'not_validated_on_dataset', 'x', model_version FROM publication_caveats")
    log.close()
    upgraded = open_log(log_path)
    assert upgraded.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION == 12
    assert upgraded.execute("SELECT * FROM publication_caveats").fetchall() == rows
    later = publish(upgraded, pre_match("2026-10-23T08:00:00Z"), match_distributions(),
                    [Caveat("results", "not_validated_on_dataset", VERSION)])  # fmt: skip
    assert upgraded.execute(
        "SELECT description FROM publication_caveats WHERE publication_id = ?", (later,)
    ).fetchone()[0] == "No blind verdict of Vesta covers this dataset."
    with pytest.raises(sqlite3.DatabaseError):  # still append-only
        upgraded.execute("DELETE FROM publication_caveats")
    upgraded.close()

def eleven(first_player: int = 501) -> list[ExpectedSlot]:
    return [ExpectedSlot(slot, first_player + slot, "DC", 0.9, 0.8) for slot in range(1, 12)]


def test_an_expected_eleven_is_written_whole_with_one_player_per_slot(log):
    publication_id = publish(log, pre_match("2026-10-22T08:00:00Z"), match_distributions())
    record_expected_eleven(log, publication_id, 13, "4-3-3", 0.6, eleven(), VERSION)
    assert count(log, "expected_elevens") == 11
    with pytest.raises(ForecastLogError, match="slots 1 to 11"):
        record_expected_eleven(log, publication_id, 14, "4-3-3", 0.6, eleven()[:10], VERSION)
    twice = [*eleven(700)[:10], ExpectedSlot(11, 701, "FW", 0.5, 0.5)]  # 701 holds slot 1 too
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        record_expected_eleven(log, publication_id, 14, "4-3-3", 0.6, twice, VERSION)
    assert count(log, "expected_elevens") == 11  # nothing of the refused eleven stays
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        log.execute("DELETE FROM expected_elevens")


def test_the_movement_finds_the_publications_of_record_through_an_index(log):
    steps = log.execute("EXPLAIN QUERY PLAN SELECT * FROM movement WHERE match_id = 1")
    plan = " ".join(row[3] for row in steps)
    assert "forecasts_of_record_by_publication" in plan




def test_a_decided_line_is_written_as_decided_and_exactly_zero_or_one(log):
    whistle = Publication(stage="full_time", trigger="final_whistle",
                          published_at="2026-10-24T20:55:00Z", code_commit="abc1234",
                          match_id=MATCH)  # fmt: skip
    three_goals = {"probabilities": [0.0, 0.0, 0.0, 1.0, 0.0], "tail_from": 5, "tail": 0.0}
    decided = PublishedDistribution("M3.goals", MATCH, VERSION, three_goals,
                                    decided_lines=("over=2.5", "under=2.5"))  # fmt: skip
    publish(log, whistle, [decided])
    rows = dict(log.execute("SELECT line_key, decided FROM probabilities").fetchall())
    assert rows == {"over=2.5": 1, "under=2.5": 1}
    assert log.execute("SELECT decided_lines FROM distributions").fetchone() == (
        '["over=2.5", "under=2.5"]',
    )


def test_a_line_is_never_decided_by_a_probability_that_is_not_zero_or_one(log):
    open_line = PublishedDistribution("M3.goals", MATCH, VERSION, GOALS,
                                      decided_lines=("over=2.5",))  # fmt: skip
    with pytest.raises(ForecastLogError, match="not 0 or 1"):
        publish(log, pre_match("2026-10-22T08:00:00Z"), [open_line])
    unknown = PublishedDistribution("M3.goals", MATCH, VERSION, GOALS, decided_lines=("odd",))
    with pytest.raises(ForecastLogError, match="not one of"):
        publish(log, pre_match("2026-10-22T08:00:00Z"), [unknown])




def test_the_day_carries_the_round_in_numbers_of_a_day_and_nothing_else(log):
    register_quantity(
        log,
        Quantity("C4.goalless_draw.day", "C4", "seasons", "competition", "a 0-0 in a day",
                 "full_time", "binary", False, "matches", ("yes", "no")),  # fmt: skip
    )
    register_quantity(
        log,
        Quantity("C4.goalless_draw.matchday", "C4", "seasons", "competition",
                 "a 0-0 in a matchday", "full_time", "binary", False, "matches",
                 ("yes", "no")),  # fmt: skip
    )
    day = Publication(stage="day", trigger="morning", published_at="2026-10-25T06:00:00Z",
                      code_commit="abc1234")  # fmt: skip
    every = PublishedDistribution("C4.goalless_draw.day", 0, VERSION, {"yes": 0.6},
                                  "2026-10-25/2026-10-25")  # fmt: skip
    assert publish(log, day, [every]) > 0
    matchday = PublishedDistribution("C4.goalless_draw.matchday", 1, VERSION, {"yes": 0.4},
                                     "matchday=9")  # fmt: skip
    with pytest.raises(ForecastLogError, match="stage 'day'"):
        publish(log, day, [every, matchday])
    season = Publication(stage="season_day", trigger="morning",
                         published_at="2026-10-25T06:00:00Z", code_commit="abc1234",
                         season_id=4411)  # fmt: skip
    with pytest.raises(ForecastLogError, match="stage 'day'"):
        publish(log, season, [every])
    assert publish(log, season, [matchday]) > 0
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        publish(log, Publication(stage="day", trigger="morning",
                                 published_at="2026-10-25T06:00:00Z", code_commit="abc1234",
                                 season_id=4411), [every])  # fmt: skip






def test_a_log_of_version_one_is_migrated_with_its_rows_kept(tmp_path):
    import re

    from vesta.forecast_log import store

    schema = store.SCHEMA_FILE.read_text(encoding="utf-8")
    first = re.sub(r"    -- season_day only:.*?'season_day'\)\),\n", "", schema, flags=re.S)
    first = re.sub(r"    -- live only \(schema version 7\).*?stage = 'live'\)\),\n", "", first,
                   flags=re.S)  # fmt: skip
    first = re.sub(r"\n-- schema version 3.*?publication_id\);\n", "\n", first, flags=re.S)
    first = re.sub(r"\n-- schema version 4.*?computed_at\)\n\);\n", "\n", first, flags=re.S)
    first = re.sub(r"\n-- schema version 5 .*?(?=\n-- The views)", "\n", first, flags=re.S)
    assert "live_headlines" not in first and "track_record_hits" not in first
    assert "tie_break_carried_forward" not in first and "at_break" not in first
    assert "publication_caveats" not in first and "publication_inputs" not in first
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("BEGIN;" + first + store._build_append_only_triggers(first) + "COMMIT;")
    old.execute("PRAGMA user_version = 1")
    old.execute("INSERT INTO publications (stage, trigger, published_at, season_id,"
                " code_commit) VALUES ('season_day', 'first', '2026-10-28T07:00:00Z', 7,"
                " 'abc1234')")  # fmt: skip
    old.commit()
    old.close()
    migrated = open_log(path)
    assert migrated.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION
    assert migrated.execute("SELECT stage, tie_break_carried_forward, at_break FROM publications"
                            ).fetchall() == [("season_day", None, None)]  # fmt: skip
    tables = {row[0] for row in migrated.execute("SELECT name FROM sqlite_master")}
    assert {"publication_caveats", "publication_changes", "publication_inputs"} <= tables
    with pytest.raises(sqlite3.DatabaseError):  # the rows stay append-only
        migrated.execute("DELETE FROM publications")
    migrated.close()


def test_a_publication_writes_its_caveats_changes_and_inputs_with_it(log):
    caveat = Caveat("results", "injuries_not_included", VERSION)
    change = Change("weekly_fit", 601, {"fit_week": "2026-11-02", "results_added": [73501]}, None,
                    VERSION)  # fmt: skip
    side = SideInputs(601, "2026-11-02", (9, 3, 5))
    publication_id = publish(log, pre_match("2026-10-22T08:00:00Z", trigger="first"),
                             match_distributions(), [caveat], [change], [side])  # fmt: skip
    assert log.execute("SELECT probability_kind, code, description FROM publication_caveats"
                       " WHERE publication_id = ?", (publication_id,)).fetchall() == [
        ("results", "injuries_not_included", "Injuries are not yet included in this chance.")]
    assert log.execute("SELECT code, team_id, detail FROM publication_changes").fetchall() == [
        ("weekly_fit", 601, '{"fit_week": "2026-11-02", "results_added": [73501]}')]
    assert log.execute("SELECT team_id, fit_week, eleven FROM publication_inputs").fetchall() == [
        (601, "2026-11-02", "[3, 5, 9]")]
    for wrong in ({"caveats": [Caveat("results", "weather_not_included", VERSION)]},
                  {"changes": [Change("a_rumour", 0, {}, None, VERSION)]}):
        with pytest.raises(ForecastLogError, match="not a"):
            publish(log, pre_match("2026-10-23T08:00:00Z"), match_distributions(), **wrong)
    assert count(log, "publications") == 1  # refused before anything is written


def test_a_log_of_version_five_takes_the_new_caveat_with_its_rows_kept(log, log_path):
    """Schema 6 (3 November 2026): `publication_caveats` rebuilt with `suspensions_from_cards` in
    its CHECK, the caveats already written kept, still append-only."""
    import re

    from vesta.forecast_log import store

    publication_id = publish(log, pre_match("2026-10-22T08:00:00Z", trigger="first"),
                             match_distributions(),
                             [Caveat("results", "injuries_not_included", VERSION)])  # fmt: skip
    rows = log.execute("SELECT * FROM publication_caveats").fetchall()
    statement = log.execute("SELECT sql FROM sqlite_master WHERE name = 'publication_caveats'"
                            ).fetchone()[0]  # fmt: skip
    log.execute("ALTER TABLE publication_caveats RENAME TO caveats_six")
    for trigger in ("no_delete", "no_update"):
        log.execute(f"DROP TRIGGER publication_caveats_{trigger}")
    log.execute(re.sub(r"-- schema version [68].*?\n\s*|'suspensions_from_cards',\s*"
                       r"|,\s*'shots_stale'", "", statement))  # fmt: skip
    before = log.execute("SELECT sql FROM sqlite_master WHERE name = 'publication_caveats'"
                         ).fetchone()[0]  # fmt: skip
    assert "suspensions_from_cards" not in before and "shots_stale" not in before
    log.execute("INSERT INTO publication_caveats SELECT * FROM caveats_six")
    log.execute("DROP TABLE caveats_six")
    log.execute("ALTER TABLE publications DROP COLUMN at_break")
    log.execute("PRAGMA user_version = 5")  # the log as schema version 5 left it
    log.close()
    upgraded = open_log(log_path)
    assert upgraded.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION == 12
    assert upgraded.execute("SELECT * FROM publication_caveats").fetchall() == rows
    later = publish(upgraded, pre_match("2026-10-23T08:00:00Z"), match_distributions(),
                    [Caveat("lineups", "suspensions_from_cards", VERSION)])  # fmt: skip
    assert later != publication_id and upgraded.execute(
        "SELECT description FROM publication_caveats WHERE publication_id = ?", (later,)
    ).fetchone()[0].startswith("Suspensions are derived from the cards")
    stale = publish(upgraded, pre_match("2026-10-24T08:00:00Z"), match_distributions(),
                    [Caveat("scorers", "shots_stale", VERSION)])  # fmt: skip
    assert upgraded.execute(
        "SELECT description FROM publication_caveats WHERE publication_id = ?", (stale,)
    ).fetchone()[0].startswith("Built on the shots of earlier matches")
    with pytest.raises(sqlite3.DatabaseError):  # still append-only
        upgraded.execute("DELETE FROM publication_caveats")
    upgraded.close()


def test_only_a_season_morning_says_whose_tie_break_rule_it_used(log):
    season = Publication(stage="season_day", trigger="first", published_at="2026-10-28T07:00:00Z",
                         code_commit="abc1234", season_id=7, tie_break_carried_forward=True)
    register_quantity(log, Quantity("T1.points", "T1", "seasons", "season_team", "points",
                                    "season", "count", False, "the final table", ()))  # fmt: skip
    points = {"probabilities": [0.5, 0.5], "tail_from": 2, "tail": 0.0}
    publish(log, season, [PublishedDistribution("T1.points", 3, VERSION, points)])
    assert log.execute("SELECT tie_break_carried_forward FROM publications").fetchone() == (1,)
    with pytest.raises(sqlite3.IntegrityError):
        log.execute("INSERT INTO publications (stage, trigger, published_at, match_id,"
                    " scheduled_kickoff_utc, code_commit, tie_break_carried_forward)"
                    " VALUES ('pre_match', 'first', '2026-10-24T08:00:00Z', 1,"
                    " '2026-10-24T19:00:00Z', 'abc1234', 1)")  # fmt: skip














def test_a_log_of_version_ten_takes_the_round_caveat_with_its_rows_kept(log, log_path):
    """Schema 11 (8 November 2026, the round snapshot): `publication_caveats` rebuilt with
    `before_previous_match` in its CHECK, the caveats already written kept."""
    from vesta.forecast_log import store

    publish(log, pre_match("2026-10-22T08:00:00Z", trigger="first"), match_distributions(),
            [Caveat("results", "not_validated_on_dataset", VERSION)])  # fmt: skip
    rows = log.execute("SELECT * FROM publication_caveats").fetchall()
    statement = log.execute("SELECT sql FROM sqlite_master WHERE name = 'publication_caveats'"
                            ).fetchone()[0]  # fmt: skip
    log.execute("ALTER TABLE publication_caveats RENAME TO caveats_ten")
    for trigger in ("no_delete", "no_update"):
        log.execute(f"DROP TRIGGER publication_caveats_{trigger}")
    log.execute(statement.replace(",\n        'before_previous_match'", ""))
    log.execute("INSERT INTO publication_caveats SELECT * FROM caveats_ten")
    log.execute("DROP TABLE caveats_ten")
    log.execute("PRAGMA user_version = 10")
    with pytest.raises(sqlite3.IntegrityError):
        log.execute("INSERT INTO publication_caveats SELECT publication_id, 'goals',"
                    " 'before_previous_match', 'x', model_version FROM publication_caveats")
    log.close()
    upgraded = open_log(log_path)
    assert upgraded.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION == 12
    assert upgraded.execute("SELECT * FROM publication_caveats").fetchall() == rows
    publish(upgraded, pre_match("2026-10-23T08:00:00Z"), match_distributions(),
            [Caveat("results", "before_previous_match", VERSION)])  # fmt: skip


def test_the_movement_runs_from_the_first_publication_to_the_whistle(log):
    first = publish(log, pre_match("2026-10-22T08:00:00Z", trigger="first"), match_distributions())
    record_forecasts_of_record(log, MATCH, KICKOFF, KICKOFF)
    whistle = Publication(stage="full_time", trigger="final_whistle",
                          published_at="2026-10-24T20:55:00Z", code_commit="abc1234",
                          match_id=MATCH)  # fmt: skip
    final = publish(log, whistle, match_distributions(0.7))
    rows = log.execute("SELECT publication_id, stage, of_record FROM movement").fetchall()
    assert rows == [(first, "pre_match", 1), (final, "full_time", 0)]
    assert count(log, "probabilities") == 2 * 5  # the record's lines and the whistle's


def test_the_reader_sees_a_commit_through_data_version_and_cannot_write(log, log_path):
    from vesta.forecast_log.store import open_for_reading

    reader = open_for_reading(log_path)
    before = reader.execute("PRAGMA data_version").fetchone()[0]
    publish(log, pre_match("2026-10-22T08:00:00Z", trigger="first"), match_distributions())
    assert reader.execute("PRAGMA data_version").fetchone()[0] != before
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        reader.execute("INSERT INTO voids (match_id, reason, evidence, recorded_at)"
                       " VALUES (1, 'cancelled', 'x', 'y')")  # fmt: skip
    reader.close()


def test_a_sum_of_floats_just_beyond_one_is_held_at_one_never_farther(log):
    register_quantity(log, Quantity("M8.first_goal_time", "M8", "timing", "match",
                                    "first goal", "full_time", "time_to_event", False,
                                    "goal minutes", ()))  # fmt: skip
    certain = {"bands": [[1, 15, 1.0000000000000002], [16, 30, 0.0]], "never": 0.0}
    publication_id = publish(log, pre_match("2026-10-22T08:00:00Z", trigger="first"),
                             [PublishedDistribution("M8.first_goal_time", MATCH, VERSION,
                                                    certain)])  # fmt: skip
    stored = log.execute("SELECT distribution_values FROM distributions WHERE publication_id = ?",
                         (publication_id,)).fetchone()[0]  # fmt: skip
    assert json.loads(stored)["bands"][0] == [1, 15, 1.0]
    beyond = {"bands": [[1, 15, 1.01], [16, 30, 0.0]], "never": 0.0}
    with pytest.raises(DistributionError):
        publish(log, pre_match("2026-10-23T08:00:00Z"),
                [PublishedDistribution("M8.first_goal_time", MATCH, VERSION, beyond)])


def test_a_log_of_version_eleven_takes_the_after_kickoff_forecast_with_its_rows_kept(log,
                                                                                    log_path):
    """Schema 12 (8 November 2026, the round under way): `publications` and
    `publication_caveats` rebuilt with the `after_kickoff` trigger and its caveat, every row
    kept, the other tables' references and the views still on `publications`."""
    from vesta.forecast_log import store

    first = publish(log, pre_match("2026-10-22T08:00:00Z", trigger="first"),
                    match_distributions(), [Caveat("results", "injuries_not_included", VERSION)])
    publications = log.execute("SELECT * FROM publications").fetchall()
    distributions = log.execute("SELECT count(*) FROM distributions").fetchone()[0]
    log.execute("PRAGMA user_version = 11")  # rebuilt again from 11, as an old log would be
    log.close()
    upgraded = open_log(log_path)
    assert upgraded.execute("PRAGMA user_version").fetchone()[0] == store.SCHEMA_VERSION == 12
    assert upgraded.execute("SELECT * FROM publications").fetchall() == publications
    assert upgraded.execute("SELECT count(*) FROM distributions").fetchone()[0] == distributions
    assert upgraded.execute("PRAGMA foreign_key_check").fetchall() == []
    references = upgraded.execute(
        "SELECT sql FROM sqlite_master WHERE sql LIKE '%publications_before%'").fetchall()
    assert references == []  # no table, view or trigger points at the dropped table
    after = Publication(stage="pre_match", trigger="after_kickoff",
                        published_at="2026-10-25T09:00:00Z", code_commit="abc1234",
                        match_id=MATCH, scheduled_kickoff_utc=KICKOFF)  # fmt: skip
    later = publish(upgraded, after, match_distributions(0.6),
                    [Caveat("results", "forecast_after_kickoff", VERSION)])
    assert later > first
    with pytest.raises(sqlite3.IntegrityError):  # any other trigger stays before the kick-off
        publish(upgraded, pre_match("2026-10-25T09:00:00Z"), match_distributions())
    record_forecasts_of_record(upgraded, MATCH, KICKOFF, KICKOFF)
    of_record = upgraded.execute("SELECT DISTINCT publication_id FROM forecasts_of_record"
                                 ).fetchall()  # fmt: skip
    assert of_record == [(first,)]  # the after-kick-off forecast is never of record


def test_a_migrated_log_still_enforces_its_references(log, log_path):
    """The foreign keys are off only inside the migration's transaction."""
    log.execute("PRAGMA user_version = 11")
    log.close()
    upgraded = open_log(log_path)
    assert upgraded.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        upgraded.execute("INSERT INTO distributions (publication_id, quantity_key, subject_id,"
                         " model_version, distribution_values, values_sha256)"
                         " VALUES (999999, 'M1.result', 1, ?, '[]', 'x')", (VERSION,))


def test_an_upgrade_keeps_a_copy_of_the_log_as_it_was(log, log_path):
    """Before a log is upgraded in place, a copy of it as it was is kept beside it, write-ahead
    rows included, and never overwritten."""
    from vesta.forecast_log import store

    first = publish(log, pre_match("2026-10-22T08:00:00Z", trigger="first"),
                    match_distributions())  # fmt: skip
    log.execute("PRAGMA user_version = 11")
    log.close()
    open_log(log_path).close()
    copy = log_path.with_name(f"{log_path.name}.before-schema-{store.SCHEMA_VERSION}")
    kept = sqlite3.connect(copy)
    assert kept.execute("PRAGMA user_version").fetchone()[0] == 11
    assert kept.execute("SELECT publication_id FROM publications").fetchall() == [(first,)]
    kept.close()
    written = copy.stat().st_mtime_ns
    reopened = open_log(log_path)  # already upgraded: no copy made again
    reopened.close()
    assert copy.stat().st_mtime_ns == written
