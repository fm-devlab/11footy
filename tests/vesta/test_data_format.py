"""The Vesta data format (VDF 1): the S1 readers run on a dataset that holds only the format's
columns, so a reader that names any other column fails here;
and the capability check v1."""

import sqlite3

import pytest

from vesta.data_format import (
    BLOCKS,
    OUR_DATASET,
    REQUIRED,
    VALUES,
    VERSION,
    FormatError,
    create_statements,
    merge_columns,
    open_dataset,
)
from vesta.jobs.fixtures import read_fixtures
from vesta.spine.ties_check import read_ties
from vesta.spine.world import read_club_matches, read_league_matches, read_nation_matches


def _dataset(path, dataset="someone"):
    connection = sqlite3.connect(path)
    for statement in create_statements("B7"):
        connection.execute(statement)
    connection.execute("INSERT INTO format_version VALUES (?, ?, ?)",
                       (VERSION, dataset, "2026-11-08T00:00:00Z"))  # fmt: skip
    connection.executemany("INSERT INTO competitions VALUES (?, ?, ?, ?)", [
        (1, "XL", "league", "Europe/Rome"), (2, "XC", "cup", "Europe/Rome"),
        (3, "XN", "friendly", "UTC")])  # fmt: skip
    connection.executemany(
        "INSERT INTO seasons VALUES (?, ?, ?, ?)",
        [(1, 1, 2025, "open"), (2, 2, 2025, "open"), (3, 3, 2025, "open")],
    )
    connection.executemany(
        "INSERT INTO teams VALUES (?, ?, ?)",
        [(1, "club", "A"), (2, "club", "B"), (3, "nation", "Italy"), (4, "nation", "Spain")],
    )
    connection.execute("INSERT INTO venues VALUES (1, 'Italy')")
    rows = [  # id, season, kick-off, home, away, status, goals, stage, leg, tie, extra time
        (1, 1, "2025-09-01T18:00:00Z", 1, 2, "played", 2, 1, None, None, None, 0),
        (2, 2, "2025-10-01T18:00:00Z", 1, 2, "played", 1, 1, "round of 16", 1, 1, 0),
        (3, 2, "2025-10-08T18:00:00Z", 2, 1, "played", 2, 1, "round of 16", 2, 1, 1),
        (4, 3, "2025-10-10T18:00:00Z", 3, 4, "played", 0, 0, None, None, None, 0),
        (5, 1, "2025-11-09T18:00:00Z", 2, 1, "scheduled", None, None, None, None, None, 0),
    ]
    for match_id, season, kickoff, home, away, status, hg, ag, stage, leg, tie, extra in rows:
        connection.execute(
            "INSERT INTO matches (match_id, season_id, kickoff_utc, kickoff_precision, local_date,"
            " home_team_id, away_team_id, neutral_venue, status, home_goals, away_goals, stage,"
            " leg, tie_id, extra_time_played, home_goals_regular_time, away_goals_regular_time,"
            " venue_id) VALUES (?, ?, ?, 'minute', ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
            (match_id, season, kickoff, kickoff[:10], home, away, status, hg, ag, stage, leg,
             tie, extra, 1 if extra else None, 1 if extra else None))  # fmt: skip
    connection.execute("INSERT INTO match_ties VALUES (1, 2, 2, 3, 2, 'extra_time')")
    connection.commit()
    connection.close()
    return path


def test_the_s1_readers_need_no_column_outside_the_format(tmp_path):
    connection, header = open_dataset(_dataset(tmp_path / "vdf.db"))
    assert not header.ours
    league = read_league_matches(connection, ["XL"], 2025, 2025)
    assert [(m.match_id, m.home_goals, m.away_goals) for m in league] == [(1, 2, 1)]
    clubs = read_club_matches(connection, ["XL"], ["XC"], 2025, 2025, 2025)
    assert [m.match_id for m in clubs] == [1, 2, 3]
    assert [m.match_id for m in read_nation_matches(connection)] == [4]
    assert [t.tie_id for t in read_ties(connection, ["XC"])] == [1]
    fixtures = read_fixtures(connection, "2025-11-08T00:00:00Z", 48.0, ["XL"], ["XC"])
    assert [f.match_id for f in fixtures] == [5]


def test_a_dataset_without_its_header_is_refused(tmp_path):
    path = tmp_path / "plain.db"
    sqlite3.connect(path).close()
    with pytest.raises(FormatError):
        open_dataset(path)


def test_our_dataset_is_named_by_its_header(tmp_path):
    _, header = open_dataset(_dataset(tmp_path / "ours.db", dataset=OUR_DATASET))
    assert header.ours


def test_every_value_set_names_a_format_column():
    every = merge_columns(*(block for block in BLOCKS if block != REQUIRED))
    for table, column in VALUES:
        assert column in every[table], (table, column)


def _league_history(path, seasons, per_season):
    """Matches of league XL in `seasons`, `per_season` each, between clubs 1 and 2."""
    connection = sqlite3.connect(path)
    match_id = 100
    for season in seasons:
        connection.execute("INSERT OR IGNORE INTO seasons VALUES (?, 1, ?, 'closed')",
                           (10 + season - 2020, season))  # fmt: skip
        for day in range(per_season):
            match_id += 1
            kickoff = f"{season}-{9 + day // 28:02d}-{1 + day % 28:02d}T15:00:00Z"
            connection.execute(
                "INSERT INTO matches (match_id, season_id, kickoff_utc, kickoff_precision,"
                " local_date, home_team_id, away_team_id, neutral_venue, status, home_goals,"
                " away_goals, extra_time_played) VALUES (?, ?, ?, 'minute', ?, 1, 2, 0,"
                " 'played', 1, 0, 0)",
                (match_id, 10 + season - 2020, kickoff, kickoff[:10]),
            )
    connection.commit()
    connection.close()


def test_the_check_says_why_each_competition_is_not_served(tmp_path):
    from vesta.capabilities import NO_NATIONS_FIXTURE, NOT_VALIDATED, check, find_latest_release

    report = check(_dataset(tmp_path / "vdf.db"), find_latest_release())
    assert not report.errors
    reasons = {item.code: item for item in report.competitions}
    assert not reasons["XL"].served and reasons["XL"].reason.startswith("too little history")
    assert "this cup's rules" in reasons["XC"].reason
    assert reasons["XN"].reason == NO_NATIONS_FIXTURE  # served by the nations model with a match
    assert all(item.validation in (None, NOT_VALIDATED) for item in report.competitions)


def test_a_league_with_enough_history_is_served_by_the_baseline_not_validated(tmp_path):
    from vesta.capabilities import NOT_VALIDATED, check, find_latest_release

    path = _dataset(tmp_path / "vdf.db")
    _league_history(path, (2023, 2024), 60)
    item = {c.code: c for c in check(path, find_latest_release()).competitions}["XL"]
    assert item.model == "B0" and item.served and item.validation == NOT_VALIDATED
    assert "not both teams to score" in item.reason


def test_a_value_outside_the_format_is_an_error(tmp_path):
    from vesta.capabilities import check, find_latest_release

    path = _dataset(tmp_path / "vdf.db")
    connection = sqlite3.connect(path)
    connection.execute("UPDATE matches SET status = 'finished' WHERE match_id = 1")
    connection.commit()
    connection.close()
    report = check(path, find_latest_release())
    assert any("status outside" in problem.what for problem in report.errors)
    assert report.competitions == []


def test_the_dataset_caveat_is_on_every_foreign_publication_and_only_there_on_ours():
    from vesta.data_format import Header, make_dataset_caveats

    foreign = Header(VERSION, "someone", "2026-11-08T00:00:00Z")
    ours = Header(VERSION, OUR_DATASET, "2026-11-08T00:00:00Z")
    kinds = ("results", "goals")
    assert [c.code for c in make_dataset_caveats(foreign, "PL", {"PL"}, kinds, "v")] == [
        "not_validated_on_dataset"] * 2
    assert make_dataset_caveats(ours, "PL", {"PL"}, kinds, "v") == []
    assert len(make_dataset_caveats(ours, "XL", {"PL"}, kinds, "v")) == 2


def _foreign_league(path):
    """A foreign league XL of four clubs, two full seasons (2023, 2024) of seeded scores, and one
    match of 2025 to play."""
    import random

    connection = sqlite3.connect(path)
    for statement in create_statements("B7"):
        connection.execute(statement)
    connection.execute("INSERT INTO format_version VALUES (?, 'someone', '2025-09-01T00:00:00Z')",
                       (VERSION,))  # fmt: skip
    connection.execute("INSERT INTO competitions VALUES (1, 'XL', 'league', 'Europe/Rome')")
    for season_id, season in ((1, 2023), (2, 2024), (3, 2025)):
        connection.execute("INSERT INTO seasons VALUES (?, 1, ?, ?)",
                           (season_id, season, "open" if season == 2025 else "closed"))
    connection.executemany("INSERT INTO teams VALUES (?, 'club', ?)",
                           [(team, f"club {team}") for team in range(1, 5)])  # fmt: skip
    generator = random.Random(7)
    match_id = 0
    for season_id, season in ((1, 2023), (2, 2024)):
        day = 0
        for _ in range(9):  # nine rounds of every pairing home and away: 108 matches a season
            for home in range(1, 5):
                for away in range(1, 5):
                    if home == away:
                        continue
                    match_id += 1
                    day += 1
                    kickoff = (f"{season}-{8 + day // 28:02d}-{1 + day % 28:02d}T"
                               f"{12 + match_id % 9:02d}:00:00Z")  # fmt: skip
                    connection.execute(
                        "INSERT INTO matches (match_id, season_id, kickoff_utc, kickoff_precision,"
                        " local_date, home_team_id, away_team_id, neutral_venue, status,"
                        " home_goals, away_goals, extra_time_played) VALUES (?, ?, ?, 'minute',"
                        " ?, ?, ?, 0, 'played', ?, ?, 0)",
                        (match_id, season_id, kickoff, kickoff[:10], home, away,
                         generator.choice((0, 1, 1, 2, 2, 3)), generator.choice((0, 0, 1, 1, 2))))
    connection.execute(
        "INSERT INTO matches (match_id, season_id, kickoff_utc, kickoff_precision, local_date,"
        " home_team_id, away_team_id, neutral_venue, status, extra_time_played) VALUES"
        " (9999, 3, '2025-09-06T15:00:00Z', 'minute', '2025-09-06', 1, 2, 0, 'scheduled', 0)")
    connection.commit()
    connection.close()
    return path


def test_the_dataset_job_publishes_the_baseline_with_the_caveat(tmp_path):
    from vesta.capabilities import BASELINE_KEYS
    from vesta.jobs.dataset_pre_match import run

    dataset = _foreign_league(tmp_path / "foreign.db")
    log_path = tmp_path / "log.db"
    counts = run(dataset, log_path, 168.0, "2025-09-01T08:00:00Z", say=lambda _: None)
    assert counts["published"] == 1
    log = sqlite3.connect(log_path)
    keys = {key for (key,) in log.execute("SELECT DISTINCT quantity_key FROM distributions")}
    assert keys == set(BASELINE_KEYS)  # M4 is not served by the baseline
    caveats = log.execute("SELECT probability_kind, code FROM publication_caveats").fetchall()
    assert sorted(caveats) == [("goals", "not_validated_on_dataset"),
                               ("results", "not_validated_on_dataset")]
    import json
    import tomllib
    from dataclasses import replace

    from vesta.capabilities import ROOT
    from vesta.spine.dixon_coles import DixonColesParameters
    from vesta.spine.timing import TimeRules
    from vesta.spine.walk_forward import WalkForwardSettings, forecast_with_baseline
    from vesta.spine.world import read_league_matches

    published = dict(json.loads(log.execute(
        "SELECT distribution_values FROM distributions WHERE quantity_key = 'M1.result'"
    ).fetchone()[0]))  # fmt: skip
    # the same match through B0's verdict code, as if played, forecast before its kick-off
    played = sqlite3.connect(dataset)
    played.execute("UPDATE matches SET status = 'played', home_goals = 1, away_goals = 0"
                   " WHERE match_id = 9999")
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    settings = replace(WalkForwardSettings.from_config(spine), competitions=("XL",))
    forecast = [item for item in forecast_with_baseline(
        read_league_matches(played, ["XL"], 2014, 2025), TimeRules.from_config(spine),
        DixonColesParameters.from_config(spine), settings, (2025,))
        if item.match_id == 9999][0]  # fmt: skip
    played.close()
    assert [published[side] for side in ("home", "draw", "away")] == pytest.approx(
        forecast.probabilities, abs=1e-12)
    again = run(dataset, log_path, 168.0, "2025-09-01T09:00:00Z", say=lambda _: None)
    assert again["published"] == 0  # one a day per match


def test_the_code_version_needs_no_git(tmp_path):
    from vesta.code_version import read_code_version

    (tmp_path / "config/vesta/releases").mkdir(parents=True)
    (tmp_path / "src/vesta").mkdir(parents=True)
    for version in ("2.10.0", "2.9.0"):
        (tmp_path / f"config/vesta/releases/{version}.toml").write_text(
            f'version = "{version}"\n', encoding="utf-8")
    (tmp_path / "src/vesta/model.py").write_text("x = 1\n", encoding="utf-8")
    first = read_code_version(tmp_path)
    assert first.startswith("2.10.0+") and not (tmp_path / ".git").exists()
    (tmp_path / "src/vesta/model.py").write_text("x = 2\n", encoding="utf-8")
    assert read_code_version(tmp_path) != first
