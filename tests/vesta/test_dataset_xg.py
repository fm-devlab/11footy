"""S3a: our xG written into the dataset by shot id, read back as given, and the champion served
by the rule of 9 November 2026."""

import sqlite3
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from vesta.capabilities import NOT_VALIDATED, VALIDATED, assess_competitions
from vesta.data_format import VERSION, Header, create_statements
from vesta.expected_goals.dataset import is_dataset, read_rated_shots
from vesta.jobs.dataset_xg import Refused, run, write_expected_goals
from vesta.players.model import read_shot_players

ROOT = Path(__file__).resolve().parents[2]
SPINE = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
BUILT_AT = "2026-11-08T00:00:00Z"


def _dataset(path, code="PL", seasons=(2023, 2024, 2025), per_season=60):
    """League `code` between clubs 1 and 2, `per_season` matches a season, two shots a match:
    a tracked one by the home side, a commentary one by the away side."""
    connection = sqlite3.connect(path)
    for statement in create_statements("B3", "B5", "B7"):
        connection.execute(statement)
    connection.execute("INSERT INTO format_version VALUES (?, 'someone', ?)", (VERSION, BUILT_AT))
    connection.execute("INSERT INTO competitions VALUES (1, ?, 'league', 'Europe/London')",
                       (code,))  # fmt: skip
    connection.executemany("INSERT INTO teams VALUES (?, 'club', ?)", [(1, "A"), (2, "B")])
    match_id = 0
    for season in seasons:
        connection.execute("INSERT INTO seasons VALUES (?, 1, ?, 'closed')", (season, season))
        for day in range(per_season):
            match_id += 1
            kickoff = f"{season}-{9 + day // 28:02d}-{1 + day % 28:02d}T15:00:00Z"
            connection.execute(
                "INSERT INTO matches (match_id, season_id, kickoff_utc, kickoff_precision,"
                " local_date, home_team_id, away_team_id, neutral_venue, status, home_goals,"
                " away_goals) VALUES (?, ?, ?, 'minute', ?, 1, 2, 0, 'played', 1, 0)",
                (match_id, season, kickoff, kickoff[:10]))  # fmt: skip
            for shot, team, player, layer in ((2 * match_id + 1, 1, 10, "tracked"),
                                              (2 * match_id, 2, 20, "commentary")):  # fmt: skip
                connection.execute(
                    "INSERT INTO shots (shot_id, match_id, team_id, player_id, minute, result,"
                    " situation, body_part, is_penalty, layer) VALUES (?, ?, ?, ?, 30, 'saved',"
                    " 'open_play', 'head', 0, ?)",
                    (shot, match_id, team, player, layer))  # fmt: skip
    connection.commit()
    return connection


def test_the_chances_are_written_by_shot_id_and_read_back_as_given(tmp_path):
    connection = _dataset(tmp_path / "ours.db")
    tracked = [row[0] for row in connection.execute(
        "SELECT shot_id FROM shots WHERE layer = 'tracked' ORDER BY shot_id DESC")]  # fmt: skip
    chances = np.linspace(0.01, 0.6, len(tracked))
    assert write_expected_goals(connection, np.array(tracked), chances) == len(tracked)
    shots, given = read_rated_shots(connection, ["PL"], 2023, 2025)
    by_id = dict(zip(tracked, chances.tolist(), strict=True))
    assert len(shots) == len(tracked) and set(shots.home.tolist()) == {1}
    assert given.tolist() == [by_id[shot] for shot in sorted(tracked)]  # match, then shot order
    assert list(shots.match_id) == sorted(shots.match_id)
    assert is_dataset(connection)
    players = read_shot_players(connection, ["PL"], 2023, 2025, ["tracked"])
    assert len(players) == len(shots) and {row[0] for row in players} == {10}


def test_a_chance_for_a_shot_the_dataset_lacks_or_does_not_track_is_refused(tmp_path):
    connection = _dataset(tmp_path / "ours.db")
    with pytest.raises(Refused, match="not in the dataset"):
        write_expected_goals(connection, np.array([999_999]), np.array([0.1]))
    with pytest.raises(Refused, match="not tracked"):
        write_expected_goals(connection, np.array([2]), np.array([0.1]))  # a commentary shot
    assert connection.execute("SELECT count(xg) FROM shots").fetchone()[0] == 0


def test_a_dataset_of_another_build_is_refused(tmp_path):
    _dataset(tmp_path / "ours.db").close()
    world = sqlite3.connect(tmp_path / "world.db")
    world.execute("CREATE TABLE build_information (key TEXT, value TEXT)")
    world.execute("INSERT INTO build_information VALUES ('built_at', '2026-11-01T00:00:00Z')")
    world.commit()
    world.close()
    with pytest.raises(Refused, match="not its export"):
        run(tmp_path / "world.db", tmp_path / "ours.db")


def _assess(connection, ours: bool):
    present = defaultdict(lambda: False, B3=True, B5=True)
    header = Header(VERSION, "11footy" if ours else "someone", BUILT_AT)
    proved = {"PL"} if ours else set()
    found = assess_competitions(connection, header, present, SPINE, {"PL": "spine"}, proved)
    return {item.code: item for item in found}


def test_the_champion_reads_a_foreign_xg_with_the_caveat_and_ours_validated(tmp_path):
    connection = _dataset(tmp_path / "foreign.db")
    connection.execute("UPDATE shots SET xg = 0.1 WHERE layer = 'tracked'")
    foreign = _assess(connection, ours=False)["PL"]
    assert (foreign.model, foreign.served, foreign.validation) == (
        "champion", True, NOT_VALIDATED)
    assert _assess(connection, ours=True)["PL"].validation == VALIDATED


def test_without_xg_the_champion_is_not_served_and_the_league_falls_back_to_b0(tmp_path):
    item = _assess(_dataset(tmp_path / "foreign.db"), ours=False)["PL"]
    assert item.model == "B0" and item.served and item.validation == NOT_VALIDATED
    assert item.reason.startswith("the basic model, as the dataset's shots carry no expected")
