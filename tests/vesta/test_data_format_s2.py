"""VDF S2: the readers of team counts, player counts and the goals' history run on a dataset that
holds only the format's columns, reading layers where the world reads providers."""

import sqlite3
import tomllib
from pathlib import Path

from vesta.data_format import VERSION, create_statements, open_dataset
from vesta.players.model import _read_counted_assists
from vesta.spine.goal_minutes import read_goal_history
from vesta.spine.timing import TimeRules
from vesta.spine.world import read_goal_methods, read_half_time_scores, read_match_stories
from vesta.team_counts.cups import read_cup_count_matches
from vesta.team_counts.model import read_count_matches
from vesta.team_counts.outcomes import read_match_counts

ROOT = Path(__file__).resolve().parents[2]
SPINE = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))


def _dataset(path):
    connection = sqlite3.connect(path)
    for statement in create_statements("B1", "B2", "B3", "B4", "B5", "B6", "B7"):
        connection.execute(statement)
    connection.execute("INSERT INTO format_version VALUES (?, 'someone', ?)",
                       (VERSION, "2026-11-08T00:00:00Z"))  # fmt: skip
    connection.execute("INSERT INTO competitions VALUES (1, 'XL', 'league', 'Europe/Rome')")
    connection.execute("INSERT INTO seasons VALUES (1, 1, 2025, 'open')")
    connection.executemany("INSERT INTO teams VALUES (?, 'club', ?)", [(1, "A"), (2, "B")])
    for match_id, day in ((1, "01"), (2, "08"), (3, "15")):
        connection.execute(
            "INSERT INTO matches (match_id, season_id, kickoff_utc, kickoff_precision, local_date,"
            " home_team_id, away_team_id, neutral_venue, status, home_goals, away_goals,"
            " home_goals_half_time, away_goals_half_time) VALUES (?, 1, ?, 'minute', ?, 1, 2, 0,"
            " 'played', 1, 0, 1, 0)",
            (match_id, f"2025-09-{day}T18:00:00Z", f"2025-09-{day}"))  # fmt: skip
        connection.execute(
            "INSERT INTO match_events (event_id, match_id, sequence, period, minute, kind,"
            " team_id, player_id, shot_id, layer) VALUES (?, ?, 1, 1, 20, 'goal', 1, 10, ?,"
            " 'summary')", (match_id, match_id, match_id))  # fmt: skip
        connection.execute(
            "INSERT INTO shots (shot_id, match_id, team_id, player_id, minute, result, situation,"
            " body_part, is_penalty, zone, layer) VALUES (?, ?, 1, 10, 20, 'goal', 'open_play',"
            " 'head', 0, 'penalty_area', 'commentary')", (match_id, match_id))  # fmt: skip
    # match 1: both layers (the report is the leagues' count of record); match 2: a feed stub
    # (fouls and corners 0 on both sides: unknown but for the cards); match 3: no count
    rows = [(1, 1, "report", 14, 5, 11, 2), (1, 2, "report", 9, 4, 10, 1),
            (1, 1, "feed", 15, 5, 12, 2), (1, 2, "feed", 10, 4, 10, 1),
            (2, 1, "feed", 3, 0, 0, 2), (2, 2, "feed", 2, 0, 0, 3)]  # fmt: skip
    connection.executemany(
        "INSERT INTO match_team_statistics (match_id, team_id, layer, shots, corners, fouls,"
        " yellow_cards, red_cards, offsides) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 1)", rows)
    connection.execute("INSERT INTO player_match_statistics (match_id, team_id, player_id,"
                       " assists) VALUES (1, 1, 11, 1)")  # fmt: skip
    connection.commit()
    connection.close()
    return path


def test_the_team_counts_read_the_report_first_then_the_feed_and_know_a_stub(tmp_path):
    connection, _ = open_dataset(_dataset(tmp_path / "vdf.db"))
    rules = TimeRules.from_config(SPINE)
    shots = {m.match_id: (m.home_count, m.away_count, m.layer)
             for m in read_count_matches(connection, ["XL"], 2025, 2025, "shots", rules)}
    assert shots == {1: (14, 9, "report"), 2: (None, None, None), 3: (None, None, None)}
    cards = {m.match_id: (m.home_count, m.away_count)
             for m in read_count_matches(connection, ["XL"], 2025, 2025, "cards", rules)}
    assert cards[2] == (2, 3)  # a stub's cards are counted
    assert read_match_counts(connection, {1, 2, 3}, "shots", "leagues") == {1: (14, 9, "report")}
    assert read_match_counts(connection, {1, 2, 3}, "shots", "cups") == {1: (15, 10, "feed")}
    # the cups' reader runs on the format's columns (its leagues and cups are our codes: none
    # here, so it finds no match, but every query has run)
    assert read_cup_count_matches(connection, SPINE, 2025, 2025, "shots") == []


def test_the_goals_history_reads_the_format_alone(tmp_path):
    connection, _ = open_dataset(_dataset(tmp_path / "vdf.db"))
    history = read_goal_history(connection, ["XL"], 2025, 2025)
    assert history.match_id.tolist() == [1, 2, 3]
    assert history.home_goals.sum(axis=1).tolist() == [1, 1, 1]
    assert read_goal_methods(connection, ["XL"], 2025, 2025) == ["header"] * 3
    assert len(read_half_time_scores(connection, ["XL"], 2025, 2025).home_goals) == 3
    assert read_match_stories(connection, {1, 2, 3})


def test_the_players_read_every_player_row_of_a_dataset(tmp_path):
    connection, _ = open_dataset(_dataset(tmp_path / "vdf.db"))
    assert [tuple(row) for row in _read_counted_assists(connection, {1})] == [(1, 11, 1)]
