import shutil
import sqlite3
from datetime import UTC, datetime

from desk_datasets import THEIR_COMPETITION, build_theirs
from small_world import ARSENAL, AWARDED, BOURNEMOUTH, COMPETITION, PLAYED, TO_COME

from desk import reading
from desk.order import read_competition_shown
from desk.reading import Reader, find_log, find_tab, name_cut

BEFORE = datetime(2026, 10, 24, 12, tzinfo=UTC)


def test_our_dataset_names_its_competitions(ours):
    reader = Reader(ours, find_log(ours))
    assert reader.ours and reader.name == "11footy"
    names = {c["competition_id"]: c["name"] for c in reader.list_competitions()}
    assert names[COMPETITION] == "Premier League"


def test_ours_lists_only_what_the_order_file_shows(ours, monkeypatch):
    """`always` stays, `no` and an unnamed code go; `around_matches` only from 90 days before
    a match to 30 days after (the fixture's cup match is on 12 March 2025)."""
    reader = Reader(ours, find_log(ours))

    def listed(rules: dict, now: datetime) -> list[str]:
        monkeypatch.setattr(reading, "read_competition_shown", lambda: rules)
        return [c["name"] for c in reader.list_competitions(now)]

    assert listed({"PL": "always", "CL": "no"}, BEFORE) == ["Premier League"]
    assert listed({"PL": "always"}, BEFORE) == ["Premier League"]
    tournament = {"PL": "always", "CL": "around_matches"}
    assert len(listed(tournament, datetime(2024, 12, 20, tzinfo=UTC))) == 2
    assert len(listed(tournament, datetime(2025, 4, 5, tzinfo=UTC))) == 2
    assert listed(tournament, datetime(2024, 11, 1, tzinfo=UTC)) == ["Premier League"]
    assert listed(tournament, BEFORE) == ["Premier League"]


def test_the_shipped_order_file_names_a_rule_for_each_competition():
    assert set(read_competition_shown().values()) <= {"always", "around_matches", "no"}
    assert read_competition_shown()["SA"] == "always"
    assert read_competition_shown()["WC"] == "around_matches"
    assert read_competition_shown()["CLQ"] == "no"


def test_the_current_round_carries_its_result_chances(ours):
    reader = Reader(ours, find_log(ours))
    kicked_off = datetime(2026, 10, 25, 16, tzinfo=UTC)
    found = reader.read_current_round(COMPETITION, now=kicked_off)
    # round 2 by its matchday (the build numbers a league's rounds since 8 November 2026): the
    # awarded match of August and the one in play
    assert [match["match_id"] for match in found.matches] == [AWARDED, TO_COME]
    assert found.matches[1]["status"] == "In play", "a match stays at its kick-off"
    chances = reader.list_result_chances(found.matches[1])
    assert len(chances) == 3
    assert sum(int(chance.strip("<>%")) for chance in chances) == 100


def test_once_a_round_is_finished_the_next_one_shows_and_the_season_stays_at_hand(ours):
    reader = Reader(ours, find_log(ours))
    # matchday 1 is over in the dataset: matchday 2 is current a day before its kick-off
    found = reader.read_current_round(COMPETITION, now=BEFORE)
    assert [match["match_id"] for match in found.matches] == [AWARDED, TO_COME]
    assert (found.place, found.index, found.count) == ("current", 1, 2)
    season = reader.read_rounds(COMPETITION, now=BEFORE)
    before = reader.read_round(season, 0, now=BEFORE)
    assert before.place == "past" and before.label == "Matchday 1"
    assert [match["status"] for match in before.matches] == ["Finished 2-1", "Finished 0-0"]
    assert reader.list_result_chances(before.matches[0]), "a past round keeps its forecasts"


def test_a_match_shows_every_forecast_of_its_newest_publication(ours):
    view = Reader(ours, find_log(ours)).read_match(TO_COME)
    assert view.summary["home_team"]["team_id"] == ARSENAL
    assert view.summary["away_team"]["team_id"] == BOURNEMOUTH
    assert view.summary["competition"] == "Premier League"
    assert view.published_at is not None
    result = view.tabs["Result"][0]
    home, away = view.summary["home_team"]["name"], view.summary["away_team"]["name"]
    assert [row.line for row in result.children][:3] == [home, "Draw", away]
    assert all(row.chance.endswith("%") for row in result.children)


def test_our_season_table_stands_beside_its_chances(ours):
    view = Reader(ours, find_log(ours)).read_season(COMPETITION, now=BEFORE)
    assert view.season_name == "2026-27", "the season's own name"
    assert view.rows, "our dataset carries the standings"
    assert view.columns[:5] == ["Place", "Team", "Played", "Points", "Champion"]
    assert all(len(row) == len(view.columns) for row in view.rows)


def copy_ours(ours, tmp_path):
    copy = tmp_path / "11footy.db"
    shutil.copy(ours, copy)
    return copy


def test_a_season_played_in_groups_names_each_team_s_group(ours, tmp_path):
    data = copy_ours(ours, tmp_path)
    with sqlite3.connect(data) as connection:
        connection.execute("update standings set league_name = 'League A', group_name ="
                           " case when team_id in (82, 213) then 'Group A1' else 'Group A2' end")
    view = Reader(data, find_log(data)).read_season(COMPETITION, now=BEFORE)
    assert view.columns[:3] == ["Group", "Place", "Team"]
    assert [row[0] for row in view.rows] == ["League A, Group A1"] * 2 + ["League A, Group A2"] * 2
    assert not view.partial
    # one group alone in a whole table needs no column
    with sqlite3.connect(data) as connection:
        connection.execute("update standings set group_name = 'Group A1'")
    view = Reader(data, find_log(data)).read_season(COMPETITION, now=BEFORE)
    assert view.columns[0] == "Place"


def test_a_table_of_part_of_the_teams_says_so(ours, tmp_path):
    """The standings of one group of four teams playing: the table is partial, its group named."""
    data = copy_ours(ours, tmp_path)
    with sqlite3.connect(data) as connection:
        connection.execute("delete from standings where team_id in (652, 773)")
        connection.execute("update standings set group_name = 'Group C'")
    view = Reader(data, find_log(data)).read_season(COMPETITION, now=BEFORE)
    assert (view.teams_in_table, view.teams_playing) == (2, 4) and view.partial
    assert view.columns[0] == "Group" and view.rows[0][0] == "Group C"


def test_the_season_is_the_one_the_matches_screen_shows(ours, tmp_path):
    """Not the newest: a season to come with one match, or one without matches or a table, is not
    the season of now; a competition without matches shows its newest season with a table."""
    data = copy_ours(ours, tmp_path)
    with sqlite3.connect(data) as connection:
        connection.executemany(
            "insert into seasons (season_id, competition_id, season, name) values (?, ?, ?, ?)",
            [(7, COMPETITION, 2027, "2027-28"), (8, COMPETITION, 2028, None),
             (9, 9, 2020, None), (10, 9, 2021, None)],
        )
        connection.execute(
            "insert into matches (match_id, season_id, kickoff_utc, home_team_id, away_team_id,"
            " status) values (901, 7, '2027-08-21T14:00:00Z', ?, ?, 'scheduled')",
            (ARSENAL, BOURNEMOUTH),
        )
        connection.execute("insert into competitions (competition_id, code, name, kind)"
                           " values (9, 'OLDCUP', 'Old Cup', 'cup')")
        connection.executescript(
            "create temp table copied as select * from standings where season_id = 1;"
            " update copied set season_id = 9; insert into standings select * from copied;"
        )
    reader = Reader(data, find_log(data))
    assert reader.read_season(COMPETITION, now=BEFORE).season_name == "2026-27"
    assert reader.read_season(COMPETITION, now=datetime(2027, 8, 1, tzinfo=UTC)).rows == []
    old = reader.read_season(9, now=BEFORE)
    assert old.season_name == "2020" and old.rows


def test_their_dataset_falls_back_to_codes_and_no_table(theirs):
    reader = Reader(theirs, find_log(theirs))
    assert not reader.ours and reader.name == "someone"
    assert [c["name"] for c in reader.list_competitions()] == ["XL"]
    matches = reader.read_current_round(THEIR_COMPETITION, now=BEFORE).matches
    assert [match["home_team"]["name"] for match in matches] == ["Club A"]
    assert matches[0]["status"] == "To play"
    assert reader.read_player_name(4001) == "player 4001"
    view = reader.read_season(THEIR_COMPETITION)
    assert view.rows == [] and view.columns == []


def test_only_competitions_played_in_the_last_four_years_are_listed(tmp_path):
    """Built on 20 October 2026: a cup last played in 2022-11 stays, one of 1930 or one
    without a match goes."""
    data = build_theirs(tmp_path / "old.db")
    with sqlite3.connect(data) as connection:
        connection.executemany(
            "insert into competitions values (?, ?, 'cup', 'UTC')",
            [(91, "RECENT"), (92, "OLD"), (93, "EMPTY")],
        )
        connection.executemany(
            "insert into seasons values (?, ?, ?, 'closed')", [(91, 91, 2022), (92, 92, 1930)]
        )
        connection.executemany(
            "insert into matches values (?, ?, ?, 'minute', ?, ?, ?, 0, 'played', 1, 0, 1, 0)",
            [(9101, 91, "2022-11-01T18:00:00Z", "2022-11-01", ARSENAL, BOURNEMOUTH),
             (9201, 92, "1930-07-13T15:00:00Z", "1930-07-13", ARSENAL, BOURNEMOUTH)],
        )
    names = [c["name"] for c in Reader(data, find_log(data)).list_competitions()]
    assert sorted(names) == ["RECENT", "XL"]


def test_without_a_log_nothing_is_computed_yet(tmp_path):
    bare = build_theirs(tmp_path / "bare.db")
    reader = Reader(bare, find_log(bare))
    matches = reader.read_current_round(THEIR_COMPETITION, now=BEFORE).matches
    assert reader.list_result_chances(matches[0]) == []
    assert reader.read_match(TO_COME).published_at is None


def test_each_catalogue_item_has_its_tab():
    assert find_tab("M1") == "Result"
    assert find_tab("M15") == "Corners"
    assert [find_tab(code) for code in ("M19", "M20", "M21")] == ["Offsides", "Shots", "Fouls"]
    assert find_tab("L1") == "Lineups"
    assert find_tab("P3") == "Players"
    assert find_tab("M30") == "Other"


def test_season_lines_name_their_cut():
    assert name_cut("place=1") == "Champion"
    assert name_cut("top=4") == "Top 4"
    assert name_cut("bottom=3") == "Bottom 3"
    assert name_cut("place=2") is None


def test_a_played_match_shows_its_forecast_before_kickoff(ours):
    """The log holds the match's live and full-time publications too; the window, made for the
    time before the match, shows the last one before kick-off."""
    reader = Reader(ours, find_log(ours))
    view = reader.read_match(PLAYED)
    assert view.published_at is not None
    publication = reader.log.find_latest_pre_match_publication(PLAYED)
    assert view.published_at == publication["published_at"]
    assert publication["stage"] == "pre_match"


def test_a_competition_is_active_around_its_matches(ours):
    reader = Reader(ours, find_log(ours))
    assert reader.is_active(COMPETITION, now=BEFORE)
    assert not reader.is_active(COMPETITION, now=datetime(2027, 6, 1, tzinfo=UTC))
