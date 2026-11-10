"""The families beyond result and goals in the user's run (`vesta.run`, `vesta.round_families`):
the team counts, goals in time, the players and the lineups published for the round, each from
the release's part, valid, and with the caveats of a dataset no verdict covers; a family the
release does not cover is reported as not served, with its reason."""

import json
import sqlite3
import tomllib
from pathlib import Path

import pytest
from tests.vesta.detailed_league import write_detailed_league

from vesta.capabilities import (
    NOT_VALIDATED,
    VALIDATED,
    Competition,
    assess_families,
    describe_families,
    read_family_parts,
)
from vesta.distributions import check_distribution
from vesta.release_gate import find_latest_plan

NOW = "2025-08-25T08:00:00Z"  # after the league's fourth round, before its fifth
PRESENT = {block: True for block in ("B1", "B2", "B3", "B4", "B5", "B6", "B7")}


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    """The run on the detailed league, once: its log and its counts. The season part is left out
    (its tests are the season job's): a league of 14 clubs is not the release's 20."""
    import io

    from vesta import run as user_run

    dataset = write_detailed_league(tmp_path_factory.mktemp("league") / "league.db")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(user_run, "read_season_part", lambda root: ((), []))
        counts = user_run.run([dataset], now=NOW, out=io.StringIO())
    log = sqlite3.connect(dataset.with_name(dataset.name + ".forecast-log.db"))
    yield log, counts
    log.close()


def _check_family(log, family: str) -> set[str]:
    """Every distribution of the family's keys valid in its kind's shape, the keys published
    those of the release's part for PL, each fixture with them; returns the keys."""
    keys = set(read_family_parts(find_latest_plan())[family]["PL"])
    kinds = dict(log.execute("SELECT quantity_key, distribution_kind FROM quantities"))
    rows = log.execute("SELECT p.match_id, d.quantity_key, d.distribution_values FROM"
                       " distributions d JOIN publications p USING (publication_id)").fetchall()
    found = {}
    for match_id, key, values in rows:
        if key in keys:
            check_distribution(kinds[key], json.loads(values))
            found.setdefault(match_id, set()).add(key)
    fixtures = {row[0] for row in log.execute("SELECT match_id FROM publications")}
    assert set(found) == fixtures and len(fixtures) == 7  # the fifth round: 14 clubs, 7 matches
    published_keys = set().union(*found.values())
    assert published_keys <= keys
    return published_keys


def _caveat_codes(log, kind: str) -> set[str]:
    return {row[0] for row in log.execute(
        "SELECT DISTINCT code FROM publication_caveats WHERE probability_kind = ?", (kind,))}


def test_the_run_publishes_the_team_counts(published):
    log, counts = published
    keys = _check_family(log, "team counts")
    assert {"M15.corners.team", "M17.cards", "M20.shots", "M21.fouls", "M19.offsides"} <= keys
    assert counts["families"]["team counts"] == 7
    assert _caveat_codes(log, "team_counts") == {"not_validated_on_dataset"}


def test_the_run_publishes_goals_in_time(published):
    log, counts = published
    keys = _check_family(log, "goals in time")
    assert {"M8.first_goal_time", "M9.half_time_full_time", "M10.result.second_half"} <= keys
    assert counts["families"]["goals in time"] == 7
    assert _caveat_codes(log, "timing") == {"not_validated_on_dataset"}


def test_the_run_publishes_the_players(published):
    log, counts = published
    keys = _check_family(log, "players")
    assert {"P3.scores", "P3.first_scorer", "P1.shots", "P9.booked"} <= keys
    assert counts["families"]["players"] == 7
    assert _caveat_codes(log, "scorers") == {"not_validated_on_dataset",
                                             "suspensions_not_included", "injuries_not_included"}


def test_the_run_publishes_the_lineups_and_the_expected_elevens(published):
    log, counts = published
    keys = _check_family(log, "lineups")
    assert keys == {"L1.starts", "L6.minutes", "L9.position", "L2.formation.team"}
    assert counts["families"]["lineups"] == 7
    assert log.execute("SELECT count(*) FROM expected_elevens").fetchone()[0] == 7 * 2 * 11
    assert "not_validated_on_dataset" in _caveat_codes(log, "lineups")


def _spine() -> dict:
    root = Path(__file__).resolve().parents[2]
    return tomllib.loads((root / "config/vesta/spine.toml").read_text(encoding="utf-8"))


def test_a_family_the_release_does_not_cover_is_not_served_with_its_reason():
    found = [Competition("ELC", "league", 2025, "champion", True, "", NOT_VALIDATED),
             Competition("CL", "cup", 2025, "cups Elo", True, "", NOT_VALIDATED)]
    entries = assess_families(found, PRESENT, _spine(), find_latest_plan(), NOW, True)
    by_pair = {(entry.code, entry.family): entry for entry in entries}
    for family in ("team counts", "goals in time", "players", "lineups"):
        assert not by_pair[("ELC", family)].served
        assert "released for the five top leagues" in by_pair[("ELC", family)].reason
    assert by_pair[("CL", "team counts")].served  # the cups' team counts are released
    assert not by_pair[("CL", "lineups")].served
    assert "Lineups: not served" in describe_families(entries, "CL")


def test_a_family_says_what_the_dataset_lacks_and_validates_only_on_our_five_leagues():
    spine = _spine()
    leagues = spine["walk_forward"]["competitions"]
    found = [Competition("PL", "league", 2025, "champion", True, "", VALIDATED)]
    lacking = assess_families(found, {**PRESENT, "B4": False}, spine, find_latest_plan(), NOW,
                              False, set(leagues))  # fmt: skip
    by_family = {entry.family: entry for entry in lacking}
    assert by_family["team counts"].reason.endswith("the dataset holds no teams' match statistics")
    assert by_family["players"].served and "by their number" in by_family["players"].reason
    assert "as of 2025-08-25T08:00:00Z" in by_family["lineups"].reason
    assert all(entry.validation == VALIDATED for entry in lacking if entry.served)
    alone = assess_families(found, PRESENT, spine, find_latest_plan(), NOW, True, {"PL"})
    assert all(entry.validation == NOT_VALIDATED for entry in alone)


def _nations(path):
    """Eight national teams, every pairing once a year from 2014 to 2024 in friendlies, a
    qualifying competition whose matches are all played, and four friendlies to play in
    October 2025."""
    import random

    from vesta.data_format import VERSION, create_statements

    connection = sqlite3.connect(path)
    for statement in create_statements("B7"):
        connection.execute(statement)
    connection.execute("INSERT INTO format_version VALUES (?, 'someone', '2025-10-01T00:00:00Z')",
                       (VERSION,))  # fmt: skip
    connection.execute("INSERT INTO competitions VALUES (1, 'FRIENDLY', 'friendly', 'UTC')")
    connection.execute("INSERT INTO competitions VALUES (2, 'XQ', 'nations qualifying', 'UTC')")
    connection.executemany("INSERT INTO seasons VALUES (?, ?, ?, 'closed')",
                           [(year - 2000, 1, year) for year in range(2014, 2026)]
                           + [(99, 2, 2014)])  # fmt: skip
    connection.executemany("INSERT INTO teams VALUES (?, 'nation', ?)",
                           [(team, f"nation {team}") for team in range(1, 9)])  # fmt: skip
    generator = random.Random(3)
    match_id = 0

    def add(season_id, day, home, away, played):
        nonlocal match_id
        match_id += 1
        connection.execute(
            "INSERT INTO matches (match_id, season_id, kickoff_utc, kickoff_precision, local_date,"
            " home_team_id, away_team_id, neutral_venue, status, home_goals, away_goals,"
            " extra_time_played) VALUES (?, ?, ?, 'minute', ?, ?, ?, 0, ?, ?, ?, 0)",
            (match_id, season_id, f"{day}T18:00:00Z", day, home, away,
             "played" if played else "scheduled",
             generator.choice((0, 1, 1, 2, 3)) if played else None,
             generator.choice((0, 0, 1, 1, 2)) if played else None))  # fmt: skip

    for year in range(2014, 2025):
        pairs = [(home, away) for home in range(1, 9) for away in range(home + 1, 9)]
        for number, (home, away) in enumerate(pairs):
            add(year - 2000, f"{year}-{1 + number // 3:02d}-{1 + number % 3 * 9:02d}", home, away,
                True)
    add(99, "2014-12-20", 1, 2, True)
    for number, (home, away) in enumerate(((1, 2), (3, 4), (5, 6), (7, 8))):
        add(25, f"2025-10-{10 + number % 2:02d}", home, away, False)
    connection.commit()
    connection.close()
    return path


def test_the_national_teams_are_forecast_by_the_nations_model_marked_not_validated(tmp_path):
    import io

    from vesta.capabilities import FAILED, NO_NATIONS_FIXTURE, SPINE_KEYS
    from vesta.run import run

    dataset = _nations(tmp_path / "nations.db")
    counts = run([dataset], now="2025-10-01T08:00:00Z", out=io.StringIO())
    assert counts["published"] == 4
    log = sqlite3.connect(tmp_path / "nations.db.forecast-log.db")
    keys = {row[0] for row in log.execute("SELECT DISTINCT quantity_key FROM distributions")}
    assert keys == set(SPINE_KEYS)
    publications = log.execute("SELECT count(*) FROM publications").fetchone()[0]
    marked = log.execute("SELECT count(DISTINCT publication_id) FROM publication_caveats WHERE"
                         " code = 'not_validated_on_dataset'").fetchone()[0]  # fmt: skip
    assert marked == publications == 4
    log.close()
    from vesta.capabilities import assess_competitions, find_present_blocks
    from vesta.data_format import open_dataset

    connection, header = open_dataset(dataset)
    found = {item.code: item for item in assess_competitions(
        connection, header, find_present_blocks(connection), _spine(), {})}  # fmt: skip
    connection.close()
    assert found["FRIENDLY"].served and found["FRIENDLY"].validation == FAILED
    assert "did not pass Vesta's tests" in found["FRIENDLY"].reason
    assert not found["XQ"].served and found["XQ"].reason == NO_NATIONS_FIXTURE


def test_on_one_league_proved_ours_the_spine_is_released_and_the_families_say_why_not(tmp_path):
    """The release's code on a file proved ours holding one of the five leagues: its result and
    goals carry the release's version and no caveat; the families, whose fit pools over the five
    leagues, carry the dataset's version and `not_validated_on_dataset`."""
    import io

    from vesta import run as user_run
    from vesta.dataset_proof import Proof

    dataset = write_detailed_league(tmp_path / "league.db", dataset="11footy")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(user_run, "read_season_part", lambda root: ((), []))
        patch.setattr(user_run, "prove", lambda item: Proof(True, "signed", "signed"))
        patch.setattr(user_run, "check_manifest", lambda root: ("release", "the release"))
        counts = user_run.run([dataset], now=NOW, out=io.StringIO())
    assert counts["proved"] == ["PL"] and counts["published"] == 7
    log = sqlite3.connect(tmp_path / "league.db.forecast-log.db")
    versions = dict(log.execute("SELECT DISTINCT quantity_key, model_version FROM distributions"))
    assert versions["M1.result"] == "vesta-2.7.0" and versions["M4.both_score"] == "vesta-2.7.0"
    assert versions["M15.corners.team"] == versions["L1.starts"] == "vesta-2.7.0-dataset"
    caveated = {row for row in log.execute("SELECT DISTINCT probability_kind, model_version FROM"
                                           " publication_caveats WHERE code ="
                                           " 'not_validated_on_dataset'")}  # fmt: skip
    assert ("team_counts", "vesta-2.7.0-dataset") in caveated
    assert all(version == "vesta-2.7.0-dataset" for _, version in caveated)
    log.close()
