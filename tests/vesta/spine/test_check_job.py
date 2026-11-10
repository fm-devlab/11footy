"""The check job: what happened to each forecast of record, its lines checked once (again only
when the world changes its answer), a cancelled or awarded match void, and the figures: each key
scored on its own beside the reference, each probability kind summarised by its keys' skill."""

import json
from pathlib import Path

import numpy as np

from vesta.forecast_log.store import open_log
from vesta.forecast_log.writing import (
    Publication,
    PublishedDistribution,
    publish,
    record_forecasts_of_record,
    register_model_version,
    register_published_quantity,
    register_quantity,
)
from vesta.jobs.check import keep_checkable, run, settle_spine_outcomes
from vesta.quantities import load_quantities
from vesta.spine.publication import build_spine_distributions
from vesta.spine.score_matrix import Calibration, score_matrices

DEFINITIONS = load_quantities()
MATRIX = score_matrices(np.array([1.5]), np.array([1.1]), Calibration())[0]
VERSION = "vesta-2.0.0"


def _forecast(log, world, match_id: int, version: str = VERSION, matrix=MATRIX) -> None:
    home, away, kickoff = world.execute(
        "SELECT home_team_id, away_team_id, kickoff_utc FROM matches WHERE match_id = ?",
        (match_id,)).fetchone()  # fmt: skip
    publish(log, Publication(stage="pre_match", trigger="first",
                             published_at=kickoff[:10] + "T08:00:00Z", code_commit="test",
                             match_id=match_id, scheduled_kickoff_utc=kickoff),
            build_spine_distributions(matrix, match_id, home, away, version))  # fmt: skip
    record_forecasts_of_record(log, match_id, kickoff, kickoff)


def _publish_at(log, world, match_id: int, published_at: str, trigger: str = "first",
                matrix=MATRIX) -> None:
    home, away, kickoff = world.execute(
        "SELECT home_team_id, away_team_id, kickoff_utc FROM matches WHERE match_id = ?",
        (match_id,)).fetchone()  # fmt: skip
    publish(log, Publication(stage="pre_match", trigger=trigger, published_at=published_at,
                             code_commit="test", match_id=match_id, scheduled_kickoff_utc=kickoff),
            build_spine_distributions(matrix, match_id, home, away, VERSION))  # fmt: skip


def test_the_check_records_the_last_forecast_before_each_kickoff_once(world, tmp_path: Path):
    """A match forecast twice before its kick-off and once after: the check records the second,
    never the one after; a second run writes nothing and rewrites nothing."""
    world_path, log_path = tmp_path / "football-world.db", tmp_path / "forecast-log.db"
    log = open_log(log_path)
    for item in DEFINITIONS.values():
        register_quantity(log, item.quantity)
    register_model_version(log, VERSION, "spine", "2017-07-01T00:00:00Z", "the spine")
    for key in ("M1.result", "M2.score", "M3.goals", "M4.both_score", "M5.goals.team"):
        register_published_quantity(log, key, 1, VERSION, "2017-07-01T00:00:00Z", True, False)
    played = _played_match(world)
    kickoff = world.execute("SELECT kickoff_utc FROM matches WHERE match_id = ?",
                            (played,)).fetchone()[0]  # fmt: skip
    day = kickoff[:10]
    later = score_matrices(np.array([0.9]), np.array([1.6]), Calibration())[0]
    _publish_at(log, world, played, f"{day}T06:00:00Z")
    _publish_at(log, world, played, f"{day}T08:00:00Z", trigger="morning", matrix=later)
    _publish_at(log, world, played, kickoff, trigger="after_kickoff")
    second, after = (row[0] for row in log.execute(
        "SELECT publication_id FROM publications WHERE match_id = ? ORDER BY publication_id"
        " LIMIT 2 OFFSET 1", (played,)))  # fmt: skip
    log.close()

    run(world_path, log_path, "2018-06-01T06:00:00Z", say=lambda _: None)
    log = open_log(log_path)
    recorded = log.execute("SELECT DISTINCT publication_id, real_kickoff_utc, recorded_at FROM"
                           " forecasts_of_record WHERE match_id = ?", (played,)).fetchall()
    assert recorded == [(second, kickoff, "2018-06-01T06:00:00Z")] and second != after
    log.close()
    run(world_path, log_path, "2018-06-08T06:00:00Z", say=lambda _: None)
    log = open_log(log_path)
    assert log.execute("SELECT DISTINCT publication_id, recorded_at FROM forecasts_of_record"
                       " WHERE match_id = ?", (played,)).fetchall() == [
        (second, "2018-06-01T06:00:00Z")]  # fmt: skip
    log.close()


def _played_match(world) -> int:
    return world.execute(
        "SELECT m.match_id FROM matches m JOIN seasons s ON s.season_id = m.season_id"
        " WHERE s.season = 2017 AND s.competition_id = 1 AND m.status = 'played'"
        " ORDER BY m.match_id LIMIT 1").fetchone()[0]  # fmt: skip


def test_the_spines_outcomes_are_the_score_on_the_pitch():
    happened = settle_spine_outcomes(2, 1)
    assert happened[("M1.result", "match")] == "home"
    assert happened[("M2.score", "match")] == (2, 1)
    assert happened[("M3.goals", "match")] == 3 and happened[("M4.both_score", "match")]
    assert happened[("M5.goals.team", "away")] == 1


def test_the_check_of_the_forecasts_of_record(world, tmp_path: Path):
    world_path = tmp_path / "football-world.db"
    log_path = tmp_path / "forecast-log.db"
    log = open_log(log_path)
    for item in DEFINITIONS.values():
        register_quantity(log, item.quantity)
    register_model_version(log, VERSION, "spine", "2017-07-01T00:00:00Z", "the spine")
    for key in ("M1.result", "M2.score", "M3.goals", "M4.both_score", "M5.goals.team"):
        register_published_quantity(log, key, 1, VERSION, "2017-07-01T00:00:00Z", True, False)
    played = _played_match(world)
    world.execute("UPDATE matches SET season_id = (SELECT season_id FROM matches WHERE"
                  " match_id = ?) WHERE match_id = 6", (played,))  # the cancelled one, in 2017
    world.commit()
    for match_id in (played, 6, 5):  # played, cancelled, awarded (the small world's)
        _forecast(log, world, match_id)
    log.close()

    counts = run(world_path, log_path, "2018-06-01T06:00:00Z", say=lambda _: None)
    assert (counts["final"], counts["void"], counts["outcomes"]) == (1, 2, 6)
    log = open_log(log_path)
    lines = log.execute("SELECT COUNT(*) FROM probabilities WHERE publication_id IN (SELECT"
                        " publication_id FROM forecasts_of_record WHERE match_id = ?)",
                        (played,)).fetchone()[0]
    assert counts["line_checks"] == lines > 0
    home, away = world.execute("SELECT home_goals, away_goals FROM matches WHERE match_id = ?",
                               (played,)).fetchone()  # fmt: skip
    stored = dict(log.execute("SELECT quantity_key, outcome_value FROM outcomes WHERE"
                              " quantity_key IN ('M2.score', 'M3.goals')"))  # fmt: skip
    assert json.loads(stored["M2.score"]) == [home, away]
    assert json.loads(stored["M3.goals"]) == home + away
    assert log.execute("SELECT match_id, reason FROM voids ORDER BY match_id").fetchall() == [
        (5, "awarded"), (6, "cancelled")]  # the awarded score is not checked, the spine's too
    keys = log.execute("SELECT DISTINCT quantity_key FROM track_record_key_scores"
                       " WHERE period = 'all' AND competition_id = 0").fetchall()  # fmt: skip
    assert {key for (key,) in keys} == {"M1.result", "M2.score", "M3.goals", "M4.both_score",
                                        "M5.goals.team"}  # fmt: skip
    summary = log.execute("SELECT probability_kind, metric, value, void_count FROM"
                          " track_record_scores WHERE period = 'all' AND competition_id = 0"
                          " AND model_version = ? ORDER BY probability_kind",
                          (VERSION,)).fetchall()  # fmt: skip
    assert [(kind, metric, voids) for kind, metric, _, voids in summary] == [
        ("goals", "skill", 10), ("results", "skill", 2)]  # the two void matches' forecasts
    goals_skill = log.execute(
        "SELECT AVG(skill) FROM track_record_key_scores WHERE period = 'all' AND competition_id"
        " = 0 AND probability_kind = 'goals' AND metric IN ('log_loss', 'ranked_probability_score')"
        " AND NOT (quantity_key IN ('M3.goals', 'M5.goals.team') AND metric = 'log_loss')"
        ).fetchone()[0]  # fmt: skip
    assert abs(summary[0][2] - goals_skill) < 1e-12  # the keys' skills averaged, not their losses
    together = log.execute("SELECT probability_kind, metric, value, void_count FROM"
                           " track_record_scores WHERE period = 'all' AND competition_id = 0"
                           " AND model_version = 'all' ORDER BY probability_kind").fetchall()
    assert together == summary  # one version: every version together is that version
    assert log.execute("SELECT COUNT(*) FROM calibration_buckets").fetchone()[0] > 0
    log.close()

    again = run(world_path, log_path, "2018-06-02T06:00:00Z", say=lambda _: None)
    assert (again["outcomes"], again["line_checks"]) == (0, 0)  # nothing new
    world.execute("UPDATE matches SET home_goals = home_goals + 1 WHERE match_id = ?", (played,))
    world.commit()
    corrected = run(world_path, log_path, "2018-06-03T06:00:00Z", say=lambda _: None)
    before, after = settle_spine_outcomes(home, away), settle_spine_outcomes(home + 1, away)
    changed = sum(before[key] != after[key] for key in before)  # the score, goals, home's M5...
    assert corrected["outcomes"] == changed >= 3
    log = open_log(log_path)
    reasons = log.execute("SELECT COUNT(*) FROM outcomes WHERE correction_reason IS NOT NULL")
    assert reasons.fetchone()[0] == changed
    log.close()

    world.execute("UPDATE matches SET status = 'awarded', home_goals_on_pitch = home_goals,"
                  " away_goals_on_pitch = away_goals WHERE match_id = ?", (played,))
    world.commit()
    awarded = run(world_path, log_path, "2018-06-04T06:00:00Z", say=lambda _: None)
    assert (awarded["final"], awarded["void"]) == (0, 3)  # awarded after its check: void now
    log = open_log(log_path)
    assert log.execute("SELECT reason FROM voids WHERE match_id = ?", (played,)).fetchall() == [
        ("awarded",)]
    latest = log.execute("SELECT count, void_count FROM track_record_scores WHERE computed_at ="
                         " '2018-06-04T06:00:00Z' AND period = 'all' AND competition_id = 0"
                         " AND model_version = 'all' ORDER BY probability_kind").fetchall()
    assert latest == [(0, 15), (0, 3)]  # nothing counted is left, and the round says so
    log.close()


def test_every_version_together_pools_each_keys_losses_and_never_averages_the_versions(
        world, tmp_path: Path):  # fmt: skip
    """5 November 2026: Vesta is shown as one thing. model_version 'all' takes
    each key's own and reference losses over every version's forecasts, then the key's skill,
    then the kind's average over its keys; void, withdrawn and unmatched summed."""
    world_path, log_path = tmp_path / "football-world.db", tmp_path / "forecast-log.db"
    log = open_log(log_path)
    for item in DEFINITIONS.values():
        register_quantity(log, item.quantity)
    later = "vesta-2.1.0"
    for version in (VERSION, later):
        register_model_version(log, version, "spine", "2017-07-01T00:00:00Z", "the spine")
        for key in ("M1.result", "M2.score", "M3.goals", "M4.both_score", "M5.goals.team"):
            register_published_quantity(log, key, 1, version, "2017-07-01T00:00:00Z", True,
                                        False)  # fmt: skip
    played = [row[0] for row in world.execute(
        "SELECT m.match_id FROM matches m JOIN seasons s ON s.season_id = m.season_id"
        " WHERE s.season = 2017 AND s.competition_id = 1 AND m.status = 'played'"
        " ORDER BY m.match_id LIMIT 3")]  # fmt: skip
    other = score_matrices(np.array([0.8]), np.array([1.9]), Calibration())[0]
    _forecast(log, world, played[0])
    for match_id in played[1:]:
        _forecast(log, world, match_id, later, other)
    log.close()
    run(world_path, log_path, "2018-06-01T06:00:00Z", say=lambda _: None)

    log = open_log(log_path)
    where = " WHERE period = 'all' AND competition_id = 0 AND probability_kind = 'goals'"
    rows = log.execute("SELECT model_version, value, count FROM track_record_scores" + where
                       ).fetchall()  # fmt: skip
    summary = {version: (value, count) for version, value, count in rows}
    assert set(summary) == {VERSION, later, "all"}
    assert summary["all"][1] == summary[VERSION][1] + summary[later][1]
    keys = log.execute("SELECT quantity_key, metric, model_version, value, baseline_value, count"
                       " FROM track_record_key_scores" + where).fetchall()  # fmt: skip
    assert "all" not in {row[2] for row in keys}  # the key scores stay by version
    pooled = {}
    for key, metric, _, value, baseline, count in keys:
        own, naive, total = pooled.get((key, metric), (0.0, 0.0, 0))
        pooled[(key, metric)] = (own + value * count, naive + baseline * count, total + count)
    deciding = {"M2.score": "log_loss", "M3.goals": "ranked_probability_score",
                "M4.both_score": "log_loss", "M5.goals.team": "ranked_probability_score"}
    skills = [1.0 - own / naive for (key, metric), (own, naive, _) in pooled.items()
              if deciding.get(key) == metric]  # fmt: skip
    assert abs(summary["all"][0] - sum(skills) / len(skills)) < 1e-12
    averaged = (summary[VERSION][0] + summary[later][0]) / 2
    weighted = ((summary[VERSION][0] * summary[VERSION][1] + summary[later][0]
                 * summary[later][1]) / summary["all"][1])  # fmt: skip
    assert abs(summary["all"][0] - averaged) > 1e-6 and abs(summary["all"][0] - weighted) > 1e-6
    assert log.execute("SELECT COUNT(*) FROM calibration_buckets WHERE model_version = 'all'"
                       ).fetchone()[0] == 0  # fmt: skip
    log.close()


def test_a_key_the_job_does_not_know_or_never_published_is_skipped_not_a_stop():
    said = []
    records = [(1, "M1.result", 1), (1, "P3.anytime_scorer", 7), (1, "M3.goals", 1),
               (2, "P3.anytime_scorer", 8)]  # fmt: skip
    kept = keep_checkable(records, DEFINITIONS, {"M1.result"}, said.append)
    assert kept == [(1, "M1.result", 1)]
    assert said == ["skipped M3.goals: never in published_quantities",
                    "skipped P3.anytime_scorer: not in config/vesta/quantities.csv"]  # once each



def test_team_counts_happened_as_the_provider_of_record_counted_them(world):
    from vesta.jobs.check import read_team_count_outcomes

    match_id, home, away = world.execute(
        "SELECT match_id, home_team_id, away_team_id FROM matches WHERE status = 'played'"
        " ORDER BY match_id LIMIT 1").fetchone()  # fmt: skip
    for team, report, feed in ((home, (7, 2, 0), (8, 3, 0)), (away, (25, 3, 1), (24, 3, 1))):
        for layer, (corners, yellow, red) in (("report", report), ("feed", feed)):
            world.execute("INSERT INTO match_team_statistics (match_id, team_id, source, layer,"
                          " corners, yellow_cards, red_cards) VALUES (?, ?, ?, ?, ?, ?, ?)",
                          (match_id, team, layer, layer, corners, yellow, red))  # fmt: skip
    world.commit()
    happened = read_team_count_outcomes(world, {match_id})[match_id]
    assert happened[("M15.corners.team", "home")] == 7  # the report first, not the feed's 8
    assert happened[("M15.corners.team", "away")] == 20  # folded: the last cell is 20 or more
    assert happened[("M15.corners.sides", "match")] == (7, 20)
    assert happened[("M17.cards", "match")] == 6 and happened[("M17.cards.team", "away")] == 4


def test_a_position_forecast_of_a_player_who_did_not_start_is_void_and_the_job_goes_on(
        world, tmp_path: Path):  # fmt: skip
    from vesta.lineups.publication import list_outcomes

    world_path = tmp_path / "football-world.db"
    log = open_log(tmp_path / "forecast-log.db")
    for key in ("M1.result", "L9.position"):
        register_quantity(log, DEFINITIONS[key].quantity)
    register_model_version(log, VERSION, "spine", "2017-07-01T00:00:00Z", "the spine")
    for key in ("M1.result", "L9.position"):
        register_published_quantity(log, key, 1, VERSION, "2017-07-01T00:00:00Z", False, False)
    played = _played_match(world)
    home, away, kickoff = world.execute("SELECT home_team_id, away_team_id, kickoff_utc FROM"
                                        " matches WHERE match_id = ?", (played,)).fetchone()
    world.executemany("INSERT INTO players (player_id, name) VALUES (?, ?)",
                      [(7, "Seven"), (8, "Eight")])
    world.execute("INSERT INTO match_team_sheets (match_id, team_id, formation, starters_count,"
                  " bench_count, source) VALUES (?, ?, '4-4-2', 11, 7, 'feed')", (played, home))
    world.executemany("INSERT INTO match_players (match_id, team_id, player_id, started,"
                      " minutes_played, position_code, on_bench, source)"
                      " VALUES (?, ?, ?, ?, ?, ?, 0, 'feed')",
                      [(played, home, 7, 0, 25, None), (played, home, 8, 1, 90, "G")])
    world.commit()
    outcomes = list_outcomes("L9.position")
    flat = {outcome: 1 / len(outcomes) for outcome in outcomes}
    publish(log, Publication(stage="pre_match", trigger="first",
                             published_at=kickoff[:10] + "T08:00:00Z", code_commit="test",
                             match_id=played, scheduled_kickoff_utc=kickoff),
            [*build_spine_distributions(MATRIX, played, home, away, VERSION)[:1],
             *(PublishedDistribution("L9.position", player, VERSION, flat)
               for player in (7, 8))])  # fmt: skip
    record_forecasts_of_record(log, played, kickoff, kickoff)
    log.close()
    counts = run(world_path, tmp_path / "forecast-log.db", "2018-06-01T06:00:00Z",
                 say=lambda _: None)  # fmt: skip
    assert counts["final"] == 1
    log = open_log(tmp_path / "forecast-log.db")
    assert log.execute("SELECT subject_id, reason FROM voids WHERE quantity_key = 'L9.position'"
                       ).fetchall() == [(7, "did_not_play")]  # fmt: skip
    assert log.execute("SELECT subject_id, outcome_value FROM outcomes WHERE quantity_key ="
                       " 'L9.position'").fetchall() == [(8, '"G"')]  # fmt: skip
    log.close()


def test_the_scorers_of_a_player_who_played_and_the_void_of_one_who_did_not(world,
                                                                            tmp_path: Path):
    from vesta.players.publication import make_poisson_count

    world_path = tmp_path / "football-world.db"
    log = open_log(tmp_path / "forecast-log.db")
    keys = ("P3.scores", "P3.goals", "P3.first_scorer", "P5.assists", "P5.scores_or_assists")
    register_model_version(log, VERSION, "spine", "2017-07-01T00:00:00Z", "the spine")
    for key in ("M1.result", *keys):
        register_quantity(log, DEFINITIONS[key].quantity)
        register_published_quantity(log, key, 1, VERSION, "2017-07-01T00:00:00Z", False, False)
    played = _played_match(world)
    home, away, kickoff = world.execute("SELECT home_team_id, away_team_id, kickoff_utc FROM"
                                        " matches WHERE match_id = ?", (played,)).fetchone()
    world.executemany("INSERT INTO players (player_id, name) VALUES (?, ?)",
                      [(7, "Seven"), (8, "Eight"), (9, "Nine")])
    world.execute("INSERT INTO match_team_sheets (match_id, team_id, formation, starters_count,"
                  " bench_count, source) VALUES (?, ?, '4-4-2', 11, 7, 'feed')", (played, home))
    world.executemany("INSERT INTO match_players (match_id, team_id, player_id, started,"
                      " minutes_played, position_code, on_bench, source)"
                      " VALUES (?, ?, ?, ?, ?, ?, ?, 'feed')",
                      [(played, home, 7, 1, 90, "CF", 0), (played, home, 8, 1, 90, "CM", 0),
                       (played, home, 9, 0, 0, None, 1)])  # fmt: skip
    world.executemany("INSERT INTO match_events (match_id, sequence, period, minute, kind,"
                      " team_id, player_id, related_player_id, source)"
                      " VALUES (?, ?, 1, ?, ?, ?, ?, ?, 'feed')",
                      [(played, 1, 12, "goal", home, 7, 8),
                       (played, 2, 50, "penalty_goal", home, 7, None)])  # fmt: skip
    world.commit()
    yes = {"yes": 0.3}
    publish(log, Publication(stage="pre_match", trigger="first",
                             published_at=kickoff[:10] + "T08:00:00Z", code_commit="test",
                             match_id=played, scheduled_kickoff_utc=kickoff),
            [*build_spine_distributions(MATRIX, played, home, away, VERSION)[:1],
             *(PublishedDistribution(key, player, VERSION,
                                     make_poisson_count(0.4) if key == "P3.goals" else yes)
               for player in (7, 8, 9) for key in keys)])  # fmt: skip
    record_forecasts_of_record(log, played, kickoff, kickoff)
    log.close()
    run(world_path, tmp_path / "forecast-log.db", "2018-06-01T06:00:00Z", say=lambda _: None)
    log = open_log(tmp_path / "forecast-log.db")
    happened = {(key, subject): json.loads(value) for key, subject, value in log.execute(
        "SELECT quantity_key, subject_id, outcome_value FROM outcomes"
        " WHERE quantity_key LIKE 'P%'")}  # fmt: skip
    assert happened == {
        ("P3.scores", 7): True, ("P3.goals", 7): 2, ("P3.first_scorer", 7): True,
        ("P5.assists", 7): False, ("P5.scores_or_assists", 7): True,
        ("P3.scores", 8): False, ("P3.goals", 8): 0, ("P3.first_scorer", 8): False,
        ("P5.assists", 8): True, ("P5.scores_or_assists", 8): True}  # fmt: skip
    assert sorted(log.execute("SELECT quantity_key, subject_id, reason FROM voids WHERE"
                              " quantity_key LIKE 'P%'")) == [(key, 9, "did_not_play")
                                                              for key in sorted(keys)]  # fmt: skip
    log.close()


def test_the_likeliest_result_hits_ties_and_the_matches_left_out():
    from vesta.jobs.check import count_hits, find_likeliest_outcome

    assert find_likeliest_outcome([["home", 0.5], ["draw", 0.3], ["away", 0.2]]) == "home"
    assert (
        find_likeliest_outcome([["home", 0.382], ["draw", 0.381], ["away", 0.237]]) is None
    )  # 38 % / 38 %
    assert (
        find_likeliest_outcome([["home", 0.386], ["draw", 0.381], ["away", 0.233]]) == "home"
    )  # 39 / 38
    # on the figures shown together (since 4 November 2026), not each rounded alone:
    assert (
        find_likeliest_outcome([["home", 0.1772], ["draw", 0.40725], ["away", 0.41555]]) is None
    )  # 41/41
    assert (
        find_likeliest_outcome([["home", 0.46531], ["draw", 0.46515], ["away", 0.06954]]) == "home"
    )

    class Log:
        def execute(self, sql):
            if "withdrawals" in sql:
                return [(4,)]
            return [(3,)]  # match 3 is void

    def record(match, publication, values, version="vesta-2.4.0"):
        return (match, "M1.result", match, publication, version, "", json.dumps(values), "")

    clear, tie = [["home", 0.6], ["draw", 0.25], ["away", 0.15]], [
        ["home", 0.38], ["draw", 0.38], ["away", 0.24]]
    records = [record(1, 1, clear), record(2, 2, clear, "vesta-2.5.0"), record(3, 3, clear),
               record(4, 4, clear), record(5, 5, clear), record(6, 6, tie),
               ("x", "M3.goals", 1, 1, "vesta-2.4.0", "", "{}", "")]  # fmt: skip
    state = {match: {"competition_id": 1, "season_id": 70, "status": "played"}
             for match in range(1, 7)}  # fmt: skip
    happened = {(1, "M1.result", 1): "home", (2, "M1.result", 2): "away",
                (6, "M1.result", 6): "home", (3, "M1.result", 3): "home",
                (4, "M1.result", 4): "home"}  # 5 not yet final  # fmt: skip
    rows = {row.model_version: row for row in count_hits(Log(), records, state, happened)}
    every = rows["all"]
    assert (every.matches, every.hits, every.ties) == (2, 1, 1)  # 1 hit, 2 missed, 6 a tie
    assert (every.void_count, every.withdrawn_count, every.pending_count) == (1, 1, 1)
    assert (rows["vesta-2.5.0"].matches, rows["vesta-2.5.0"].hits) == (1, 0)
    assert (rows["vesta-2.4.0"].matches, rows["vesta-2.4.0"].hits) == (1, 1)
    assert every.competition_id == 1 and every.season_id == 70


def test_a_player_threshold_reads_the_feeds_count_and_is_void_or_pending_when_it_must():
    from vesta.jobs.check import settle_threshold_outcome
    from vesta.players.thresholds import COUNTS

    sheet = {"players": {7: (True, 90, "F"), 8: (False, 0, None), 9: (False, 20, None)}}
    counts = np.full(len(COUNTS), np.nan)
    counts[COUNTS.index("shots")], counts[COUNTS.index("booked")] = 3, 1
    assert settle_threshold_outcome("P1.shots", 7, sheet, (counts, np.nan)) == (True, 3)
    assert settle_threshold_outcome("P9.booked", 7, sheet, (counts, np.nan)) == (True, True)
    assert settle_threshold_outcome("P1.shots", 8, sheet, None) == (
        True,
        None,
    )  # did not play: void
    assert settle_threshold_outcome("P1.shots", 9, sheet, None) == (False, None)  # no row: pending
    assert settle_threshold_outcome("P7.key_passes", 7, sheet, (counts, np.nan)) == (False, None)
