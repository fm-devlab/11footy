"""The pre-match job on our world: the fixtures it forecasts, the gate of what is published, its
loop over a day's fixtures, and what a publication says changed since the last one."""

import sqlite3
from pathlib import Path

import numpy as np

from vesta.forecast_log.store import open_log
from vesta.forecast_log.writing import (
    Publication,
    publish,
    register_model_version,
    register_published_quantity,
    register_quantity,
)
from vesta.jobs import pre_match
from vesta.jobs.fixtures import FixtureForecast, read_fixtures
from vesta.jobs.pre_match import build_distributions, read_published_versions, read_trigger
from vesta.quantities import load_quantities
from vesta.spine.goals_in_time import GoalsInTimeParameters, TimeShares
from vesta.spine.publication import build_spine_distributions
from vesta.spine.score_matrix import Calibration, score_matrices

ROOT = Path(__file__).resolve().parents[3]
DEFINITIONS = load_quantities()
MATRIX = score_matrices(np.array([1.5]), np.array([1.1]), Calibration())[0]
SHARES = TimeShares(0.5, 0.5, (0.1, 0.03, 0.04, 0.15, 0.15, 0.53))
MATCH, HOME, AWAY = 900, 101, 102


def test_a_fixture_waits_for_both_teams_previous_match(world):
    world.execute(
        "INSERT INTO matches (match_id, season_id, kickoff_utc, kickoff_precision, local_date,"
        " home_team_id, away_team_id, status, source) VALUES"
        " (9001, 117, '2018-06-02T15:00:00Z', 'minute', '2018-06-02', 100, 101, 'scheduled', 't'),"
        " (9002, 117, '2018-06-01T15:00:00Z', 'minute', '2018-06-01', 102, 103, 'scheduled', 't'),"
        " (9003, 117, '2018-06-03T15:00:00Z', 'minute', '2018-06-03', 103, 104, 'scheduled', 't')"
    )  # fmt: skip
    fixtures = read_fixtures(world, "2018-05-30T00:00:00Z", 168, ["PL", "SA"], [])
    # 9003 waits: club 103 plays 9002 before it
    assert [(fixture.match_id, fixture.kind) for fixture in fixtures] == [(9002, "league"),
                                                                         (9001, "league")]


def test_the_gate_publishes_only_the_registered_keys_once_a_day(tmp_path):
    log = _log(tmp_path)
    assert read_published_versions(log) == {}  # before the freeze: nothing
    forecast = FixtureForecast(_fixture(), MATRIX, 1.5, 1.1, None)
    parameters = GoalsInTimeParameters.from_config(_spine())
    assert build_distributions(forecast, SHARES, parameters, read_published_versions(log), 1) == []
    _register(log, ("M1.result", "M8.goal_in_band.band=76-90"))
    chosen = build_distributions(forecast, SHARES, parameters, read_published_versions(log), 1)
    assert {item.quantity_key for item in chosen} == {"M1.result", "M8.goal_in_band.band=76-90"}
    assert {item.model_version for item in chosen} == {"2.0.0"}
    assert build_distributions(forecast, SHARES, parameters, read_published_versions(log), 2) == []
    assert read_trigger(log, MATCH, "2026-10-30") == "first"
    _pre_match(log, "2026-10-30T07:00:00Z")
    assert read_trigger(log, MATCH, "2026-10-30") is None
    assert read_trigger(log, MATCH, "2026-10-31") == "morning"














def _spine() -> dict:
    import tomllib

    return tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))


def _fixture():
    from vesta.jobs.fixtures import Fixture

    return Fixture(MATCH, "PL", "league", 2026, HOME, AWAY, "2026-10-31T15:00:00Z", False, "")


def _log(tmp_path: Path) -> sqlite3.Connection:
    log = open_log(tmp_path / "forecast-log.db")
    if not log.execute("SELECT 1 FROM quantities LIMIT 1").fetchone():
        for item in DEFINITIONS.values():
            register_quantity(log, item.quantity)
        register_model_version(log, "2.0.0", "spine", "2026-10-27T00:00:00Z", "the spine")
    return log


def _register(log, keys, moves_live=False) -> None:
    for key in keys:
        register_published_quantity(log, key, 1, "2.0.0", "2026-10-27T00:00:00Z", moves_live,
                                    False)  # fmt: skip


def _pre_match(log, published_at: str) -> None:
    publish(log, Publication(stage="pre_match", trigger="first", published_at=published_at,
                             code_commit="test", match_id=MATCH,
                             scheduled_kickoff_utc="2026-10-31T15:00:00Z"),
            build_spine_distributions(MATRIX, MATCH, HOME, AWAY, "2.0.0"))  # fmt: skip






def test_the_morning_loop_publishes_every_fixture_with_team_counts_and_resumes_once_a_day(
        tmp_path, monkeypatch):  # fmt: skip
    """The job's whole loop over two league fixtures with the team counts registered (the crash of
    29 October: the day's tally and the team counts shared a name); then a second run the same day,
    after a run stopped past its first publication, writes only the missing match."""
    from vesta.jobs.fixtures import Fixture
    from vesta.team_counts.model import count_distributions

    class TeamCounts:
        def build_fixture_distributions(self, competition, match_id, home_team_id, away_team_id):
            found = []
            for count, size in (("corners", 21), ("cards", 11)):
                found += count_distributions(count, match_id, home_team_id, away_team_id,
                                             (5.0, 4.0), 12.0, size, "")  # fmt: skip
            return found

    other = Fixture(MATCH + 1, "PL", "league", 2026, 103, 104, "2026-10-31T17:30:00Z", False, "")
    cup = Fixture(MATCH + 2, "CL", "cup", 2026, 105, 106, "2026-11-04T20:00:00Z", False, "")
    forecasts = [FixtureForecast(_fixture(), MATRIX, 1.5, 1.1, None),
                 FixtureForecast(other, MATRIX, 1.5, 1.1, None),
                 FixtureForecast(cup, MATRIX, 1.5, 1.1, None)]  # fmt: skip
    monkeypatch.setattr(pre_match, "read_fixtures", lambda *_: [f.fixture for f in forecasts])
    monkeypatch.setattr(pre_match, "read_current_season", lambda *_: 2026)
    monkeypatch.setattr(pre_match, "read_known_results", lambda *_: {})
    monkeypatch.setattr(pre_match, "hash_release", lambda *_: "test")
    monkeypatch.setattr(pre_match, "forecast_league_fixtures", lambda *_: (list(forecasts), {}))
    monkeypatch.setattr(pre_match, "forecast_cup_fixtures", lambda *_: [])
    monkeypatch.setattr(pre_match, "forecast_nation_fixtures", lambda *_: [])
    monkeypatch.setattr(pre_match, "fit_current_timing", lambda *_: (SHARES, (2.0, 5.0)))
    monkeypatch.setattr(pre_match, "forecast_team_counts", lambda *_: (TeamCounts(), None))
    world = tmp_path / "football-world.db"
    connection = sqlite3.connect(world)
    connection.execute("CREATE TABLE competitions (code TEXT, competition_id INTEGER)")
    connection.execute("INSERT INTO competitions VALUES ('PL', 1), ('CL', 6), ('BL', 4)")
    connection.commit()
    connection.close()
    log = _log(tmp_path)
    keys = ("M1.result", "M15.corners", "M15.corners.team", "M15.corners.sides", "M17.cards",
            "M17.cards.team", "M17.cards.sides")  # fmt: skip
    _register(log, keys)
    _pre_match(log, "2026-10-30T02:19:02Z")  # the run that stopped after its first publication
    log.close()
    cache = tmp_path / "cache"
    counts = pre_match.run(world, tmp_path / "forecast-log.db", cache, 168,
                           "2026-10-30T05:00:00Z", say=lambda *_: None)  # fmt: skip
    assert counts["fixtures"] == 3 and counts["published"] == 1
    assert counts["held"] == {"CL": 1} and counts["held_leagues"] == []  # no cup key yet
    log = open_log(tmp_path / "forecast-log.db")
    rows = log.execute("SELECT match_id, trigger FROM publications ORDER BY publication_id"
                       ).fetchall()  # fmt: skip
    assert rows == [(MATCH, "first"), (MATCH + 1, "first")]
    published = {row[0] for row in log.execute(
        "SELECT quantity_key FROM distributions d JOIN publications p USING (publication_id)"
        " WHERE p.match_id = ?", (MATCH + 1,))}  # fmt: skip
    assert published == set(keys)
    log.close()
    # a fresh day: both fixtures, each a `morning`
    counts = pre_match.run(world, tmp_path / "forecast-log.db", cache, 168,
                           "2026-10-31T05:00:00Z", say=lambda *_: None)  # fmt: skip
    assert counts["published"] == 2 and counts["would_publish"] == 1
    # a league match the gate holds (nothing registered for BL here) is named
    forecasts.append(FixtureForecast(Fixture(MATCH + 3, "BL", "league", 2026, 107, 108,
                                             "2026-11-01T14:30:00Z", False, ""),
                                     MATRIX, 1.5, 1.1, None))  # fmt: skip
    said = []
    counts = pre_match.run(world, tmp_path / "forecast-log.db", cache, 168,
                           "2026-10-31T06:00:00Z", say=said.append)  # fmt: skip
    assert counts["held_leagues"] == ["BL"]
    assert "(nothing registered for them: BL 1, CL 1)" in said[0]












def test_a_morning_says_what_changed_since_the_last_one_and_what_it_does_not_include(tmp_path):
    from vesta.forecast_log.writing import PublishedDistribution, SideInputs
    from vesta.jobs.pre_match import find_changes, make_caveats

    log = open_log(tmp_path / "forecast-log.db")
    quantities = {key: item.quantity for key, item in DEFINITIONS.items()}
    for key in ("M1.result", "L1.starts", "P3.scores", "P1.shots"):
        register_quantity(log, quantities[key])
    for version in ("vesta-2.5.0", "vesta-2.7.0"):
        register_model_version(log, version, "spine", "2026-10-01T00:00:00Z", "the spine")
    result = [["home", 0.4], ["draw", 0.28], ["away", 0.32]]
    morning = [PublishedDistribution("M1.result", 73613, "vesta-2.5.0", result)]
    previous = publish(log, Publication("pre_match", "morning", "2026-11-01T02:13:06Z", "f6db67e",
                                        match_id=73613,
                                        scheduled_kickoff_utc="2026-11-02T17:30:00Z"),
                       morning, inputs=[SideInputs(601, "2026-10-26", (1, 2, 3)),
                                        SideInputs(307, "2026-10-26", (7, 8, 9))])  # fmt: skip
    chosen = [PublishedDistribution("M1.result", 73613, "vesta-2.7.0", result),
              PublishedDistribution("L1.starts", 10790, "vesta-2.7.0", {"yes": 0.2}),
              PublishedDistribution("P3.scores", 10790, "vesta-2.7.0", {"yes": 0.1})]
    inputs = [SideInputs(601, "2026-11-02", (1, 2, 4)), SideInputs(307, "2026-11-02", (7, 8, 9))]
    known = {601: [("2026-10-25T18:00:00Z", 67514), ("2026-10-27T21:00:00Z", 73501),
                   ("2026-11-02T19:30:00Z", 73613)],
             307: [("2026-10-29T23:00:00Z", 73520)]}  # fmt: skip
    found = {(item.code, item.team_id): item.detail
             for item in find_changes(log, 73613, chosen, inputs, known)}
    assert found == {
        ("release_changed", 0): {"from": "vesta-2.5.0", "to": "vesta-2.7.0"},
        # the Monday refit takes in the results known since the last one, nothing after
        ("weekly_fit", 601): {"fit_week": "2026-11-02", "results_added": [73501]},
        ("weekly_fit", 307): {"fit_week": "2026-11-02", "results_added": [73520]},
        ("eleven_changed", 601): {"out": [3], "in": [4]},  # Fiorentina's eleven is the same
    }
    assert all(item.previous_publication_id == previous
               for item in find_changes(log, 73613, chosen, inputs, known))
    assert find_changes(log, 73704, chosen, inputs, known) == []  # a first publication
    kinds = {key: item.probability_kind for key, item in quantities.items()}
    assert sorted((item.probability_kind, item.code) for item in make_caveats(chosen, kinds)) == [
        ("lineups", "injuries_not_included"), ("lineups", "suspensions_not_included"),
        ("scorers", "injuries_not_included"), ("scorers", "suspensions_not_included")]
    # in a league whose lineups read the bans: the bans come from the cards, injuries are missing
    assert sorted((item.probability_kind, item.code)
                  for item in make_caveats(chosen, kinds, bans_read=True)) == [
        ("lineups", "injuries_not_included"), ("lineups", "suspensions_from_cards"),
        ("scorers", "injuries_not_included"), ("scorers", "suspensions_from_cards")]
    # the scorers rated on the shots of earlier matches (the source out): said on their kind only
    assert sorted((item.probability_kind, item.code)
                  for item in make_caveats(chosen, kinds, shots_stale=True)) == [
        ("lineups", "injuries_not_included"), ("lineups", "suspensions_not_included"),
        ("scorers", "injuries_not_included"), ("scorers", "shots_stale"),
        ("scorers", "suspensions_not_included")]
    log.close()




