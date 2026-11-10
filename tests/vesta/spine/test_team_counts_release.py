"""Team counts for the release: the outcome rule shared by the verdicts and the check job, the
season's corrections and their cache, and the fixtures' distributions in the leagues and the cups
(family A's plan)."""

import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

from vesta.team_counts import cups, publication
from vesta.team_counts.model import Calibration, FitParameters
from vesta.team_counts.outcomes import are_offsides_unrecorded, read_match_counts
from vesta.team_counts.publication import (
    CupCountsOfTheDay,
    SeasonCorrections,
    fit_counts_of_the_day,
    fit_season_corrections,
    load_or_build_corrections,
)

ROOT = Path(__file__).resolve().parents[3]


def _spine() -> dict:
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    spine["walk_forward"]["competitions"] = ["PL", "SA"]
    return spine


def _played(world) -> list[tuple[int, int, int, int]]:
    return world.execute(
        "SELECT m.match_id, m.season_id, m.home_team_id, m.away_team_id FROM matches m"
        " WHERE m.status = 'played' ORDER BY m.kickoff_utc, m.match_id").fetchall()  # fmt: skip


def _insert(world, match_id: int, team: int, layer: str, **values) -> None:
    """A row of `layer`; the world's `source` given the same word, as its key needs one."""
    names = ", ".join(values)
    world.execute(f"INSERT INTO match_team_statistics (match_id, team_id, source, layer, {names})"
                  f" VALUES (?, ?, ?, ?, {', '.join('?' for _ in values)})",
                  (match_id, team, layer, layer, *values.values()))  # fmt: skip




def test_offsides_of_a_season_that_did_not_record_them_are_pending(world):
    played = [row for row in _played(world) if row[1] == played_season(world)]
    for index, (match_id, _, home, away) in enumerate(played[:10]):
        for team in (home, away):  # seven of ten matches without an offside: not recorded
            _insert(world, match_id, team, "feed", fouls=10, corners=4,
                    offsides=0 if index < 7 else 2)  # fmt: skip
    world.commit()
    ids = {row[0] for row in played[:10]}
    assert read_match_counts(world, ids, "offsides", "leagues") == {}
    assert read_match_counts(world, ids, "fouls", "leagues") == {
        row[0]: (10, 10, "feed") for row in played[:10]}  # fmt: skip
    assert are_offsides_unrecorded(3, 3) is None  # a season's first days: cannot tell yet
    assert are_offsides_unrecorded(1, 10) is False and are_offsides_unrecorded(9, 10) is True


def played_season(world) -> int:
    return world.execute("SELECT season_id FROM matches WHERE status = 'played'"
                         " ORDER BY kickoff_utc LIMIT 1").fetchone()[0]  # fmt: skip


def _counted_world(world, monkeypatch) -> None:
    """Every played match's corners and shots, drawn from fixed rates, from the report; the
    fits' minimum lowered to the small world's ninety matches a season."""
    config = publication._load_config()
    config["fit"]["minimum_matches"] = 60
    monkeypatch.setattr(publication, "_load_config", lambda: config)
    state = np.random.RandomState(20261031)
    for match_id, _, home, away in _played(world):
        for team, rate in ((home, 1.1), (away, 0.95)):
            _insert(world, match_id, team, "report",
                    corners=int(state.poisson(5.0 * rate)),
                    shots=int(state.negative_binomial(20, 20 / (20 + 12.0 * rate))))  # fmt: skip
    world.commit()


def test_the_season_corrections_are_the_morning_refit_of_any_morning_of_the_season(
        world, monkeypatch):
    """2.2.0 refitted its corrections every morning on the two seasons before; built once at the
    season's start they are the same on every later morning (the condition of 31 October)."""
    _counted_world(world, monkeypatch)
    spine = _spine()
    built = fit_season_corrections(world, "leagues", 2017, "2017-08-01T00:00:00Z", spine,
                                   ("corners",))  # fmt: skip
    for now in ("2017-09-20T08:00:00Z", "2018-02-14T08:00:00Z"):
        morning = fit_counts_of_the_day(world, ["PL", "SA"], 2017, now, spine, ("corners",))
        refit, kept = morning.calibrations["corners"], built.calibrations["corners"]
        assert ((*refit.home, *refit.away, refit.tilt, *refit.total)
                == pytest.approx((*kept.home, *kept.away, kept.tilt, *kept.total)))
        same = fit_counts_of_the_day(world, ["PL", "SA"], 2017, now, spine, ("corners",), built)
        fixture = (9999, *world.execute("SELECT home_team_id, away_team_id FROM matches"
                                        " WHERE season_id = 117 LIMIT 1").fetchone())  # fmt: skip
        assert ([item.values for item in same.build_fixture_distributions("PL", *fixture)]
                == [item.values for item in morning.build_fixture_distributions("PL", *fixture)])


def test_candidate_two_sizes_replace_the_weekly_fit_and_survive_the_cache(world, tmp_path,
                                                                         monkeypatch):
    _counted_world(world, monkeypatch)
    spine = _spine()
    corrections = load_or_build_corrections(
        world,
        "leagues",
        2017,
        "2017-08-01T00:00:00Z",
        spine,
        ("corners", "shots"),
        tmp_path,
        "release-a",
    )
    assert set(corrections.sizes) == {"shots"} and set(corrections.sizes["shots"]) == {"PL", "SA"}
    assert 8 < corrections.sizes["shots"]["PL"] < 60  # drawn with a size of 20
    kept = load_or_build_corrections(
        world, "leagues", 2017, "", spine, ("corners", "shots"), tmp_path, "release-a"
    )
    assert kept == corrections
    infinite = SeasonCorrections("leagues", 2017, corrections.calibrations,
                                 {"shots": {"PL": math.inf}})  # fmt: skip
    assert infinite.look_up_size("shots", "PL", 30.0) == math.inf
    assert (
        infinite.look_up_size("shots", "SA", 30.0) == 30.0
        and infinite.look_up_size("corners", "PL", 7.0) == 7.0
    )
    day = fit_counts_of_the_day(world, ["PL", "SA"], 2017, "2017-10-04T08:00:00Z", spine,
                            ("corners", "shots"), corrections)  # fmt: skip
    keys = {item.quantity_key for item in day.build_fixture_distributions("PL", 9999, 101, 102)}
    assert keys == {"M15.corners", "M15.corners.team", "M15.corners.sides", "M20.shots",
                    "M20.shots.team", "M20.shots.sides"}


def test_a_cup_fixture_reads_its_group_its_venue_and_the_elo_pair_it_is_given():
    fit = cups.PooledFit(teams={1: 0, 2: 1}, levels={"CL": math.log(5.0), "Q": math.log(4.0)},
                         home={True: 0.1, False: 0.2}, elo=0.5, make=np.zeros(2),
                         allow=np.zeros(2), size_nb=50.0)  # fmt: skip
    parameters = FitParameters(730, 180.0, 1.0, 100, 21)
    corrections = SeasonCorrections("cups", 2026, {"corners": Calibration()},
                                    {"corners": {"cups": 9.0}})  # fmt: skip
    day = CupCountsOfTheDay({"corners": parameters}, {"corners": fit}, corrections)

    def total_mean(competition, neutral, home_elo, away_elo) -> float:
        found = day.build_fixture_distributions(
            7, competition, 1, 2, neutral=neutral, home_elo=home_elo, away_elo=away_elo
        )
        total = next(item for item in found if item.quantity_key == "M15.corners")
        return float(np.dot(np.arange(len(total.values["probabilities"])),
                            total.values["probabilities"]))  # fmt: skip

    assert total_mean("CL", True, 1500, 1500) == pytest.approx(10.0, abs=1e-3)
    assert total_mean("CLQ", True, 1500, 1500) == pytest.approx(8.0, abs=1e-3)
    home_venue = total_mean("CL", False, 1500, 1500)
    assert home_venue == pytest.approx(5.0 * math.exp(0.1) + 5.0, abs=1e-3)
    gap = 0.5 * 200 / 400
    assert total_mean("CL", True, 1700, 1500) == pytest.approx(
        5.0 * math.exp(gap) + 5.0 * math.exp(-gap), abs=5e-3)  # the tail folded at 20


def test_the_layer_of_record_counts_and_feed_zeros_that_mean_unknown_stay_pending(world):
    played = _played(world)
    (first, _, h1, a1), (stub, _, h2, a2), (missing, _, h3, _) = played[:3]
    for team, shots in ((h1, 14), (a1, 9)):
        _insert(world, first, team, "report", shots=shots, corners=5, fouls=11)
        _insert(world, first, team, "feed", shots=shots + 1, corners=5, fouls=12, offsides=2)
    for team in (h2, a2):  # a feed stub: about three shots, the cards and nothing else
        _insert(world, stub, team, "feed", shots=3, corners=0, fouls=0, offsides=0,
                yellow_cards=2, red_cards=0)  # fmt: skip
    _insert(world, missing, h3, "feed", shots=10, corners=4, fouls=9, offsides=1)  # one side only
    world.commit()
    ids = {first, stub, missing}
    assert read_match_counts(world, ids, "shots", "leagues") == {first: (14, 9, "report")}
    assert read_match_counts(world, ids, "shots", "cups") == {first: (15, 10, "feed")}
    assert read_match_counts(world, ids, "cards", "leagues") == {stub: (2, 2, "feed")}
