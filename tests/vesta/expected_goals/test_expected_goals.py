"""Our expected goals: the shots' context, the inputs, the fit, the walk-forward, on a small
world built here with the world's own schema."""

import sqlite3
import tomllib
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from tests.vesta.world_schema import create_world_schema

from vesta.expected_goals.model import FEATURES, ModelSettings, build_inputs, fit
from vesta.expected_goals.shots import read_shots
from vesta.expected_goals.walk_forward import (
    WalkForwardSettings,
    assign_fold,
    check,
    rate_every_season,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIGURATION = tomllib.loads(
    (ROOT / "config/vesta/expected_goals.toml").read_text(encoding="utf-8")
)
# small enough to fit in a second, the same rules otherwise
QUICK = ModelSettings(**{**CONFIGURATION["model"], "most_trees": 60, "early_stopping_trees": 10,
                         "trees_without_validation": 30, "min_child_samples": 20})  # fmt: skip
SITUATIONS = ("open_play", "corner", "set_piece", "direct_free_kick", "penalty")


def small_world(path: Path, matches_per_season: int = 40) -> sqlite3.Connection:
    """One league, seasons 2014 to 2017, shots whose chance falls with distance; each goal also
    an event, so the score of the moment can be read."""
    connection = sqlite3.connect(path)
    create_world_schema(connection)
    connection.execute(
        "INSERT INTO competitions VALUES (1, 'PL', 'PL', 'T', 'league', 'UTC', 2014, 1)"
    )
    connection.executemany("INSERT INTO teams (team_id, name, kind) VALUES (?, ?, 'club')",
                           [(1, "home club"), (2, "away club")])  # fmt: skip
    state = np.random.RandomState(20261024)
    match_id = 0
    for season in range(2014, 2018):
        connection.execute(
            "INSERT INTO seasons VALUES (?, 1, ?, ?, NULL, NULL, 2, 'closed', 'league')",
            (season, season, str(season)),
        )
        for _ in range(matches_per_season):
            match_id += 1
            connection.execute(
                "INSERT INTO matches (match_id, season_id, kickoff_utc, kickoff_precision,"
                " local_date, home_team_id, away_team_id, status, home_goals, away_goals, source)"
                " VALUES (?, ?, '2016-01-01T15:00:00Z', 'minute', '2016-01-01', 1, 2, 'played',"
                " 0, 0, 'test')",
                (match_id, season),
            )
            for minute in sorted(state.randint(1, 95, 24)):
                x = state.uniform(0.7, 0.99)
                y = state.uniform(0.3, 0.7)
                situation = SITUATIONS[state.randint(0, 5)]
                chance = 0.76 if situation == "penalty" else 0.6 * (x - 0.7) / 0.3
                result = "goal" if state.uniform() < chance else "saved"
                team = 1 + state.randint(0, 2)
                cursor = connection.execute(
                    "INSERT INTO shots (match_id, team_id, period, minute, x, y, result,"
                    " situation, body_part, last_action, is_penalty, source, layer,"
                    " also_in_commentary) VALUES (?, ?, 1, ?, ?, ?, ?, ?, 'right_foot', 'Pass', ?,"
                    " 'tracked', 'tracked', 0)",
                    (match_id, team, int(minute), x, y, result, situation,
                     int(situation == "penalty")),  # fmt: skip
                )
                if result == "goal":
                    connection.execute(
                        "INSERT INTO match_events (match_id, sequence, period, minute, kind,"
                        " team_id, shot_id, source) VALUES (?, ?, 1, ?, 'goal', ?, ?, 'test')",
                        (match_id, cursor.lastrowid, int(minute), team, cursor.lastrowid),
                    )
    connection.commit()
    return connection


@pytest.fixture
def world(tmp_path: Path) -> sqlite3.Connection:
    connection = small_world(tmp_path / "football-world.db")
    yield connection
    connection.close()


def test_the_shots_carry_the_score_of_the_moment_and_the_move(world):
    shots = read_shots(world, ["PL"], 2014, 2017, ["tracked"])
    assert len(shots) == 4 * 40 * 24
    first_match = shots.match_id == 1
    goals_before = 0
    for index in np.flatnonzero(first_match):
        assert abs(shots.state[index]) <= goals_before or goals_before == 0
        goals_before += shots.result[index] == "goal"
    first_shot_of_each_side = shots.gap == 10.0
    assert first_shot_of_each_side.sum() >= 2 * 4 * 40 - 4  # a side may shoot only once or never
    assert set(shots.same_minute.tolist()) >= {1}


def test_the_inputs_put_the_categories_in_the_reference_order():
    from vesta.expected_goals.shots import ShotTable

    one = ShotTable(*(np.array([value]) for value in (
        1, 2026, "PL", 0.9, 0.5, 30.0, "corner", "head", "Unknown action", "goal", 1, 0, 1, 10.0,
        0, 1)))  # fmt: skip
    row = dict(zip(FEATURES, build_inputs(one)[0], strict=True))
    assert row["situation"] == 1 and row["body"] == 2 and row["last_action"] == 18
    assert row["distance"] == pytest.approx(0.1 * 105)
    assert row["angle"] == pytest.approx(2 * np.arctan(3.66 / 10.5))


def test_a_penalty_gets_the_share_of_penalties_scored(world):
    shots = read_shots(world, ["PL"], 2014, 2015, ["tracked"])
    model = fit(shots, None, QUICK)
    penalties = shots.situation == "penalty"
    assert model.trees == 30
    assert model.penalty_rate == pytest.approx(shots.goal[penalties].mean())
    assert (model.rate(shots)[penalties] == model.penalty_rate).all()


def test_every_shot_is_rated_by_a_model_that_never_saw_its_match(world):
    shots = read_shots(world, ["PL"], 2014, 2017, ["tracked"])
    settings = WalkForwardSettings.from_config(CONFIGURATION)
    chances, models = rate_every_season(shots, QUICK, settings)
    assert np.isfinite(chances).all() and ((chances > 0) & (chances < 1)).all()
    assert set(models) == {"fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "season 2016",
                           "season 2017"}  # fmt: skip
    # the model of 2016 learned from 2014 and 2015 only (2015 chose its trees, then joined)
    expected = int(((shots.season <= 2015) & (shots.situation != "penalty")).sum())
    assert models["season 2016"].shots == expected
    again, _ = rate_every_season(shots, QUICK, settings)
    assert np.array_equal(chances, again)  # reproduced to the bit
    result = check(chances, shots.goal)
    assert result.expected_goals == pytest.approx(result.goals, rel=0.25)


def test_the_fold_of_a_match_is_stable():
    assert [assign_fold(match, 5) for match in (1, 2, 3)] == [
        assign_fold(match, 5) for match in (1, 2, 3)
    ]
    assert {assign_fold(match, 5) for match in range(100)} == {0, 1, 2, 3, 4}


def _permute_ids(connection: sqlite3.Connection, seed: int) -> None:
    """Every shot and event given another id, as keyed ids would: the links kept, the facts
    unchanged."""
    state = np.random.RandomState(seed)
    for table, column in (("shots", "shot_id"), ("match_events", "event_id")):
        ids = [row[0] for row in connection.execute(f"SELECT {column} FROM {table}")]
        offset = 10 * (max(ids) + 1)  # new ids clear of the old, so no update collides
        mapping = dict(zip(ids, (offset + int(new) for new in state.permutation(ids)),
                           strict=True))  # fmt: skip
        connection.executemany(f"UPDATE {table} SET {column} = ? WHERE {column} = ?",
                               [(new, old) for old, new in mapping.items()])
        if table == "shots":
            connection.executemany("UPDATE match_events SET shot_id = ? WHERE shot_id = ?",
                                   [(new, old) for old, new in mapping.items()])
    connection.commit()


def test_the_shots_and_their_chances_do_not_hang_on_the_ids(tmp_path):
    """Keyed ids in the data (9 November 2026): the same facts under other
    shot and event ids read as the same shots, the same chances, the same scorers."""
    from vesta.players.model import read_happened, read_shot_players

    settings = WalkForwardSettings.from_config(CONFIGURATION)
    results = []
    for name, seed in (("first.db", None), ("permuted.db", 7)):
        connection = small_world(tmp_path / name, matches_per_season=12)
        if seed is not None:
            _permute_ids(connection, seed)
        shots = read_shots(connection, ["PL"], 2014, 2017, ["tracked"])
        chances, _ = rate_every_season(shots, QUICK, settings)
        players = read_shot_players(connection, ["PL"], 2014, 2017, ["tracked"])
        happened = read_happened(connection, set(range(1, 49)))
        results.append((shots, chances, players, happened))
        connection.close()
    (shots, chances, players, happened), (other, again, other_players, other_happened) = results
    for field in fields(shots):
        assert np.array_equal(getattr(shots, field.name), getattr(other, field.name)), field.name
    assert np.array_equal(chances, again)
    assert players == other_players and happened == other_happened
