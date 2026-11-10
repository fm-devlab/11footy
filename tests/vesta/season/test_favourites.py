"""The favourites of a national tournament (T6.round_reached): the nations model's strength for
any pair, and the season job on a small world with the Asian Cup 2027's format."""

import sqlite3
import tomllib
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from tests.vesta.world_schema import create_world_schema

from vesta.quantities import load_quantities
from vesta.season.favourites import forecast_round_reached, read_host_id, read_tournament
from vesta.season.strength import build_nations_strength
from vesta.spine.nations import NationsParameters
from vesta.spine.ties import TieParameters
from vesta.spine.world import read_nation_matches

SPINE = tomllib.loads((Path(__file__).resolve().parents[3] / "config/vesta/spine.toml")
                      .read_text(encoding="utf-8"))  # fmt: skip
PARAMETERS = NationsParameters.from_config(SPINE)
TEAMS = 24


def _world(path: Path, knockout_played: bool = False) -> sqlite3.Connection:
    """24 nations of fixed strengths with eight years of friendlies, and the Asian Cup 2027 after
    its first matchday."""
    world = sqlite3.connect(path)
    create_world_schema(world)
    world.execute("INSERT INTO competitions VALUES (1, 'FRIENDLY', 'Friendlies', 'World',"
                  " 'friendly', 'UTC', 2019, 1)")  # fmt: skip
    world.execute("INSERT INTO competitions VALUES (2, 'ASIAN', 'AFC Asian Cup', 'Asia',"
                  " 'nations tournament', 'UTC', 2027, 1)")  # fmt: skip
    for team in range(1, TEAMS + 1):
        world.execute("INSERT INTO teams (team_id, name, kind) VALUES (?, ?, 'nation')",
                      (team, "Saudi Arabia" if team == 1 else f"Nation {team}"))  # fmt: skip
    state = np.random.RandomState(2027)
    strength = np.linspace(0.6, -0.6, TEAMS)
    for year in range(2019, 2027):
        world.execute("INSERT INTO seasons VALUES (?, 1, ?, ?, NULL, NULL, 0, 'closed', 'cup')",
                      (year, year, str(year)))  # fmt: skip
        for week in range(40):
            day = date(year, 1, 7) + timedelta(weeks=week)
            for home, away in state.permutation(TEAMS).reshape(-1, 2)[:4] + 1:
                _insert(world, year, day.isoformat(), int(home), int(away),
                        int(state.poisson(np.exp(0.1 + strength[home - 1] - strength[away - 1]))),
                        int(state.poisson(np.exp(strength[away - 1] - strength[home - 1]))),
                        "played", None)  # fmt: skip
    world.execute("INSERT INTO seasons VALUES (2027, 2, 2027, '2027', NULL, NULL, 24, 'running',"
                  " 'groups')")  # fmt: skip
    for index, letter in enumerate("ABCDEF"):
        members = [index * 4 + place for place in (1, 2, 3, 4)]
        pairs = [(members[0], members[1]), (members[2], members[3]), (members[0], members[2]),
                 (members[3], members[1]), (members[3], members[0]), (members[1], members[2])]
        for round_index, (home, away) in enumerate(pairs):
            day = (date(2027, 1, 10) + timedelta(days=4 * (round_index // 2))).isoformat()
            if round_index < 2:
                _insert(world, 2027, day, home, away, 1, 0, "played", f"group {letter.lower()}")
            else:
                _insert(world, 2027, day, home, away, None, None, "scheduled",
                        f"group {letter.lower()}")  # fmt: skip
    if knockout_played:
        _insert(world, 2027, "2027-01-30", 1, 5, 2, 1, "played", "round of 16")
    world.commit()
    return world


def _insert(world, season_id, day, home, away, home_goals, away_goals, status, stage):
    world.execute(
        "INSERT INTO matches (season_id, kickoff_utc, kickoff_precision, local_date, home_team_id,"
        " away_team_id, status, home_goals, away_goals, stage, neutral_venue, source) VALUES"
        " (?, ?, 'minute', ?, ?, ?, ?, ?, ?, ?, ?, 'test')",
        (season_id, f"{day}T15:00:00Z", day, home, away, status, home_goals, away_goals, stage,
         int(stage is not None)),
    )  # fmt: skip


def test_the_strength_of_any_pair_follows_the_ratings_and_the_host():
    world = _world(Path(":memory:"))
    matches = read_nation_matches(world)
    strength = build_nations_strength(matches, PARAMETERS, "2027-01-05")
    _, strong, weak = strength(2, 23, True)
    assert strong > weak  # nation 2 rated far above nation 23
    hosted = build_nations_strength(matches, PARAMETERS, "2027-01-05", frozenset({1}))
    assert hosted(1, 2, True)[1] > strength(1, 2, True)[1]  # the host at home on neutral ground
    assert read_host_id(world, "Saudi Arabia") == 1
    with pytest.raises(ValueError, match="national teams"):
        read_host_id(world, "Atlantis")


def test_the_job_gives_every_team_its_round_and_one_winner():
    world = _world(Path(":memory:"))
    strength = build_nations_strength(read_nation_matches(world), PARAMETERS, "2027-01-15")
    state = read_tournament(world, "ASIAN", 2027, strength)
    assert sorted(state.groups) == list("ABCDEF")
    assert all(len(group.played) == 2 and len(group.fixtures) == 4
               for group in state.groups.values())  # fmt: skip
    published = forecast_round_reached(state, "ASIAN", 2027, strength,
                                       TieParameters.from_config(SPINE), 400, 7, "test",
                                       load_quantities())  # fmt: skip
    assert len(published) == TEAMS
    winners = {item.subject_id: dict(item.values)["winner"] for item in published}
    assert sum(winners.values()) == pytest.approx(1.0)
    assert max(winners, key=winners.get) <= 4  # a strong side of group A is the favourite
    for item in published:
        assert sum(share for _, share in item.values) == pytest.approx(1.0)


def test_a_knockout_drawn_before_the_groups_end_is_refused():
    world = _world(Path(":memory:"), knockout_played=True)
    strength = build_nations_strength(read_nation_matches(world), PARAMETERS, "2027-01-31")
    state = read_tournament(world, "ASIAN", 2027, strength)
    with pytest.raises(ValueError, match="are not over"):
        forecast_round_reached(state, "ASIAN", 2027, strength, TieParameters.from_config(SPINE),
                               10, 7, "test", load_quantities())  # fmt: skip


# the round of 16 the rules give when the thirds of C, D, E and F go on (art. 9.8 and 9.9)
LAST_16 = (("2A", "2C"), ("1D", "3E"), ("1B", "3D"), ("1F", "2E"), ("1C", "3F"), ("1E", "2D"),
           ("1A", "3C"), ("2B", "2F"))  # fmt: skip


def _team(code: str) -> int:
    return "ABCDEF".index(code[1]) * 4 + int(code[0])


def _knockout_world(pairs=LAST_16) -> sqlite3.Connection:
    """The groups over (each group's first beats all, its second the two others, its third the
    last by as many goals as the group's number, so the thirds of C to F are the best), the round
    of 16 drawn, its first tie played."""
    world = _world(Path(":memory:"))
    world.execute("DELETE FROM matches WHERE season_id = 2027")
    for index, letter in enumerate("ABCDEF"):
        members = [index * 4 + place for place in (1, 2, 3, 4)]
        for high, low in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            margin = index + 1 if (high, low) == (2, 3) else 1
            _insert(world, 2027, "2027-01-12", members[high], members[low], margin, 0, "played",
                    f"group {letter.lower()}")  # fmt: skip
    for number, (home, away) in enumerate(pairs):
        played = number == 0
        _insert(world, 2027, "2027-01-28", _team(home), _team(away), 2 if played else None,
                1 if played else None, "played" if played else "scheduled",
                "round of 16")  # fmt: skip
    world.commit()
    return world


def test_the_knockout_is_read_from_the_world_and_what_it_settles_is_decided():
    world = _knockout_world()
    strength = build_nations_strength(read_nation_matches(world), PARAMETERS, "2027-01-29")
    state = read_tournament(world, "ASIAN", 2027, strength)
    published = {item.subject_id: item for item in forecast_round_reached(
        state, "ASIAN", 2027, strength, TieParameters.from_config(SPINE), 300, 7, "test",
        load_quantities())}  # fmt: skip
    every = set(load_quantities()["T6.round_reached"].quantity.lines)
    winner_2a, loser_2c, fourth_a = _team("2A"), _team("2C"), _team("4A")
    assert dict(published[loser_2c].values)["round_of_16"] == 1.0
    assert set(published[loser_2c].decided_lines) == every
    assert set(published[fourth_a].decided_lines) == every
    assert dict(published[fourth_a].values)["group_stage"] == 1.0
    reached = published[winner_2a]
    assert dict(reached.values)["round_of_16"] == 0.0
    assert "any=quarter_finals,semi_finals,final,winner" in reached.decided_lines
    assert "outcome=winner" not in reached.decided_lines
    assert sum(dict(item.values)["winner"] for item in published.values()) == pytest.approx(1.0)


def test_a_pairing_the_rules_do_not_give_is_refused():
    wrong = (("2A", "2B"), *LAST_16[1:7], ("2C", "2F"))
    world = _knockout_world(wrong)
    strength = build_nations_strength(read_nation_matches(world), PARAMETERS, "2027-01-29")
    state = read_tournament(world, "ASIAN", 2027, strength)
    with pytest.raises(ValueError, match="the world does not"):
        forecast_round_reached(state, "ASIAN", 2027, strength, TieParameters.from_config(SPINE),
                               10, 7, "test", load_quantities())  # fmt: skip


def test_during_the_groups_what_the_points_settle_is_decided():
    world = _knockout_world()
    world.execute("DELETE FROM matches WHERE stage = 'round of 16'")
    world.commit()
    strength = build_nations_strength(read_nation_matches(world), PARAMETERS, "2027-01-20")
    state = read_tournament(world, "ASIAN", 2027, strength)
    assert not state.knockout_started
    published = {item.subject_id: item for item in forecast_round_reached(
        state, "ASIAN", 2027, strength, TieParameters.from_config(SPINE), 200, 7, "test",
        load_quantities())}  # fmt: skip
    every = set(load_quantities()["T6.round_reached"].quantity.lines)
    fourth, winner, third = _team("4A"), _team("1A"), _team("3A")
    assert set(published[fourth].decided_lines) == every  # out: the group stage
    assert dict(published[fourth].values)["group_stage"] == 1.0
    assert "any=round_of_16,quarter_finals,semi_finals,final,winner" in (
        published[winner].decided_lines)  # surely in the round of 16
    # thirds level on points with the others: goal difference decides, not settled by points
    assert published[third].decided_lines == ()
