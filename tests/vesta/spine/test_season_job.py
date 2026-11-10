"""The season job's reading of a league season as it stands: the table from the official
results, the play-offs out of it, the fixtures still to play, the matches that moved it."""

from pathlib import Path

from vesta.season.league_job import read_finished_since, read_league_season


def test_a_season_as_it_stands(world, tmp_path: Path):
    season_id = 117  # PL 2017 of the small world
    world.execute("UPDATE matches SET status = 'scheduled', home_goals = NULL, away_goals = NULL"
                  " WHERE season_id = ? AND kickoff_utc >= '2017-11-01'", (season_id,))  # fmt: skip
    world.execute("INSERT INTO matches (season_id, stage, kickoff_utc, kickoff_precision,"
                  " local_date, home_team_id, away_team_id, status, home_goals, away_goals,"
                  " source) VALUES (?, 'play-off', '2018-06-30T15:00:00Z', 'minute',"
                  " '2018-06-30', 100, 101, 'played', 1, 0, 't')", (season_id,))  # fmt: skip
    # the world lists the fixtures a few weeks ahead only: the later ones are not there yet
    world.execute("DELETE FROM matches WHERE season_id = ? AND kickoff_utc >= '2017-11-20'",
                  (season_id,))  # fmt: skip
    world.commit()
    competition_id, found_season, state, fixtures, finished = read_league_season(world, "PL",
                                                                                 2017)
    assert (competition_id, found_season) == (1, season_id)
    assert len(state.team_ids) == 10
    assert len(state.played) + len(fixtures) == 90  # the play-off is out of the table
    assert len(state.fixtures) == len(fixtures) > 0
    assert all(fixture.kickoff_utc >= "2017-11-01" for fixture in fixtures)
    # the whole double round robin is played or to play: the unlisted pairings are added
    pairs = [(home, away) for home, away, *_ in state.played] + list(state.fixtures)
    assert len(set(pairs)) == len(pairs) == 90
    assert any(fixture.match_id < 0 for fixture in fixtures)
    moved = read_finished_since(world, finished, "2017-10-01T00:00:00Z")
    assert moved and set(moved) <= set(finished)
