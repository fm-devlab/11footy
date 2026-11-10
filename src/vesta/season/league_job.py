"""The season job of the five leagues: every morning, each team's final place and points (T1).

    python -m vesta.season.league_job <football-world.db> <forecast-log.db> <cache-folder>
                                      [--now <instant>]

Run after the morning's pre-match job, whose kept past it reads in the same `<cache-folder>`
(`vesta.jobs.past_cache`). For each league of the perimeter, its current season as it stands: the
table from the official results (play-offs out of it), the points deducted, and every match
still to play forecast by the released spine at `now` (the champion with what is known now, not
waiting for rule C). `[season] simulations` seasons from one seeded generator, ranked by the
season's tie-break rule, give `T1.final_position` and `T1.points` of every team with their
decided lines (`vesta.season.league`), written as one `season_day` publication per league and
day with the matches finished since the previous one (`moved_by`). From 5 rounds on, the same
simulated seasons also give the league clubs' streaks, next k matches and 30 days (T3, T12,
T13; `vesta.season.league_windows`), the UEFA phase matches of the 30 days drawn from the cups
model at `now`. Season items are not forecasts of record (choice D).

It refuses what it cannot do honestly: a league whose tie-break rule is neither verified nor
carried forward by a decision (`vesta.season.standings.CARRIED_FORWARD`), a match to play
without a forecast. The gate is the pre-match job's: only the keys `published_quantities`
holds for the competition, at the version registered there; before the release it says what it
would have published.

The plan and the verdict: the season simulation plan (28 October 2026).
"""

import argparse
import sqlite3
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vesta.data_format import has_column
from vesta.determinism import run_on_one_thread
from vesta.forecast_log.store import open_log
from vesta.forecast_log.writing import Publication, publish
from vesta.jobs.dataset_pre_match import forecast_fixtures_with_baseline
from vesta.jobs.fixtures import Fixture, forecast_league_fixtures, format_now_utc
from vesta.jobs.past_cache import hash_release
from vesta.jobs.pre_match import read_current_season, read_published_versions
from vesta.quantities import load_quantities
from vesta.research.repository import run_git
from vesta.season.cup_progression import read_cup_season
from vesta.season.cups_strength import build_cups_strength, fit_cups_mapping
from vesta.season.league import (
    POINTS,
    POSITION,
    LeagueSeason,
    build_league_distributions,
    simulate_league,
)
from vesta.season.league_windows import (
    KEYS as WINDOW_KEYS,
)
from vesta.season.league_windows import (
    T13,
    CupGame,
    build_window_distributions,
    find_window_dates,
    is_publishable,
)
from vesta.season.standings import read_tie_break_rule
from vesta.spine.cups import CupsParameters
from vesta.spine.world import RESULT_STATUSES, open_world, read_club_matches

ROOT = Path(__file__).resolve().parents[3]
TO_PLAY = ("scheduled", "postponed")


def read_league_season(world: sqlite3.Connection, competition: str, season: int
                       ) -> tuple[int, int, LeagueSeason, list[Fixture], list[int]]:  # fmt: skip
    """(competition id, season id, the season as it stands without matrices, its fixtures, the
    ids of its matches finished) of a league's season from the world."""
    competition_id, season_id = world.execute(
        "SELECT c.competition_id, s.season_id FROM seasons s JOIN competitions c"
        " ON c.competition_id = s.competition_id WHERE c.code = ? AND s.season = ?",
        (competition, season)).fetchone()  # fmt: skip
    marks = ", ".join("?" for _ in RESULT_STATUSES)
    stage = "COALESCE(stage, '')" if has_column(world, "matches", "stage") else "''"  # B7
    rows = world.execute(
        f"SELECT match_id, home_team_id, away_team_id, home_goals, away_goals, status,"
        f" kickoff_utc, neutral_venue FROM matches WHERE season_id = ?"
        f" AND {stage} <> 'play-off' AND (status IN ({marks}) OR status IN (?, ?))"
        f" ORDER BY kickoff_utc, match_id",
        (season_id, *RESULT_STATUSES, *TO_PLAY)).fetchall()  # fmt: skip
    played, fixtures, finished = [], [], []
    for match_id, home, away, home_goals, away_goals, status, kickoff, neutral in rows:
        if status in RESULT_STATUSES and home_goals is not None:
            played.append((home, away, home_goals, away_goals))
            finished.append(match_id)
        elif status in TO_PLAY:
            fixtures.append(Fixture(match_id, competition, "league", season, home, away, kickoff,
                                    bool(neutral), ""))  # fmt: skip
    teams = sorted({row[1] for row in rows} | {row[2] for row in rows})
    fixtures += make_unscheduled_fixtures(competition, season, season_id, teams, played, fixtures,
                                          max((row[6] for row in rows), default=""))  # fmt: skip
    has_standings = world.execute("SELECT 1 FROM sqlite_master WHERE type = 'table'"
                                  " AND name = 'standings'").fetchone()  # B8, optional
    last = world.execute("SELECT MAX(after_matchday) FROM standings WHERE season_id = ?",
                         (season_id,)).fetchone()[0] if has_standings else None  # fmt: skip
    deductions = dict(world.execute(
        "SELECT team_id, points_deducted FROM standings WHERE season_id = ? AND after_matchday = ?"
        " AND points_deducted <> 0", (season_id, last))) if last is not None else {}  # fmt: skip
    season_state = LeagueSeason(tuple(teams), tuple(played), deductions,
                                tuple((f.home_team_id, f.away_team_id) for f in fixtures),
                                np.zeros((0, 11, 11)))  # fmt: skip
    return competition_id, season_id, season_state, fixtures, finished


def make_unscheduled_fixtures(
    competition: str,
    season: int,
    season_id: int,
    teams: list[int],
    played: list,
    fixtures: list[Fixture],
    latest_kickoff: str,
) -> list[Fixture]:
    """The matches of the double round robin the world does not list yet (its fixtures reach a
    few weeks ahead): every home-away pairing neither played nor scheduled, forecast at now like
    the others. Their ids are negative, never the world's; their kick-off is unknown, so the
    season's latest one stands in (the forecast is made at now, never at the kick-off)."""
    known = {(home, away) for home, away, *_ in played}
    known |= {(fixture.home_team_id, fixture.away_team_id) for fixture in fixtures}
    missing = [(home, away) for home in teams for away in teams
               if home != away and (home, away) not in known]  # fmt: skip
    return [Fixture(-(season_id * 10_000 + index), competition, "league", season, home, away,
                    latest_kickoff, False, "")
            for index, (home, away) in enumerate(missing, start=1)]  # fmt: skip


def read_cup_games(world: sqlite3.Connection, spine: dict, season: int, now: str
                   ) -> dict[int, list[CupGame]]:  # fmt: skip
    """Per club, its UEFA phase matches still to play in the 30 days from `now`, each with the
    cups model's matrix at `now` (T13)."""
    parameters = CupsParameters.from_config(spine)
    end, _ = find_window_dates(now, "UTC")
    matches = read_club_matches(world, list(parameters.leagues), list(parameters.cups),
                                parameters.first_league_season, parameters.first_cup_season,
                                season)  # fmt: skip
    strength = build_cups_strength(
        matches, parameters, fit_cups_mapping(matches, parameters, season), now
    )
    games: dict[int, list[CupGame]] = {}
    for code in parameters.cups:
        try:
            cup = read_cup_season(world, code, season)
        except ValueError:
            continue  # not in the world yet
        for match in cup.phase:
            if match.played or not now < match.kickoff <= end:
                continue
            matrix = strength(match.home, match.away, match.neutral)[0]
            games.setdefault(match.home, []).append(CupGame(match.kickoff, True, matrix))
            games.setdefault(match.away, []).append(CupGame(match.kickoff, False, matrix))
    return games


def read_previous_publication(log: sqlite3.Connection, season_id: int) -> str | None:
    row = log.execute("SELECT MAX(published_at) FROM publications WHERE stage = 'season_day'"
                      " AND season_id = ?", (season_id,)).fetchone()  # fmt: skip
    return row[0]


@dataclass(frozen=True)
class SeasonDay:
    """One league's season as simulated at `now`, ready to publish."""

    competition: str
    competition_id: int
    season_id: int
    distributions: list
    carried_forward: bool
    finished: list[int]  # the ids of its matches finished


def simulate_league_seasons(world: sqlite3.Connection, leagues: list[str], now: str,
                            spine: dict, expected: dict, allowed: dict, cache: Path | None,
                            release: str, skip=lambda season_id: False, say=print,
                            counts: dict | None = None,
                            baseline_leagues: tuple[str, ...] = ()
                            ) -> tuple[int, list[SeasonDay]]:  # fmt: skip
    """The season of each league as it stands at `now`, simulated, with the distributions the
    gate `allowed` ((key, competition id) -> version) publishes; `skip(season_id)` leaves a
    league out (already published today). The matches to play are the champion's matrices, but
    in `baseline_leagues` (a dataset without the shots' xg) B0's. Returns the season and the
    days to publish."""
    research = tomllib.loads((ROOT / "config/vesta/research.toml").read_text(encoding="utf-8"))
    draws, seed = int(spine["season"]["simulations"]), int(research["verdict"]["seed"])
    definitions = load_quantities()
    counts = counts if counts is not None else {}
    for name in ("published", "would_publish", "refused", "already"):
        counts.setdefault(name, 0)
    season = read_current_season(world, leagues, now)
    states = {league: read_league_season(world, league, season) for league in leagues}
    fixtures = [fixture for state in states.values() for fixture in state[3]]
    led = [fixture for fixture in fixtures if fixture.competition not in baseline_leagues]
    forecasts = (forecast_league_fixtures(world, led, now, spine, expected, season, cache,
                                          release)[0] if led else [])  # fmt: skip
    forecasts += forecast_fixtures_with_baseline(world, fixtures, now, spine,
                                                 list(baseline_leagues))  # fmt: skip
    zones = dict(world.execute("SELECT code, time_zone FROM competitions"))
    thirty_days = (read_cup_games(world, spine, season, now)
                   if any(allowed.get((T13, state[0])) for state in states.values())
                   else {})  # fmt: skip
    matrices = {forecast.fixture.match_id: forecast.matrix for forecast in forecasts}
    days = []
    for league, (competition_id, season_id, state, league_fixtures, finished) in states.items():
        rule = read_tie_break_rule(league, season)
        if not rule.publishable:
            say(f"{league} {season}: its tie-break rule is neither verified nor carried forward"
                " by a decision; nothing is published")
            counts["refused"] += 1
            continue
        missing = [f.match_id for f in league_fixtures if f.match_id not in matrices]
        if missing:
            say(f"{league} {season}: no forecast for the matches {missing[:5]}; nothing is"
                " published")  # fmt: skip
            counts["refused"] += 1
            continue
        versions = {allowed.get((key, competition_id)) for key in (POSITION, POINTS)}
        if skip(season_id):
            counts["already"] += 1
            continue
        windows = {key: allowed[(key, competition_id)] for key in WINDOW_KEYS
                   if allowed.get((key, competition_id))}  # fmt: skip
        if not is_publishable(len(state.team_ids), len(state.played)):
            windows = {}  # before 5 rounds: not published
        whole = None not in versions and len(versions) == 1
        if not whole and not windows:
            say(f"{league} {season}: T1 is not published for it (the gate); it would be")
            counts["would_publish"] += 1
            continue
        state = LeagueSeason(state.team_ids, state.played, state.deductions, state.fixtures,
                             np.array([matrices[f.match_id] for f in league_fixtures]
                                      ).reshape(-1, 11, 11))  # fmt: skip
        simulated = simulate_league(state, rule, draws, seed)
        distributions = (build_league_distributions(state, simulated, versions.pop(), definitions)
                         if whole else [])  # fmt: skip
        distributions += build_window_distributions(
            state.team_ids, state.played, league_fixtures, simulated, now, zones[league],
            windows, definitions, thirty_days, np.random.default_rng([seed, season_id]))
        days.append(SeasonDay(league, competition_id, season_id, distributions,
                              rule.carried_forward, finished))  # fmt: skip
    return season, days


@run_on_one_thread
def run(world_path: Path, log_path: Path, cache_folder: Path, now: str, say=print) -> dict:
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    expected = tomllib.loads((ROOT / "config/vesta/expected_goals.toml").read_text(
        encoding="utf-8"))  # fmt: skip
    leagues = list(spine["walk_forward"]["competitions"])
    log = open_log(log_path)
    allowed = read_published_versions(log)
    commit = run_git(ROOT, "rev-parse", "HEAD").strip()
    counts = {"published": 0, "would_publish": 0, "refused": 0, "already": 0}
    previous = {}

    def is_done_today(season_id: int) -> bool:
        previous[season_id] = read_previous_publication(log, season_id)
        return previous[season_id] is not None and previous[season_id][:10] == now[:10]

    with open_world(world_path) as world:
        season, days = simulate_league_seasons(world, leagues, now, spine, expected, allowed,
                                               cache_folder, hash_release(ROOT),
                                               is_done_today, say, counts)  # fmt: skip
        for day in days:
            before = previous.get(day.season_id)
            moved_by = read_finished_since(world, day.finished, before) if before else []
            publish(log, Publication(stage="season_day", trigger="morning" if before else "first",
                                     published_at=now, code_commit=commit,
                                     season_id=day.season_id, moved_by=moved_by,
                                     world_build=world_path.name,
                                     tie_break_carried_forward=day.carried_forward),
                    day.distributions)  # fmt: skip
            counts["published"] += 1
    world.close()
    log.close()
    say(f"season {season}: {counts['published']} leagues published, {counts['would_publish']}"
        f" held by the gate, {counts['refused']} refused, {counts['already']} already today")
    return counts


def read_finished_since(world: sqlite3.Connection, finished: list[int], since: str) -> list[int]:
    """The matches finished whose kick-off falls after the previous publication."""
    if not finished:
        return []
    marks = ", ".join("?" for _ in finished)
    return [row[0] for row in world.execute(
        f"SELECT match_id FROM matches WHERE match_id IN ({marks}) AND kickoff_utc > ?"
        f" ORDER BY kickoff_utc, match_id", (*finished, since))]  # fmt: skip


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("world", type=Path)
    parser.add_argument("log", type=Path)
    parser.add_argument("cache_folder", type=Path)
    parser.add_argument("--now", default=None)
    options = parser.parse_args(arguments)
    run(options.world, options.log, options.cache_folder, options.now or format_now_utc())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
