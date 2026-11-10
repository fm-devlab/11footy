"""The pre-match job on our world: the forecasts of the matches of the coming days, every family.

    python -m vesta.jobs.pre_match <football-world.db> <forecast-log.db> <cache-folder>
                                   [--hours 168] [--now <instant>]

For every match of the perimeter to play in the next `hours` whose two teams have played their
previous match (`vesta.jobs.fixtures`): the forecast made now, its spine distributions (M1-M5),
its goals-in-time distributions, the team counts, lineups, scorers and player thresholds the
release publishes, and the explanations of the explained lines, written as one pre-match
publication (trigger `first` for the match's first, `morning` after; at most one a day per
match).

The gate: a distribution is written only when the log's `published_quantities` holds its key for
the match's competition (registered from the release plan, `vesta.release_gate`); its
`model_version` is the one registered there. With nothing registered the job publishes nothing
and says what it would have published, by competition. A side's scorers (P3, P5) are held when a
player of its pool has a last league match without located shots
(`vesta.players.publication.uncovered_last_matches`); a side whose latest matches without them
are no more than `vesta.players.publication.STALE_MATCHES` is rated on its earlier matches'
shots, and its scorers carry the caveat `shots_stale`.

The past of the seasons before the current one, and the season's priors and corrections, are kept
in the cache folder (`vesta.jobs.past_cache`), so a later run rates only the current season.
"""

import argparse
import json
import sqlite3
import sys
import tomllib
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from vesta.code_version import read_code_version
from vesta.forecast_log.store import open_log
from vesta.forecast_log.writing import (
    Caveat,
    Change,
    Publication,
    PublishedDistribution,
    SideInputs,
    publish,
    record_expected_eleven,
    record_explanation,
)
from vesta.jobs.fixtures import (
    FixtureForecast,
    forecast_cup_fixtures,
    forecast_league_fixtures,
    forecast_nation_fixtures,
    format_now_utc,
    read_fixtures,
)
from vesta.jobs.past_cache import hash_release
from vesta.lineups.model import load_ban_competitions
from vesta.lineups.publication import LineupsOfTheDay, TeamLineup
from vesta.players.publication import PlayersOfTheDay
from vesta.players.thresholds_publication import ThresholdsOfTheDay
from vesta.spine.cups import CupsParameters
from vesta.spine.explanation import EloParts, ForecastParts, explain_elo_match, explain_match
from vesta.spine.goal_minutes import (
    TimingSeasons,
    fit_profile,
    measure_added_minutes,
    read_goal_history,
)
from vesta.spine.goals_in_time import (
    GoalsInTimeParameters,
    TimeShares,
    fit_first_half_shares,
    fit_method_shares,
    simulate,
    split_regular_minutes,
    uses_minute_profile,
)
from vesta.spine.goals_in_time_publication import build_goals_in_time_distributions
from vesta.spine.publication import build_spine_distributions
from vesta.spine.timing import TimeRules, find_result_known_at
from vesta.spine.world import (
    open_world,
    read_goal_methods,
    read_half_time_scores,
    read_league_matches,
)
from vesta.team_counts.publication import (
    CANDIDATES,
    fit_counts_of_the_day,
    fit_cup_counts_of_the_day,
    load_or_build_corrections,
)

ROOT = Path(__file__).resolve().parents[3]


def read_published_versions(log: sqlite3.Connection) -> dict[tuple[str, int], str]:
    """(quantity key, competition id) -> the model version publishing it now."""
    return {(key, competition): version for key, competition, version in log.execute(
        "SELECT quantity_key, competition_id, model_version FROM published_quantities"
        " WHERE published_until IS NULL")}  # fmt: skip


TEAM_COUNT_PREFIXES = ("M15.", "M17.", "M19.", "M20.", "M21.")


def forecast_team_counts(world, leagues, season: int, now: str, spine: dict, allowed, fixtures,
                         folder: Path, identity: str):  # fmt: skip
    """The leagues' and the cups' team counts of the morning (family A,
    `vesta.team_counts.publication`): each count's candidate, the season's corrections built
    once a season and kept in the cache folder under `identity`, the fits of the week;
    (None, None) when no team-count key is published."""
    if not any(key.startswith(TEAM_COUNT_PREFIXES) for key, _ in allowed):
        return None, None
    counts = tuple(CANDIDATES["leagues"])
    leagues_counts = fit_counts_of_the_day(
        world,
        leagues,
        season,
        now,
        spine,
        counts,
        load_or_build_corrections(world, "leagues", season, now, spine, counts, folder, identity),
    )
    if not any(fixture.kind == "cup" for fixture in fixtures):
        return leagues_counts, None
    counts = tuple(CANDIDATES["cups"])
    return leagues_counts, fit_cup_counts_of_the_day(
        world, season, now, spine,
        load_or_build_corrections(world, "cups", season, now, spine, counts, folder, identity))


def fit_current_timing(world: sqlite3.Connection, spine: dict,
                       season: int) -> tuple[TimeShares, tuple[float, float]]:  # fmt: skip
    """Goals in time's shares and the added time read from the goals' minute profile, fitted on
    the `[timing]` seasons before `season` as the verdicts did."""
    timing = TimingSeasons.from_config(spine)
    competitions = list(spine["walk_forward"]["competitions"])
    parameters = GoalsInTimeParameters.from_config(spine)
    history = read_goal_history(world, competitions, timing.first_season, season - 1)
    profile = fit_profile(
        history,
        history.select(history.season >= timing.find_recent_seasons(season)[0]),
        timing.smoothing_minutes,
    )
    scores = read_half_time_scores(world, competitions, timing.first_season, season - 1)
    home, away = fit_first_half_shares(
        *(np.array(column) for column in (scores.home_goals, scores.away_goals,
                                          scores.home_half_time, scores.away_half_time)),
        parameters.first_half_share_without_data)  # fmt: skip
    methods = read_goal_methods(world, competitions, timing.first_season, season - 1)
    # the minute within a half from the profile when the published model says so (H2)
    minutes = split_regular_minutes(profile) if uses_minute_profile(spine) else None
    shares = TimeShares(home, away, fit_method_shares(methods), regular_minutes=minutes)
    return shares, measure_added_minutes(profile)


def build_distributions(forecast: FixtureForecast, shares: TimeShares | None, parameters, allowed,
                        competition_id: int, team_counts=None,
                        lineups: tuple[TeamLineup | None, ...] = (),
                        scorers: list[PublishedDistribution] | tuple = (),
                        cup_counts=None) -> list[PublishedDistribution]:  # fmt: skip
    """The fixture's distributions whose keys the gate lets through, each with its version;
    with `team_counts` (`vesta.team_counts.publication.CountsOfTheDay`) its team counts too,
    with `lineups` (each side's `vesta.lineups.publication.TeamLineup`) its lineups, and with
    `scorers` its players' scorers and thresholds (`vesta.players.publication.PlayersOfTheDay`,
    `vesta.players.thresholds_publication.ThresholdsOfTheDay`). Without `shares`, no goals in
    time."""
    fixture = forecast.fixture
    found = build_spine_distributions(forecast.matrix, fixture.match_id, fixture.home_team_id,
                                      fixture.away_team_id, "")  # fmt: skip
    if shares is not None:
        population = simulate(forecast.matrix / forecast.matrix.sum(), shares, parameters,
                              fixture.match_id)  # fmt: skip
        found += build_goals_in_time_distributions(population, fixture.match_id,
                                                   fixture.home_team_id, fixture.away_team_id,
                                                   "")  # fmt: skip
    if team_counts is not None and fixture.kind == "league":
        found += team_counts.build_fixture_distributions(
            fixture.competition, fixture.match_id, fixture.home_team_id, fixture.away_team_id
        )
    if cup_counts is not None and fixture.kind == "cup" and forecast.parts is not None:
        found += cup_counts.build_fixture_distributions(
            fixture.match_id, fixture.competition, fixture.home_team_id, fixture.away_team_id,
            neutral=fixture.neutral, home_elo=forecast.parts.home_rating,
            away_elo=forecast.parts.away_rating)  # fmt: skip
    for side in lineups:
        if side is not None:
            found += side.distributions
    found += list(scorers)
    chosen = []
    for item in found:
        version = allowed.get((item.quantity_key, competition_id))
        if version is not None:
            chosen.append(PublishedDistribution(item.quantity_key, item.subject_id, version,
                                                item.values, item.window_key))  # fmt: skip
    return chosen


def read_trigger(log: sqlite3.Connection, match_id: int, today: str) -> str | None:
    """`first`, `morning`, or None when the match was published today already."""
    rows = log.execute("SELECT published_at FROM publications WHERE match_id = ? AND stage ="
                       " 'pre_match'", (match_id,)).fetchall()  # fmt: skip
    if not rows:
        return "first"
    return None if any(row[0][:10] == today for row in rows) else "morning"


def run(world_path: Path, log_path: Path, cache: Path, hours: float, now: str,
        say=print) -> dict:  # fmt: skip
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    expected = tomllib.loads((ROOT / "config/vesta/expected_goals.toml").read_text(
        encoding="utf-8"))  # fmt: skip
    leagues = list(spine["walk_forward"]["competitions"])
    cups = list(CupsParameters.from_config(spine).cups)
    log = open_log(log_path)
    allowed = read_published_versions(log)
    kinds = dict(log.execute("SELECT quantity_key, probability_kind FROM quantities").fetchall())
    commit = read_code_version(ROOT)
    cache.mkdir(parents=True, exist_ok=True)
    counts = {"fixtures": 0, "published": 0, "would_publish": 0, "explanations": 0,
              "elevens": 0}
    held: Counter[str] = Counter()
    with open_world(world_path) as world:
        ids = dict(world.execute("SELECT code, competition_id FROM competitions").fetchall())
        fixtures = read_fixtures(world, now, hours, leagues, cups)
        counts["fixtures"] = len(fixtures)
        season = read_current_season(world, leagues, now)
        known = read_known_results(world, leagues, season, TimeRules.from_config(spine))
        rated: dict = {}
        forecasts, past_matrices = forecast_league_fixtures(
            world, fixtures, now, spine, expected, season, cache, hash_release(ROOT), rated
        )
        forecasts += (forecast_cup_fixtures(world, fixtures, now, spine)
                      + forecast_nation_fixtures(world, fixtures, now, spine))  # fmt: skip
        shares, added = fit_current_timing(world, spine, season)
        team_counts, cup_counts = forecast_team_counts(
            world, leagues, season, now, spine, allowed, fixtures, cache, hash_release(ROOT))
        parameters = replace(GoalsInTimeParameters.from_config(spine), added_time_minutes=added)
        lineups, players, thresholds, sides, scorers = forecast_who_plays(
            world, leagues, season, now, spine, expected, cache, hash_release(ROOT), allowed,
            forecasts, past_matrices, rated)  # fmt: skip
    banned = (load_ban_competitions() if lineups is not None and lineups.world.bans_read
              else frozenset())  # the leagues whose lineups read the bans
    for forecast in forecasts:
        fixture = forecast.fixture
        trigger = read_trigger(log, fixture.match_id, now[:10])
        if trigger is None:
            continue
        chosen = build_distributions(forecast, shares, parameters, allowed,
                                     ids[fixture.competition], team_counts,
                                     sides.get(fixture.match_id, ()),
                                     scorers.get(fixture.match_id, ()), cup_counts)  # fmt: skip
        if not chosen:
            counts["would_publish"] += 1
            held[fixture.competition] += 1
            continue
        publication = Publication(stage="pre_match", trigger=trigger, published_at=now,
                                  code_commit=commit, match_id=fixture.match_id,
                                  scheduled_kickoff_utc=fixture.kickoff_utc,
                                  world_build=world_path.name)  # fmt: skip
        inputs = list_side_inputs(forecast)
        stale_sides = players.stale if players is not None else set()
        caveats = make_caveats(chosen, kinds, fixture.competition in banned,
                               bool({fixture.home_team_id, fixture.away_team_id} & stale_sides))
        publication_id = publish(log, publication, chosen, caveats,
                                 find_changes(log, fixture.match_id, chosen, inputs, known),
                                 inputs)  # fmt: skip
        counts["published"] += 1
        counts["explanations"] += _explain(log, forecast, publication_id, chosen)
        counts["elevens"] += write_expected_elevens(log, publication_id, chosen,
                                            sides.get(fixture.match_id, ()))  # fmt: skip
    log.close()
    by_competition = ", ".join(f"{code} {held[code]}" for code in sorted(held))
    say(f"{counts['fixtures']} fixtures; {counts['published']} published; "
        f"{counts['would_publish']} held by the gate (nothing registered for them"
        f"{': ' + by_competition if held else ''}); {counts['explanations']} explanations; "
        f"{counts['elevens']} expected elevens")
    counts["held"] = dict(held)
    counts["held_leagues"] = sorted(code for code in held if code in leagues)
    uncovered = players.held if players is not None else []
    counts["scorers_held"] = len(uncovered)
    counts["scorers_stale"] = len(players.stale) if players is not None else 0
    if counts["scorers_stale"]:
        say(f"P3/P5 of {counts['scorers_stale']} sides published on the shots of earlier"
            f" matches (shots_stale)")
    counts["uncovered_matches"] = sorted({match for *_, matches in uncovered for match in matches})
    if uncovered:  # the shots are late: the scorers wait for them rather than read zeros
        say(f"P3/P5 held for {len(uncovered)} sides without the shots of their last league"
            f" matches ({', '.join(f'{code} team {team}' for code, team, _ in uncovered)}):"
            f" {', '.join(str(match) for match in counts['uncovered_matches'])}")
    return counts


# the kinds whose chances depend on who plays, and their caveats (FORECAST_LOG §4
# `publication_caveats`): the lineups read the bans of the rules in the leagues of
# `config/vesta/bans.toml` when the world holds them, bans that may miss a rescinded card; no
# injury yet
WHO_PLAYS_KINDS = ("lineups", "scorers", "player_counts")


def make_caveats(chosen: list[PublishedDistribution], kinds: dict[str, str],
                 bans_read: bool = False, shots_stale: bool = False) -> list[Caveat]:  # fmt: skip
    """The caveats of a publication: every kind of its distributions that depends on who plays,
    each absence the lineups model did not read (the bans outside the leagues that read them),
    and where it read the bans, that they come from the cards; with that kind's version. With
    `shots_stale`, the scorers were rated without the shots of their latest matches."""
    missing = ("suspensions_from_cards" if bans_read else "suspensions_not_included",
               "injuries_not_included")
    versions: dict[str, str] = {}
    for item in chosen:
        kind = kinds.get(item.quantity_key)
        if kind in WHO_PLAYS_KINDS:
            versions.setdefault(kind, item.model_version)
    found = [Caveat(kind, code, version) for kind, version in sorted(versions.items())
             for code in missing]  # fmt: skip
    if shots_stale and "scorers" in versions:
        found.append(Caveat("scorers", "shots_stale", versions["scorers"]))
    return found


def read_known_results(world, leagues: list[str], season: int,
                       rules: TimeRules) -> dict[int, list[tuple[str, int]]]:  # fmt: skip
    """team -> its league results of the season and the one before, each with the instant it is
    known, in that order (the weekly fit takes those known by its Monday)."""
    found: dict[int, list[tuple[str, int]]] = {}
    for match in read_league_matches(world, leagues, season - 1, season):
        known_at = find_result_known_at(
            match.kickoff_utc, match.kickoff_precision, match.local_date, rules
        )
        for team in (match.home_team_id, match.away_team_id):
            found.setdefault(team, []).append((known_at, match.match_id))
    return {team: sorted(items) for team, items in found.items()}


def list_side_inputs(forecast: FixtureForecast) -> list[SideInputs]:
    """What the forecast read for each side: the Monday of the player ratings' weekly fit and
    the eleven they read; none when the ratings had no part in it."""
    parts = forecast.parts
    ratings = parts.ratings if isinstance(parts, ForecastParts) else None
    if ratings is None:
        return []
    week = datetime.fromtimestamp(ratings["fit_week"], UTC).strftime("%Y-%m-%d")
    fixture = forecast.fixture
    return [SideInputs(fixture.home_team_id, week, tuple(ratings["home_eleven"])),
            SideInputs(fixture.away_team_id, week, tuple(ratings["away_eleven"]))]


def find_changes(log, match_id: int, chosen: list[PublishedDistribution],
                 inputs: list[SideInputs],
                 known: dict[int, list[tuple[str, int]]]) -> list[Change]:  # fmt: skip
    """What changed since the match's previous pre-match publication, facts only (FORECAST_LOG
    §4 `publication_changes`): a new release, a new weekly fit with the side's league results it
    added, a different eleven read. None for a first
    publication; a comparison the previous publication did not record (before schema 5) is not
    made."""
    row = log.execute("SELECT publication_id FROM publications WHERE match_id = ? AND stage ="
                      " 'pre_match' ORDER BY publication_id DESC LIMIT 1",
                      (match_id,)).fetchone()  # fmt: skip
    if row is None or not chosen:
        return []
    previous = row[0]
    version = next((item.model_version for item in chosen if item.quantity_key == "M1.result"),
                   chosen[0].model_version)  # fmt: skip
    found = []
    before = log.execute("SELECT model_version FROM distributions WHERE publication_id = ? AND"
                         " quantity_key = 'M1.result'", (previous,)).fetchone()  # fmt: skip
    if before is not None and before[0] != version:
        found.append(Change("release_changed", 0, {"from": before[0], "to": version}, previous,
                            version))  # fmt: skip
    read = {team: (week, json.loads(eleven)) for team, week, eleven in log.execute(
        "SELECT team_id, fit_week, eleven FROM publication_inputs WHERE publication_id = ?",
        (previous,))}  # fmt: skip
    for side in inputs:
        if side.team_id not in read:
            continue
        week, eleven = read[side.team_id]
        if week is not None and side.fit_week is not None and week != side.fit_week:
            added = [match for known_at, match in known.get(side.team_id, [])
                     if f"{week}T00:00:00Z" < known_at <= f"{side.fit_week}T00:00:00Z"]
            found.append(Change("weekly_fit", side.team_id,
                                {"fit_week": side.fit_week, "results_added": added}, previous,
                                version))  # fmt: skip
        if set(eleven) != set(side.eleven):
            found.append(Change("eleven_changed", side.team_id,
                                {"out": sorted(set(eleven) - set(side.eleven)),
                                 "in": sorted(set(side.eleven) - set(eleven))}, previous,
                                version))  # fmt: skip
    return found


def write_expected_elevens(log, publication_id: int, chosen, sides) -> int:
    """The expected eleven of each side whose starts the gate published, with their version."""
    versions = {item.subject_id: item.model_version for item in chosen
                if item.quantity_key == "L1.starts"}  # fmt: skip
    written = 0
    for side in sides:
        if side is None or not side.slots:
            continue
        version = versions.get(side.slots[0].player_id)
        if version is None:
            continue
        record_expected_eleven(log, publication_id, side.team_id, side.formation,
                               side.probability_formation, side.slots, version)  # fmt: skip
        written += 1
    return written


def _explain(log, forecast: FixtureForecast, publication_id: int, chosen) -> int:
    keys = {(item.quantity_key, item.model_version) for item in chosen}
    version = next(iter(keys))[1]
    if isinstance(forecast.parts, ForecastParts):
        explanations = explain_match(forecast.parts, publication_id, forecast.fixture.match_id,
                                     version)  # fmt: skip
    elif isinstance(forecast.parts, EloParts):
        explanations = explain_elo_match(forecast.parts, publication_id,
                                         forecast.fixture.match_id, version)  # fmt: skip
    else:
        return 0
    published = {key for key, _ in keys}
    written = 0
    for explanation in explanations:
        if explanation.quantity_key in published:
            record_explanation(log, explanation)
            written += 1
    return written


def read_current_season(world: sqlite3.Connection, leagues: list[str], now: str) -> int:
    """The season of the league matches nearest after `now` (the last one played when none)."""
    marks = ", ".join("?" for _ in leagues)
    for comparison, order in ((">=", "ASC"), ("<", "DESC")):
        row = world.execute(
            f"SELECT s.season FROM matches m JOIN seasons s ON s.season_id = m.season_id"
            f" JOIN competitions c ON c.competition_id = s.competition_id WHERE c.code IN"
            f" ({marks}) AND m.kickoff_utc {comparison} ? ORDER BY m.kickoff_utc {order} LIMIT 1",
            (*leagues, now)).fetchone()  # fmt: skip
        if row is not None:
            return int(row[0])
    return int(now[:4])


def forecast_who_plays(world, leagues, season, now, spine, expected, cache: Path, identity: str,
                       allowed, forecasts, past_matrices, rated) -> tuple:  # fmt: skip
    """The morning's lineups, scorers and thresholds of the day (None where no key of theirs is
    published), each league fixture's sides and its players' distributions; their season's
    parts kept in `cache` under `identity`."""
    lineups = (LineupsOfTheDay(world, leagues, season, now, TimeRules.from_config(spine),
                               cache, identity)
               if any(key.startswith(("L1.", "L2.", "L6.", "L9.")) for key, _ in allowed)
               else None)  # fmt: skip
    players = (PlayersOfTheDay(world, leagues, season, now, TimeRules.from_config(spine),
                               rated, expected["shots"], cache, identity, lineups.world)
               if lineups is not None and rated
               and any(key.startswith(("P3.", "P5.")) for key, _ in allowed)
               else None)  # fmt: skip
    thresholds = (ThresholdsOfTheDay(world, leagues, season, now, TimeRules.from_config(spine),
                                     [item for item in forecasts
                                      if item.fixture.kind == "league"],
                                     past_matrices, cache, identity, lineups.world)
                  if lineups is not None
                  and any(key.startswith(("P1.", "P2.", "P7.", "P8.", "P9."))
                          for key, _ in allowed)
                  else None)  # fmt: skip
    sides, scorers = {}, {}
    for forecast in forecasts:
        fixture = forecast.fixture
        if lineups is not None and fixture.kind == "league":
            sides[fixture.match_id] = lineups.build_fixture_lineups(
                fixture.match_id, fixture.competition, fixture.season, fixture.kickoff_utc,
                fixture.home_team_id, fixture.away_team_id)  # fmt: skip
            if players is not None:
                scorers[fixture.match_id] = [
                    item for side, home in zip(sides[fixture.match_id], (True, False),
                                               strict=True) if side is not None
                    for item in players.build_side_distributions(side, forecast.matrix, home,
                                                                 fixture.competition)]  # fmt: skip
            if thresholds is not None:
                scorers.setdefault(fixture.match_id, []).extend(
                    item
                    for side in sides[fixture.match_id]
                    if side is not None
                    for item in thresholds.build_side_distributions(side, fixture.match_id)
                )
    return lineups, players, thresholds, sides, scorers


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("world", type=Path)
    parser.add_argument("log", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--hours", type=float, default=168.0)
    parser.add_argument("--now", default=None)
    options = parser.parse_args(arguments)
    run(options.world, options.log, options.cache, options.hours, options.now or format_now_utc())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
