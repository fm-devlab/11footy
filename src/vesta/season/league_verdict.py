"""The verdict of the season simulation of the five leagues (T1), walk-forward, declared before
the run in its plan (b4ac04d).

    python -m vesta.season.league_verdict <football-world.db> [--stage development|blind]
                                          [--draws 10000] [--out <report.md>]

At four checkpoints of every league-season (after 5, 10, 19 and 29 rounds' worth of matches: the end
of the UTC day on which rounds x teams / 2 matches had been played) the simulation runs with only
what was known then: one replay of the champion's families over every league match, with, beside
each match's own request, a request at every checkpoint for each match of the season still to play
(the point-in-time engine refuses a fact known after a request), the matrices calibrated on the
seasons before, the table as it stood, `draws` seasons ranked by the season's tie-break rule. The
truth is the final table ranked from the official results by the same rule, checked against the
world's standings.

Baselines, no model, their statistics from earlier seasons only: uniform; position (the share of
the teams in the same place of the same league at the same checkpoint that finished first, in the
top four, in the bottom three, and in each place); ppg (points per game extrapolated, normal noise
with a spread proportional to the root of the games left, fitted on the earlier seasons); and
ppg_shrunk (the same with the points per game shrunk toward the league's by c games, c chosen on
the earlier seasons). Scores per team and checkpoint: Brier and log loss (held to [0.0001,
0.9999]) of the title, the top four and the bottom three, the ranked probability score of the
final place, the log score of the final points (Vesta only). Intervals: bootstrap over the
league-seasons, paired, baseline minus Vesta (positive: Vesta better).
"""

import argparse
import json
import math
import sqlite3
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from vesta.determinism import run_on_one_thread
from vesta.expected_goals.model import ModelSettings
from vesta.expected_goals.shots import read_shots
from vesta.expected_goals.walk_forward import WalkForwardSettings as ShotWalkForwardSettings
from vesta.expected_goals.walk_forward import rate_every_season, sum_expected_goals_by_match
from vesta.research.bootstrap import bootstrap_mean
from vesta.season.league import LeagueSeason, simulate_league
from vesta.season.standings import SeasonMatches, count_season_totals, rank, read_tie_break_rule
from vesta.spine.dixon_coles import DixonColesParameters, ExpectedGoalsDixonColesFamily
from vesta.spine.player_ratings import PlayerRatingsFamily, PlayerRatingsParameters, blend
from vesta.spine.point_in_time import ForecastRequest, replay
from vesta.spine.score_matrix import score_matrices
from vesta.spine.timing import TimeRules, find_result_known_at
from vesta.spine.walk_forward import (
    MatchFacts,
    WalkForwardSettings,
    _build_events,
    _fit_calibration_or_neutral,
    _keep_forecastable,
)
from vesta.spine.world import open_world, read_league_matches, read_match_players

ROOT = Path(__file__).resolve().parents[3]
ROUNDS = (5, 10, 19, 29)
LINES = ("title", "top_four", "bottom_three")
BASELINES = ("uniform", "position", "ppg", "ppg_shrunk")
SHRINK_GRID = (0, 2, 5, 10, 15, 20, 30)
HELD = 1e-4
LEFT_OUT = {("L1", 2019)}  # stopped after 279 of 380 matches (COVID-19): no full season
PREVIOUS = {"title": 0.0184, "top_four": 0.0487, "bottom_three": 0.0694}  # reference


@dataclass(frozen=True)
class Game:
    match_id: int
    home: int
    away: int
    home_goals: int
    away_goals: int
    kickoff_utc: str
    known_at: str


@dataclass(frozen=True)
class LeagueYear:
    competition: str
    season: int
    team_ids: tuple[int, ...]
    games: tuple[Game, ...]  # in the order of kick-off
    deductions: dict[int, int]  # the final standings' points deducted
    # a play-off between two teams of the league for a place (Serie A 2022-23): (winner, loser)
    deciders: tuple[tuple[int, int], ...] = ()


def find_checkpoint_instant(games: tuple[Game, ...], teams: int, rounds: int) -> str:
    """The end of the UTC day on which rounds x teams / 2 matches had been played."""
    ordered = sorted(game.kickoff_utc for game in games)
    return f"{ordered[rounds * teams // 2 - 1][:10]}T23:59:59Z"


def read_years(world: sqlite3.Connection, competitions: list[str], time_rules: TimeRules
               ) -> dict[tuple[str, int], LeagueYear]:  # fmt: skip
    """Every closed league-season of `competitions` in the world, with its official results."""
    games: dict[tuple[str, int], list[Game]] = defaultdict(list)
    play_offs: dict[tuple[str, int], list[tuple[int, int, int]]] = defaultdict(list)
    season_ids = {}
    marks = ", ".join("?" for _ in competitions)
    for (match_id, code, season, season_id, home, away, home_goals, away_goals, kickoff,
         precision, local_date, stage, winner) in world.execute(
            f"SELECT m.match_id, c.code, s.season, s.season_id, m.home_team_id, m.away_team_id,"
            f" m.home_goals, m.away_goals, m.kickoff_utc, m.kickoff_precision, m.local_date,"
            f" m.stage, m.winner_team_id"
            f" FROM matches m JOIN seasons s ON s.season_id = m.season_id JOIN competitions c"
            f" ON c.competition_id = s.competition_id WHERE c.code IN ({marks})"
            f" AND s.status = 'closed' AND m.status IN ('played', 'awarded')"
            f" AND m.home_goals IS NOT NULL ORDER BY m.kickoff_utc, m.match_id",
            competitions):  # fmt: skip
        if stage == "play-off":  # not a match of the table
            winner = winner or (home if home_goals > away_goals else away)
            play_offs[(code, season)].append((home, away, winner))
            continue
        known = find_result_known_at(kickoff, precision, local_date, time_rules)
        games[(code, season)].append(Game(match_id, home, away, home_goals, away_goals, kickoff,
                                          known))  # fmt: skip
        season_ids[(code, season)] = season_id
    years = {}
    for key, rows in games.items():
        teams = sorted({game.home for game in rows} | {game.away for game in rows})
        if len(rows) != len(teams) * (len(teams) - 1):
            continue  # not a whole double round robin (a season stopped)
        last = world.execute("SELECT MAX(after_matchday) FROM standings WHERE season_id = ?",
                             (season_ids[key],)).fetchone()[0]  # fmt: skip
        deductions = dict(world.execute(
            "SELECT team_id, points_deducted FROM standings WHERE season_id = ? AND"
            " after_matchday = ? AND points_deducted <> 0", (season_ids[key], last)))  # fmt: skip
        deciders = tuple((winner, home if winner == away else away)
                         for home, away, winner in play_offs.get(key, ())
                         if home in teams and away in teams)  # fmt: skip
        years[key] = LeagueYear(key[0], key[1], tuple(teams), tuple(rows), deductions, deciders)
    return years


_FINAL_PLACES: dict[tuple[str, int, int], dict[int, int]] = {}


def rank_final_places(year: LeagueYear, seed: int) -> dict[int, int]:
    """Each team's final place (1 the first), ranked from the official results by the rule."""
    key = (year.competition, year.season, seed)
    if key not in _FINAL_PLACES:
        _FINAL_PLACES[key] = _rank_from_results(year, seed)
    return _FINAL_PLACES[key]


def _rank_from_results(year: LeagueYear, seed: int) -> dict[int, int]:
    column = {team: index for index, team in enumerate(year.team_ids)}
    rows = np.array([(column[g.home], column[g.away], g.home_goals, g.away_goals)
                     for g in year.games], dtype=np.int64)  # fmt: skip
    matches = SeasonMatches(rows[:, 0], rows[:, 1], rows[None, :, 2], rows[None, :, 3])
    deductions = np.array([year.deductions.get(team, 0) for team in year.team_ids])
    totals = count_season_totals(matches, len(year.team_ids), deductions)
    try:
        rule = read_tie_break_rule(year.competition, year.season)
    except ValueError:  # the seasons before the rules written (2014): the plain order
        rule = None
    if rule is None:
        order = np.lexsort((-totals["goals_for"][0], -totals["goal_difference"][0],
                            -totals["points"][0]))  # fmt: skip
        places = np.empty(len(order), dtype=int)
        places[order] = np.arange(len(order))
    else:
        places = rank(totals, matches, rule, np.random.default_rng(seed))[0]
    found = {team: int(places[column[team]]) + 1 for team in year.team_ids}
    for winner, loser in year.deciders:  # the play-off played settles the two places it decided
        if found[winner] > found[loser]:
            found[winner], found[loser] = found[loser], found[winner]
    return found


def build_table_at(year: LeagueYear, instant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points, games played, place 1-based by points, goal difference and goals) at `instant`,
    the teams in `team_ids` order; the place is the baselines' simple one."""
    column = {team: index for index, team in enumerate(year.team_ids)}
    teams = len(year.team_ids)
    points, played = np.zeros(teams), np.zeros(teams)
    difference, scored = np.zeros(teams), np.zeros(teams)
    for game in year.games:
        if game.known_at > instant:
            continue
        home, away = column[game.home], column[game.away]
        played[[home, away]] += 1
        level = game.home_goals == game.away_goals
        points[home] += 3 if game.home_goals > game.away_goals else level
        points[away] += 3 if game.away_goals > game.home_goals else level
        difference[home] += game.home_goals - game.away_goals
        difference[away] += game.away_goals - game.home_goals
        scored[home] += game.home_goals
        scored[away] += game.away_goals
    order = np.lexsort((-scored, -difference, -points))
    place = np.empty(teams, dtype=int)
    place[order] = np.arange(1, teams + 1)
    return points, played, place


def forecast_checkpoint_matrices(world_path: Path, targets: list[LeagueYear], instants: dict,
                                 spine: dict, expected_goals: dict,
                                 rounds: tuple[int, ...] = ROUNDS) -> dict:  # fmt: skip
    """(competition, season, rounds) -> {match_id: matrix} of the matches still to play at the
    checkpoint, each forecast with only what was known at its instant."""
    asked = {}
    for year in targets:
        for count in rounds:
            instant = instants[(year.competition, year.season, count)]
            for game in year.games:
                if game.known_at > instant:
                    asked[(game.match_id, instant)] = (year.competition, year.season, count)
    return forecast_league_matrices(world_path, asked, spine, expected_goals)


def forecast_league_matrices(world_path: Path, asked: dict[tuple[int, str], object], spine: dict,
                             expected_goals: dict) -> dict:  # fmt: skip
    """For every (league match, instant) asked, the champion's matrix requested at that instant
    (the point-in-time engine: only what was known then), grouped by the key each pair is asked
    under: key -> {match_id: matrix}."""
    settings = WalkForwardSettings.from_config(spine)
    with open_world(world_path) as world:
        wanted_ids = {match_id for match_id, _ in asked}
        last = max(season for (season,) in world.execute(
            "SELECT DISTINCT s.season FROM matches m JOIN seasons s ON s.season_id = m.season_id"
            f" WHERE m.match_id IN ({', '.join(str(int(i)) for i in wanted_ids)})"))  # fmt: skip
    shot_section = expected_goals["shots"]
    with open_world(world_path) as world:
        matches = read_league_matches(world, list(settings.competitions),
                                      settings.first_training_season - 1, last)  # fmt: skip
        shots = read_shots(world, shot_section["competitions"], shot_section["first_season"],
                           last, shot_section["layers"])  # fmt: skip
        players = read_match_players(world, {match.match_id for match in matches})
    chances, _ = rate_every_season(shots, ModelSettings.from_config(expected_goals),
                                   ShotWalkForwardSettings.from_config(expected_goals))  # fmt: skip
    by_match = sum_expected_goals_by_match(shots, chances)
    facts_of = {match.match_id: MatchFacts(*by_match.get(match.match_id, (math.nan, math.nan)),
                                           players.get(match.match_id, ()))
                for match in matches}  # fmt: skip
    facts, requests = _build_events(matches, TimeRules.from_config(spine), settings, facts_of)
    by_id = {match.match_id: match for match in matches}
    for match_id, instant in asked:
        match = by_id[match_id]
        requests.append(ForecastRequest(match_id, match.competition, match.season,
                                        match.home_team_id, match.away_team_id, instant))
    ratings = PlayerRatingsParameters.from_config(spine)
    families = [ExpectedGoalsDixonColesFamily(DixonColesParameters.from_config(
        spine, "dixon_coles_expected_goals")), PlayerRatingsFamily(ratings, settings.competitions)]
    rows = replay(families, facts, requests)
    weight = ratings.blend_weight

    def blend_intensities(features):
        return (blend(features["dixon_coles_expected_goals_home_intensity"],
                      features["player_ratings_home_intensity"], weight),
                blend(features["dixon_coles_expected_goals_away_intensity"],
                      features["player_ratings_away_intensity"], weight))  # fmt: skip

    own = [row for row in rows if (row.request.match_id, row.request.forecast_time) not in asked]
    past = _keep_forecastable(own, matches, blend_intensities)
    seasons = int(spine["champion_calibration"]["seasons"])
    calibrations = {}
    for season in {by_id[match_id].season for match_id, _ in asked}:
        first = max(season - seasons, settings.first_training_season)
        calibrations[season] = _fit_calibration_or_neutral(
            [item for item in past if first <= item.match.season < season], minimum_matches=1,
            fit_spread=True, fit_dispersion=True)  # fmt: skip
    wanted = defaultdict(list)
    for row in rows:
        target = asked.get((row.request.match_id, row.request.forecast_time))
        if target is not None:
            wanted[target].append((row.request.match_id, *blend_intensities(row.features)))
    result = {}
    for target, items in wanted.items():
        home = np.array([item[1] for item in items])
        away = np.array([item[2] for item in items])
        if not (np.isfinite(home).all() and np.isfinite(away).all()):
            missing = [item[0] for item in items if not math.isfinite(item[1] + item[2])]
            raise ValueError(f"{target}: no intensities for the matches {missing[:5]}")
        matrices = score_matrices(home, away, calibrations[target[1]])
        result[target] = dict(zip((item[0] for item in items), matrices, strict=True))
    return result


def simulate_vesta_probabilities(
    year: LeagueYear, instant: str, matrices: dict, draws: int, seed: int, deductions: dict
) -> dict:
    played = tuple((g.home, g.away, g.home_goals, g.away_goals) for g in year.games
                   if g.known_at <= instant)  # fmt: skip
    remaining = [g for g in year.games if g.known_at > instant]
    season = LeagueSeason(year.team_ids, played, deductions,
                          tuple((g.home, g.away) for g in remaining),
                          np.array([matrices[g.match_id] for g in remaining]))  # fmt: skip
    simulated = simulate_league(season, read_tie_break_rule(year.competition, year.season), draws,
                                seed)  # fmt: skip
    return _from_places(simulated.places, len(year.team_ids), simulated.points)


def _from_places(places: np.ndarray, teams: int, points: np.ndarray | None = None) -> dict:
    """places (n, teams), 0 the first -> the lines and the place distribution of every team."""
    distribution = np.stack([np.bincount(places[:, column], minlength=teams) / len(places)
                             for column in range(teams)])  # fmt: skip
    found = {"title": distribution[:, 0], "top_four": distribution[:, :4].sum(axis=1),
             "bottom_three": distribution[:, -3:].sum(axis=1), "places": distribution}
    if points is not None:
        found["points"] = points
    return found


def _place_from_points(points: np.ndarray) -> np.ndarray:
    """(n, teams) points drawn from a continuous law -> places, 0 the first (no level teams)."""
    return np.argsort(np.argsort(-points, axis=1, kind="stable"), axis=1, kind="stable")


def simulate_points_per_game(
    points, played, total_games, league_ppg, shrink, spread, draws, seed
) -> dict:
    left = total_games - played
    rate = (points + shrink * league_ppg) / (played + shrink)
    expected = points + rate * left
    generator = np.random.default_rng(seed)
    drawn = expected + spread * np.sqrt(left) * generator.standard_normal((draws, len(points)))
    return _from_places(_place_from_points(drawn), len(points))


@dataclass
class History:
    """The baselines' statistics from the earlier seasons of the five leagues at one checkpoint."""

    position: dict  # (competition, place at checkpoint) -> [counts of each final place]
    league_ppg: dict  # competition -> points per game
    projections: list  # (points, played, total, league ppg, final points, competition, places)


def build_history(years: dict, before: int, rounds: int, instants: dict, seed: int) -> History:
    position: dict = defaultdict(lambda: None)
    league_points: dict = defaultdict(lambda: [0.0, 0.0])
    projections = []
    for (competition, season), year in sorted(years.items()):
        if season >= before:
            continue
        teams = len(year.team_ids)
        instant = instants.get((competition, season, rounds))
        if instant is None:
            instant = find_checkpoint_instant(year.games, teams, rounds)
        final = rank_final_places(year, seed)
        points, played, place = build_table_at(year, instant)
        final_points, _, _ = build_table_at(year, "9999")
        league_points[competition][0] += final_points.sum()
        league_points[competition][1] += teams * 2 * (teams - 1)
        for column, team in enumerate(year.team_ids):
            key = (competition, teams, int(place[column]))
            counts = position[key]
            if counts is None:
                counts = position[key] = np.zeros(teams)
            counts[final[team] - 1] += 1
        projections.append((points, played, 2 * (teams - 1), competition, final_points,
                            np.array([final[team] - 1 for team in year.team_ids])))  # fmt: skip
    league_ppg = {competition: total / games for competition, (total, games)
                  in league_points.items()}  # fmt: skip
    return History(dict(position), league_ppg, projections)


def fit_spread(projections: list, league_ppg: dict, shrink: float) -> float:
    residuals = []
    for points, played, total, competition, final_points, _ in projections:
        left = total - played
        rate = ((points + shrink * league_ppg[competition]) / (played + shrink) if shrink
                else points / played)
        residuals += list((final_points - points - rate * left) / np.sqrt(left))
    return float(np.std(residuals))


def choose_shrink(projections: list, league_ppg: dict, draws: int, seed: int) -> tuple[int, float]:
    """The c of the grid with the lowest mean Brier of the three lines on the earlier seasons."""
    best = None
    for shrink in SHRINK_GRID[1:]:
        spread = fit_spread(projections, league_ppg, shrink)
        losses = []
        for points, played, total, competition, _, final in projections:
            found = simulate_points_per_game(points, played, total, league_ppg[competition], shrink,
                                             spread, draws, seed)  # fmt: skip
            losses.append(np.mean([_score_brier(found[line], _mark_happened(line, final))
                                   for line in LINES]))  # fmt: skip
        loss = float(np.mean(losses))
        if best is None or loss < best[2]:
            best = (shrink, spread, loss)
    return best[0], best[1]


def _mark_happened(line: str, final: np.ndarray) -> np.ndarray:
    teams = len(final)
    return {"title": final == 0, "top_four": final < 4, "bottom_three": final >= teams - 3}[line]


def _score_brier(probability: np.ndarray, happened: np.ndarray) -> np.ndarray:
    return (probability - happened) ** 2


def _score_log_loss(probability: np.ndarray, happened: np.ndarray) -> np.ndarray:
    held = np.clip(probability, HELD, 1 - HELD)
    return -np.where(happened, np.log(held), np.log(1 - held))


def _score_ranked_probability(distribution: np.ndarray, final: np.ndarray) -> np.ndarray:
    teams = distribution.shape[1]
    cumulative = np.cumsum(distribution, axis=1)[:, :-1]
    truth = (np.arange(teams - 1)[None, :] >= final[:, None]).astype(float)
    return ((cumulative - truth) ** 2).sum(axis=1) / (teams - 1)


@run_on_one_thread
def run(world_path: Path, stage: str, draws: int, say=print) -> dict:
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    research = tomllib.loads((ROOT / "config/vesta/research.toml").read_text(encoding="utf-8"))
    expected_goals = tomllib.loads((ROOT / "config/vesta/expected_goals.toml").read_text(
        encoding="utf-8"))  # fmt: skip
    seed = int(research["verdict"]["seed"])
    seasons = tuple(research["seasons"]["development" if stage == "development" else "closed"])
    competitions = list(WalkForwardSettings.from_config(spine).competitions)
    with open_world(world_path) as world:
        years = read_years(world, competitions, TimeRules.from_config(spine))
        standings = _read_world_final_places(world, competitions)
    targets = [year for key, year in sorted(years.items())
               if key[1] in seasons and key not in LEFT_OUT]  # fmt: skip
    instants = {}
    for year in years.values():
        for rounds in ROUNDS:
            instants[(year.competition, year.season, rounds)] = find_checkpoint_instant(
                year.games, len(year.team_ids), rounds)
    truth_problems = []
    for year in targets:
        ours = rank_final_places(year, seed)
        theirs = standings.get((year.competition, year.season), {})
        truth_problems += [(year.competition, year.season, team, ours[team], theirs.get(team))
                           for team in year.team_ids if ours[team] != theirs.get(team)]  # fmt: skip
    say(f"{len(targets)} league-seasons; the truth differs from the world's standings for"
        f" {len(truth_problems)} teams")  # fmt: skip
    matrices = forecast_checkpoint_matrices(world_path, targets, instants, spine, expected_goals)
    say("matrices of every checkpoint made")
    records = []  # (block, rounds, method, measure, value) per team
    calibration_lines = defaultdict(list)
    for rounds in ROUNDS:
        for year in targets:
            past = build_history(years, year.season, rounds, instants, seed)
            shrink, spread = choose_shrink(past.projections, past.league_ppg, 2000, seed)
            spread_plain = fit_spread(past.projections, past.league_ppg, 0)
            instant = instants[(year.competition, year.season, rounds)]
            final_map = rank_final_places(year, seed)
            final = np.array([final_map[team] - 1 for team in year.team_ids])
            final_points, _, _ = build_table_at(year, "9999")
            final_points = final_points - np.array([year.deductions.get(t, 0)
                                                    for t in year.team_ids])  # fmt: skip
            teams = len(year.team_ids)
            points, played, place = build_table_at(year, instant)
            league = past.league_ppg[year.competition]
            found = {
                "vesta": simulate_vesta_probabilities(
                    year,
                    instant,
                    matrices[(year.competition, year.season, rounds)],
                    draws,
                    seed,
                    {},
                ),
                "uniform": _from_places(np.tile(np.arange(teams), (teams, 1)).T, teams),
                "ppg": simulate_points_per_game(
                    points, played, 2 * (teams - 1), league, 0, spread_plain, draws, seed
                ),
                "ppg_shrunk": simulate_points_per_game(
                    points, played, 2 * (teams - 1), league, shrink, spread, draws, seed
                ),
            }
            position = np.stack([_find_position_shares(past.position, year.competition, teams,
                                               int(place[column])) for column in range(teams)])
            found["position"] = {"title": position[:, 0], "top_four": position[:, :4].sum(1),
                                 "bottom_three": position[:, -3:].sum(1), "places": position}
            block = f"{year.competition}-{year.season}"
            for method, values in found.items():
                for line in LINES:
                    happened = _mark_happened(line, final)
                    for column in range(teams):
                        records.append((block, rounds, method, f"brier_{line}",
                                        float(_score_brier(values[line], happened)[column])))
                        records.append((block, rounds, method, f"log_loss_{line}",
                                        float(_score_log_loss(values[line], happened)[column])))
                    if method == "vesta":
                        calibration_lines[line] += list(zip(values[line].tolist(),
                                                            happened.tolist(), strict=True))
                for value in _score_ranked_probability(values["places"], final):
                    records.append((block, rounds, method, "rps_place", float(value)))
            points_drawn = found["vesta"]["points"]
            for column in range(teams):
                share = float(np.mean(points_drawn[:, column] == final_points[column]))
                records.append((block, rounds, "vesta", "log_score_points",
                                -math.log(max(share, 1.0 / draws))))
        say(f"after {rounds} rounds: done")
    return {"records": records, "calibration": calibration_lines, "seasons": seasons,
            "targets": [(y.competition, y.season) for y in targets],
            "truth_problems": truth_problems, "research": research}  # fmt: skip


def _find_position_shares(position: dict, competition: str, teams: int, place: int) -> np.ndarray:
    counts = position.get((competition, teams, place))
    if counts is None or counts.sum() == 0:
        return np.full(teams, 1.0 / teams)
    return counts / counts.sum()


def _read_world_final_places(world: sqlite3.Connection, competitions: list[str]) -> dict:
    marks = ", ".join("?" for _ in competitions)
    found = defaultdict(dict)
    for code, season, team, place in world.execute(
            f"SELECT c.code, s.season, st.team_id, st.position FROM standings st"
            f" JOIN seasons s ON s.season_id = st.season_id JOIN competitions c"
            f" ON c.competition_id = s.competition_id WHERE c.code IN ({marks})"
            f" AND st.after_matchday = (SELECT MAX(after_matchday) FROM standings x"
            f" WHERE x.season_id = st.season_id)", competitions):  # fmt: skip
        found[(code, season)][team] = place
    return found


def reach_verdict(results: dict) -> tuple[bool, list[str], dict]:
    """The rule of the plan, then the alarm of `[alarm.rank]`; (passed, reasons, table)."""
    research = results["research"]
    repetitions = int(research["verdict"]["bootstrap_repetitions"])
    seed, coverage = int(research["verdict"]["seed"]), float(research["verdict"]["coverage"])
    alarm = research["alarm"]["rank"]
    by = defaultdict(list)
    for block, rounds, method, measure, value in results["records"]:
        by[(method, measure, rounds)].append((block, value))
        by[(method, measure, "all")].append((block, value))
    table, reasons, passed = {}, [], True
    for measure in [f"brier_{line}" for line in LINES] + [f"log_loss_{line}" for line in LINES] \
            + ["rps_place"]:
        for rounds in (*ROUNDS, "all"):
            own = by[("vesta", measure, rounds)]
            row = {"vesta": float(np.mean([v for _, v in own]))}
            for baseline in BASELINES:
                other = by[(baseline, measure, rounds)]
                differences = [b - a for (_, a), (_, b) in zip(own, other, strict=True)]
                row[baseline] = float(np.mean([v for _, v in other]))
                row[f"{baseline}_difference"] = bootstrap_mean(
                    differences, [block for block, _ in own], repetitions, seed, coverage)
            table[(measure, rounds)] = row
    for line in LINES:
        measure = f"brier_{line}"
        row = table[(measure, "all")]
        for baseline in BASELINES:
            if row["vesta"] >= row[baseline]:
                passed = False
                reasons.append(f"{line}: Vesta's Brier {row['vesta']:.4f} is not below"
                               f" {baseline}'s {row[baseline]:.4f} (all checkpoints)")
        for rounds in ROUNDS:
            for baseline in BASELINES:
                _, low, high = table[(measure, rounds)][f"{baseline}_difference"]
                if high < 0:
                    passed = False
                    reasons.append(f"{line}, after {rounds} rounds: {baseline} is better, its"
                                   f" interval [{low:+.4f}; {high:+.4f}] leaves zero")
        gain = row["ppg_shrunk_difference"][0]
        if gain > float(alarm[f"max_gain_{line}"]):
            passed = False
            reasons.append(f"ALARM {line}: the Brier gain over ppg_shrunk {gain:.4f} is above"
                           f" {alarm[f'max_gain_{line}']} - stop and look for a leak")
    rps = table[("rps_place", "all")]
    if rps["vesta"] >= rps["ppg_shrunk"]:
        passed = False
        reasons.append(f"final place: Vesta's ranked probability score {rps['vesta']:.4f} is not"
                       f" below ppg_shrunk's {rps['ppg_shrunk']:.4f}")
    return passed, reasons, table


def report(results: dict, stage: str, draws: int, world_path: Path, commit: str,
           world_sha: str, seconds: float) -> str:  # fmt: skip
    passed, reasons, table = reach_verdict(results)
    lines = [f"# The season simulation's {stage} verdict (T1, the five leagues)", "",
             f"World: `{world_path}` (sha256 {world_sha}). Code: `{commit}`. Seasons:"
             f" {', '.join(map(str, results['seasons']))}; {len(results['targets'])}"
             f" league-seasons; four checkpoints (5, 10, 19, 29 rounds); {draws} simulated"
             f" seasons each. Written by `python -m vesta.season.league_verdict`; the rule is"
             f" the plan declared before the run (b4ac04d), the alarm"
             f" `[alarm.rank]` (7a4e477). {seconds:.0f} seconds.", "",
             f"**Verdict: {'PASSED' if passed else 'NOT PASSED'}.**", ""]
    lines += [f"- {reason}" for reason in reasons] + ([""] if reasons else [])
    lines += [f"The truth (the final table ranked from the official results by the season's rule)"
              f" differs from the world's standings for {len(results['truth_problems'])} teams"
              + (": " + "; ".join(f"{c} {s} team {t}: ours {o}, world's {w}" for c, s, t, o, w
                                  in results["truth_problems"][:10]) if results["truth_problems"]
                 else "."), ""]  # fmt: skip
    names = {"title": "Title", "top_four": "Top four", "bottom_three": "Bottom three"}
    for line in LINES:
        lines += [f"## {names[line]}", "",
                  "| after | Brier Vesta | uniform | position | ppg | ppg_shrunk |"
                  " ppg_shrunk - Vesta [95%] | log loss Vesta | log loss ppg_shrunk |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for rounds in (*ROUNDS, "all"):
            row = table[(f"brier_{line}", rounds)]
            mean, low, high = row["ppg_shrunk_difference"]
            star = " *" if low > 0 or high < 0 else ""
            loss = table[(f"log_loss_{line}", rounds)]
            lines.append(f"| {rounds if rounds == 'all' else f'{rounds} rounds'} |"
                         f" {row['vesta']:.4f} | {row['uniform']:.4f} | {row['position']:.4f} |"
                         f" {row['ppg']:.4f} | {row['ppg_shrunk']:.4f} | {mean:+.4f}"
                         f" [{low:+.4f}; {high:+.4f}]{star} | {loss['vesta']:.4f} |"
                         f" {loss['ppg_shrunk']:.4f} |")
        lines += ["", f"The reference Brier, all checkpoints: {PREVIOUS[line]} (an earlier"
                  " model and world, shown beside ours, not a bar).", ""]
    lines += ["## The final place: ranked probability score", "",
              "| after | Vesta | uniform | position | ppg | ppg_shrunk |"
              " ppg_shrunk - Vesta [95%] |",
              "|---|---|---|---|---|---|---|"]
    for rounds in (*ROUNDS, "all"):
        row = table[("rps_place", rounds)]
        mean, low, high = row["ppg_shrunk_difference"]
        lines.append(f"| {rounds} | {row['vesta']:.4f} | {row['uniform']:.4f} |"
                     f" {row['position']:.4f} | {row['ppg']:.4f} | {row['ppg_shrunk']:.4f} |"
                     f" {mean:+.4f} [{low:+.4f}; {high:+.4f}] |")
    points = [value for _, rounds, method, measure, value in results["records"]
              if method == "vesta" and measure == "log_score_points"]
    lines += ["", f"The final points: Vesta's mean log score {np.mean(points):.3f} (no baseline"
              " states a points distribution; reported, not judged).", "",
              "## Calibration of Vesta's lines (every checkpoint)", ""]
    bands = (0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0001)
    for line in LINES:
        lines += [f"**{names[line]}**", "", "| band said | n | mean said | happened |",
                  "|---|---|---|---|"]
        pairs = np.array(results["calibration"][line], dtype=float)
        for low, high in zip(bands[:-1], bands[1:], strict=True):
            chosen = pairs[(pairs[:, 0] >= low) & (pairs[:, 0] < high)]
            if len(chosen):
                lines.append(f"| {low:.0%}-{min(high, 1):.0%} | {len(chosen)} |"
                             f" {chosen[:, 0].mean():.1%} | {chosen[:, 1].mean():.1%} |")
        lines.append("")
    return "\n".join(lines)


def main(arguments: list[str]) -> int:
    import hashlib
    import time

    from vesta.research.repository import run_git

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("world", type=Path)
    parser.add_argument("--stage", choices=("development", "blind"), default="development")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--records", type=Path, default=None,
                        help="also write every per-team score as JSON (for the alarm's check)")
    options = parser.parse_args(arguments)
    started = time.monotonic()
    results = run(options.world, options.stage, options.draws)
    digest = hashlib.sha256()
    with options.world.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 24), b""):
            digest.update(chunk)
    text = report(results, options.stage, options.draws, options.world,
                  run_git(ROOT, "rev-parse", "HEAD").strip(), digest.hexdigest(),
                  time.monotonic() - started)  # fmt: skip
    out = options.out or (ROOT / "docs/research"
                          / f"{date.today().isoformat()}-season-{options.stage}.md")
    out.write_text(text + "\n", encoding="utf-8")
    if options.records:
        options.records.write_text(json.dumps(results["records"]), encoding="utf-8")
    print(text.splitlines()[4])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
