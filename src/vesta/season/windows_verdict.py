"""The verdict of the streaks, the next k matches and the windows (T3, T12, T13).

    python -m vesta.season.windows_verdict <football-world.db> [--stage development|blind]
                                           [--draws N] [--out report.md]
                                           [--scope leagues|cups] [--key K ...]

The plan and its rule: the season family's streaks plan (f032c285, its
amendments cbbdf3a9 and f5495b96, approved), the alarm `[alarm.windows]`.

- **The five leagues:** checkpoints after 5, 10, 15, 20, 25 and 30 rounds' worth of matches for
  T12 and T13, after 5, 10, 19 and 29 for T3 (the season simulation's own); every league match
  still to play gets the champion's matrix requested at the checkpoint
  (`vesta.season.league_verdict.league_matrices`), and a league club's UEFA phase matches in its
  30 days the cups model at the checkpoint (`vesta.season.cups_strength`).
- **The UEFA cups' phases:** T12 with k = 3 at the eve of the phase and at its half (the cups
  verdict's C0 and C1), the cups model's matrices.
- **Vesta:** every match drawn from its matrix, `draws` times, one seeded generator per checkpoint.
- **The reference (by design blind to the opponent and the venue: the per-match ppg_shrunk):**
  each of the team's matches a score of two Poissons, its goals for and against per match so far
  in the competition this season, shrunk toward the competition's mean goals per team and match of
  the seasons before by `c` matches (5 in the leagues, 2 in the cups).

Per key, every (team, checkpoint) whose window is whole and whose outcome is not settled is
scored by the log score of what happened (held to [0.0001, 0.9999]). A key passes by the family
rule: the paired block bootstrap interval of Vesta minus the reference below zero (blocks: the
competition-seasons), at least `min_seasons_improved` development seasons improved (every season,
blind), the expected calibration error of its lines within `calibration_tolerance` of the
reference's; the alarm `[alarm.windows]` reads each checkpoint's realised log score against the
forecasts' own entropy. The leagues and the cups are reported apart.
"""

import argparse
import hashlib
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from vesta.forecast_reading import read_line
from vesta.quantities import load_quantities
from vesta.research.bootstrap import bootstrap_mean
from vesta.research.calibration import measure_calibration_error
from vesta.season.cup_progression import read_cup_season
from vesta.season.cups_strength import build_cups_strength, fit_cups_mapping
from vesta.season.cups_verdict import (
    compute_entropy,
    find_checkpoint_instants,
    list_cups,
    score_log_loss,
)
from vesta.season.league_verdict import (
    LEFT_OUT,
    LeagueYear,
    find_checkpoint_instant,
    forecast_checkpoint_matrices,
    read_years,
)
from vesta.season.windows import count_season_streaks, count_shares, count_window, find_first
from vesta.spine.cups import CupsParameters
from vesta.spine.score_matrix import SIZE
from vesta.spine.timing import TimeRules
from vesta.spine.walk_forward import WalkForwardSettings
from vesta.spine.world import open_world, read_club_matches

ROOT = Path(__file__).resolve().parents[3]
T12_ROUNDS = (5, 10, 15, 20, 25, 30)
T3_ROUNDS = (5, 10, 19, 29)
LEAGUE_K = (3, 5)
CUP_K = (3,)
FIRSTS = 10  # the first loss and the first clean sheet among the next ten matches
SHRINK = {"leagues": 5, "cups": 2}
WINDOW_DAYS = 30
METHODS = ("vesta", "reference")
COUNTS = ("points", "wins", "losses", "goals", "clean_sheets")
STREAKS = ("longest_winning_streak", "longest_unbeaten_run", "unbeaten_at_home",
           "loses_three_in_a_row")  # fmt: skip
BINARY = ("unbeaten_at_home", "loses_three_in_a_row")


def count_size(name: str, window: int) -> int:
    """The outcomes of a count of k matches, the last lumping everything above."""
    return {"points": 3 * window + 1, "goals": 4 * window + 1}.get(name, window + 1)


@dataclass(frozen=True)
class Row:
    key: str
    scope: str  # leagues or cups
    code: str
    season: int
    checkpoint: str
    first: bool  # the scope's first checkpoint (the reference is mostly its shrinkage there)
    team: int
    happened: int
    distributions: dict[str, list[float]]

    @property
    def block(self) -> str:
        return f"{self.code}|{self.season}"


# ---- draws ---------------------------------------------------------------------------------


def draw_scores(matrices: np.ndarray, draws: int, generator: np.random.Generator):
    """(draws, n) home and away goals of n matches drawn from their matrices."""
    home = np.zeros((draws, len(matrices)), dtype=np.int64)
    away = np.zeros((draws, len(matrices)), dtype=np.int64)
    for index, matrix in enumerate(matrices):
        cumulative = np.cumsum(matrix.ravel() / matrix.sum())
        cell = np.minimum(np.searchsorted(cumulative, generator.random(draws), side="right"),
                          SIZE * SIZE - 1)  # fmt: skip
        home[:, index], away[:, index] = cell // SIZE, cell % SIZE
    return home, away


def draw_form_scores(goals_for: int, goals_against: int, played: int, mean: float, shrink: int,
                     matches: int, draws: int, generator: np.random.Generator):  # fmt: skip
    """The reference: (draws, matches) goals for and against, each a Poisson of the team's rate
    so far shrunk toward `mean` by `c` matches."""
    rate_for = (goals_for + shrink * mean) / (played + shrink)
    rate_against = (goals_against + shrink * mean) / (played + shrink)
    return (generator.poisson(rate_for, (draws, matches)),
            generator.poisson(rate_against, (draws, matches)))  # fmt: skip


# ---- the leagues ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeamGame:
    index: int  # the match's column among the league's matches to play (-1: played)
    kickoff: str
    home: bool
    goals_for: int
    goals_against: int


def split_team_games(
    year: LeagueYear, team: int, instant: str, remaining: list
) -> tuple[list, list]:
    """The team's played and remaining league games at `instant`, from its side, in order."""
    column = {game.match_id: index for index, game in enumerate(remaining)}
    played, ahead = [], []
    for game in year.games:
        if team not in (game.home, game.away):
            continue
        home = game.home == team
        own = TeamGame(column.get(game.match_id, -1), game.kickoff_utc, home,
                       game.home_goals if home else game.away_goals,
                       game.away_goals if home else game.home_goals)  # fmt: skip
        (played if game.known_at <= instant else ahead).append(own)
    return played, ahead


def average_league_goals(years: dict, competition: str, season: int) -> float:
    """The mean goals per team and match of the competition's seasons before `season`."""
    earlier = [year for (code, year_season), year in years.items()
               if code == competition and year_season < season]  # fmt: skip
    goals = sum(game.home_goals + game.away_goals for year in earlier for game in year.games)
    count = sum(len(year.games) for year in earlier)
    if not count:
        raise ValueError(f"no season of {competition} before {season}")
    return goals / (2 * count)


def run(world_path: Path, stage: str, draws: int, say=print,
        seasons: tuple[int, ...] | None = None,
        scopes: tuple[str, ...] = ("leagues", "cups")) -> dict:  # fmt: skip
    """The rows of every key; `seasons` replaces the stage's only for a smoke test on seasons no
    verdict reads (2017-2018); `scopes` the leagues, the cups or both (a scope left out is not
    read, so a blind run of one scope reads the closed seasons for it alone)."""
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    research = tomllib.loads((ROOT / "config/vesta/research.toml").read_text(encoding="utf-8"))
    expected_goals = tomllib.loads((ROOT / "config/vesta/expected_goals.toml").read_text(
        encoding="utf-8"))  # fmt: skip
    seed = int(research["verdict"]["seed"])
    seasons = seasons or tuple(
        research["seasons"]["development" if stage == "development" else "closed"])
    competitions = list(WalkForwardSettings.from_config(spine).competitions)
    parameters = CupsParameters.from_config(spine)
    with open_world(world_path) as world:
        years = read_years(world, competitions, TimeRules.from_config(spine))
        club_matches = read_club_matches(world, list(parameters.leagues),
                                         list(parameters.cups), parameters.first_league_season,
                                         parameters.first_cup_season, max(seasons))  # fmt: skip
        cups = {(code, season): read_cup_season(world, code, season)
                for season in seasons for code in list_cups(season)}  # fmt: skip
    targets = [year for key, year in sorted(years.items())
               if key[1] in seasons and key not in LEFT_OUT and "leagues" in scopes]  # fmt: skip
    every_round = tuple(sorted(set(T12_ROUNDS) | set(T3_ROUNDS)))
    instants = {(year.competition, year.season, rounds): find_checkpoint_instant(
        year.games, len(year.team_ids), rounds) for year in targets for rounds in every_round}
    say(f"{len(targets)} league-seasons; matrices of {len(every_round)} checkpoints each")
    matrices = forecast_checkpoint_matrices(world_path, targets, instants, spine, expected_goals,
                                            rounds=every_round) if targets else {}  # fmt: skip
    say("league matrices made")
    mappings = {season: fit_cups_mapping(club_matches, parameters, season) for season in seasons}
    rows: list[Row] = []
    for year in targets:
        mean = average_league_goals(years, year.competition, year.season)
        for index, rounds in enumerate(every_round):
            started = time.monotonic()
            instant = instants[(year.competition, year.season, rounds)]
            rows += _compare_league_checkpoint(
                year,
                rounds,
                instant,
                matrices,
                mean,
                cups,
                club_matches,
                parameters,
                mappings[year.season],
                draws,
                seed + 1000 * year.season + 10 * index,
            )
            say(f"{year.competition} {year.season} after {rounds} rounds:"
                f" {time.monotonic() - started:.0f} s")  # fmt: skip
    for (code, season), cup in sorted(cups.items() if "cups" in scopes else ()):
        rows += _compare_cup_phase(cup, club_matches, parameters, mappings[season], draws,
                                   seed + 1000 * season + 500 + ("CL", "EL", "ECL").index(code))
        say(f"{code} {season}: the phase's checkpoints done")
    return {"stage": stage, "seasons": seasons, "rows": rows, "research": research,
            "draws": draws}


def _compare_league_checkpoint(
    year, rounds, instant, matrices, mean, cups, club_matches, parameters, mapping, draws, seed
) -> list[Row]:
    generator = np.random.default_rng(seed)
    remaining = [game for game in year.games if game.known_at > instant]
    drawn = matrices[(year.competition, year.season, rounds)]
    home_goals, away_goals = draw_scores(np.array([drawn[game.match_id] for game in remaining]),
                                         draws, generator)  # fmt: skip
    first = rounds == T12_ROUNDS[0] if rounds in T12_ROUNDS else rounds == T3_ROUNDS[0]
    checkpoint = f"after {rounds}"
    rows = []
    cup_games = _collect_cup_games(cups, year.season, instant)
    cup_strength = (build_cups_strength(club_matches, parameters, mapping, instant)
                    if rounds in T12_ROUNDS else None)  # fmt: skip
    for team in year.team_ids:
        played, ahead = split_team_games(year, team, instant, remaining)
        if not ahead:
            continue
        own_for = np.stack([home_goals[:, g.index] if g.home else away_goals[:, g.index]
                            for g in ahead], axis=1)  # fmt: skip
        own_against = np.stack([away_goals[:, g.index] if g.home else home_goals[:, g.index]
                                for g in ahead], axis=1)  # fmt: skip
        so_far_for = sum(g.goals_for for g in played)
        so_far_against = sum(g.goals_against for g in played)
        ref_for, ref_against = draw_form_scores(so_far_for, so_far_against, len(played), mean,
                                                SHRINK["leagues"], len(ahead), draws, generator)
        true_for = np.array([[g.goals_for for g in ahead]])
        true_against = np.array([[g.goals_against for g in ahead]])
        common = ("leagues", year.competition, year.season, checkpoint, first, team)
        if rounds in T12_ROUNDS:
            rows += _compare_next_matches(
                common, LEAGUE_K, own_for, own_against, ref_for, ref_against, true_for, true_against
            )
            rows += _compare_thirty_days(
                common,
                team,
                instant,
                ahead,
                own_for,
                cup_games,
                cup_strength,
                so_far_for,
                so_far_against,
                len(played),
                mean,
                draws,
                generator,
            )
        if rounds in T3_ROUNDS:
            rows += _compare_streaks(
                common, played, ahead, own_for, own_against, ref_for, ref_against
            )
    return rows


def _compare_next_matches(common, ks, own_for, own_against, ref_for, ref_against, true_for,
                          true_against) -> list[Row]:  # fmt: skip
    """T12: the counts of the next k matches, the first loss and the first clean sheet."""
    rows = []
    scope, code, season, checkpoint, first, team = common
    for window in ks:
        if own_for.shape[1] < window:
            continue  # the window would run past the season's end
        found = {"vesta": count_window(own_for[:, :window], own_against[:, :window]),
                 "reference": count_window(ref_for[:, :window], ref_against[:, :window])}
        happened = count_window(true_for[:, :window], true_against[:, :window])
        for name in COUNTS:
            size = count_size(name, window)
            rows.append(Row(f"T12.{name}.next{window}", scope, code, season, checkpoint, first,
                            team,
                            int(min(happened[name][0], size - 1)),
                            {method: count_shares(values[name], size)
                             for method, values in found.items()}))  # fmt: skip
    count = min(FIRSTS, own_for.shape[1])
    for name, own, ref, true in (
            ("next_loss", own_for[:, :count] < own_against[:, :count],
             ref_for[:, :count] < ref_against[:, :count],
             true_for[:, :count] < true_against[:, :count]),
            ("next_clean_sheet", own_against[:, :count] == 0, ref_against[:, :count] == 0,
             true_against[:, :count] == 0)):  # fmt: skip
        size = FIRSTS + 1
        rows.append(
            Row(
                f"T12.{name}",
                scope,
                code,
                season,
                checkpoint,
                first,
                team,
                _cap_first_index(int(find_first(true)[0]), count),
                {
                    "vesta": count_shares(_cap_firsts(find_first(own), count), size),
                    "reference": count_shares(_cap_firsts(find_first(ref), count), size),
                },
            )
        )
    return rows


def _cap_firsts(values: np.ndarray, count: int) -> np.ndarray:
    """A first among `count` matches as an outcome of the key ("1" .. "10", "none" the last)."""
    return np.where(values >= count, FIRSTS, values)


def _cap_first_index(value: int, count: int) -> int:
    return FIRSTS if value >= count else value


def _compare_streaks(
    common, played, ahead, own_for, own_against, ref_for, ref_against
) -> list[Row]:
    """T3: the season's streaks, the run at the checkpoint carried into the draws."""
    scope, code, season, checkpoint, first, team = common
    played_for = np.array([g.goals_for for g in played], dtype=np.int64)
    played_against = np.array([g.goals_against for g in played], dtype=np.int64)
    played_home = np.array([g.home for g in played], dtype=bool)
    home = np.array([[g.home for g in ahead]], dtype=bool)
    found = {
        "vesta": count_season_streaks(played_for, played_against, played_home, own_for, own_against,
                                      np.broadcast_to(home, own_for.shape)),
        "reference": count_season_streaks(played_for, played_against, played_home, ref_for,
                                          ref_against, np.broadcast_to(home, ref_for.shape)),
    }  # fmt: skip
    true = count_season_streaks(played_for, played_against, played_home,
                                np.array([[g.goals_for for g in ahead]]),
                                np.array([[g.goals_against for g in ahead]]), home)  # fmt: skip
    rows = []
    games = len(played) + len(ahead)
    for name in STREAKS:
        if name in BINARY:
            settled = {method: values[name].min() == values[name].max()
                       for method, values in found.items()}  # fmt: skip
            if all(settled.values()) and found["vesta"][name][0] == found["reference"][name][0]:
                continue  # already settled by the matches played
            rows.append(Row(f"T3.{name}", scope, code, season, checkpoint, first, team,
                            int(true[name][0]),
                            {method: [1 - float(values[name].mean()), float(values[name].mean())]
                             for method, values in found.items()}))  # fmt: skip
        else:
            rows.append(Row(f"T3.{name}", scope, code, season, checkpoint, first, team,
                            int(true[name][0]),
                            {method: count_shares(values[name], games + 1)
                             for method, values in found.items()}))  # fmt: skip
    return rows


def _collect_cup_games(cups: dict, season: int, instant: str) -> dict[int, list]:
    """Per club, its UEFA phase matches of `season` kicking off after `instant`."""
    games: dict[int, list] = {}
    for (_, cup_season), cup in cups.items():
        if cup_season != season:
            continue
        for match in cup.phase:
            if match.kickoff > instant and match.played:
                for team in (match.home, match.away):
                    games.setdefault(team, []).append(match)
    return games


def _compare_thirty_days(common, team, instant, ahead, own_for, cup_games, cup_strength, so_far_for,
                         so_far_against, played, mean, draws, generator) -> list[Row]:  # fmt: skip
    """T13: the team's goals in its league and UEFA phase matches of the next 30 days."""
    scope, code, season, checkpoint, first, _ = common
    end = (datetime.fromisoformat(instant.replace("Z", "+00:00"))
           + timedelta(days=WINDOW_DAYS)).isoformat().replace("+00:00", "Z")  # fmt: skip
    league = [column for column, game in enumerate(ahead) if game.kickoff <= end]
    cup = [match for match in cup_games.get(team, []) if match.kickoff <= end]
    if not league and not cup:
        return []
    vesta = own_for[:, league].sum(axis=1) if league else np.zeros(draws, dtype=np.int64)
    truth = sum(ahead[column].goals_for for column in league)
    for match in cup:
        matrix = cup_strength(match.home, match.away, match.neutral)[0]
        home, away = draw_scores(np.array([matrix]), draws, generator)
        vesta = vesta + (home[:, 0] if match.home == team else away[:, 0])
        truth += match.home_goals if match.home == team else match.away_goals
    reference, _ = draw_form_scores(so_far_for, so_far_against, played, mean, SHRINK["leagues"],
                                    len(league) + len(cup), draws, generator)  # fmt: skip
    size = 4 * (len(league) + len(cup)) + 1
    return [
        Row(
            "T13.goals.next_30_days",
            scope,
            code,
            season,
            checkpoint,
            first,
            team,
            int(min(truth, size - 1)),
            {
                "vesta": count_shares(vesta, size),
                "reference": count_shares(reference.sum(axis=1), size),
            },
        )
    ]


# ---- the cups' phases ----------------------------------------------------------------------


def average_cup_goals(club_matches, season: int) -> float:
    """The mean goals per team and match of the UEFA cups' main stages before `season`."""
    chosen = [match for match in club_matches
              if match.competition in ("CL", "EL", "ECL") and match.season < season
              and not match.qualifying]  # fmt: skip
    return sum(match.home_goals + match.away_goals for match in chosen) / (2 * len(chosen))


def _compare_cup_phase(cup, club_matches, parameters, mapping, draws, seed) -> list[Row]:
    generator = np.random.default_rng(seed)
    mean = average_cup_goals(club_matches, cup.season)
    instants = find_checkpoint_instants(cup)
    rows = []
    for checkpoint in ("C0", "C1"):
        instant = instants[checkpoint]
        strength = build_cups_strength(club_matches, parameters, mapping, instant)
        ahead = [match for match in cup.phase if match.kickoff >= instant]
        if not ahead:
            continue
        home, away = draw_scores(np.array([strength(m.home, m.away, m.neutral)[0] for m in ahead]),
                                 draws, generator)  # fmt: skip
        teams = sorted({team for match in cup.phase for team in (match.home, match.away)})
        for team in teams:
            own = [index for index, match in enumerate(ahead) if team in (match.home, match.away)]
            done = [match for match in cup.phase
                    if match.kickoff < instant and team in (match.home, match.away)]  # fmt: skip
            if not own:
                continue
            sides = [ahead[index].home == team for index in own]
            own_for = np.stack([home[:, i] if side else away[:, i]
                                for i, side in zip(own, sides, strict=True)], axis=1)
            own_against = np.stack([away[:, i] if side else home[:, i]
                                    for i, side in zip(own, sides, strict=True)], axis=1)
            so_far_for = sum(m.home_goals if m.home == team else m.away_goals for m in done)
            so_far_against = sum(m.away_goals if m.home == team else m.home_goals for m in done)
            ref_for, ref_against = draw_form_scores(so_far_for, so_far_against, len(done), mean,
                                                    SHRINK["cups"], len(own), draws, generator)
            true_for = np.array([[ahead[i].home_goals if side else ahead[i].away_goals
                                  for i, side in zip(own, sides, strict=True)]])
            true_against = np.array([[ahead[i].away_goals if side else ahead[i].home_goals
                                      for i, side in zip(own, sides, strict=True)]])
            common = ("cups", cup.code, cup.season, checkpoint, checkpoint == "C0", team)
            rows += _compare_next_matches(
                common, CUP_K, own_for, own_against, ref_for, ref_against, true_for, true_against
            )
    return rows


# ---- the verdict ---------------------------------------------------------------------------


def _format_values(kind: str, shares_list: list[float]):
    if kind == "count":
        return {"probabilities": list(shares_list), "tail_from": len(shares_list), "tail": 0.0}
    if kind == "binary":
        return {"yes": shares_list[1]}
    names = (*(str(index + 1) for index in range(FIRSTS)), "none")
    return [[name, share] for name, share in zip(names, shares_list, strict=True)]


def _collect_line_pairs(row: Row, method: str, definitions) -> list[tuple[float, bool]]:
    quantity = definitions[row.key].quantity
    kind = quantity.distribution_kind
    said = row.distributions[method]
    point = [1.0 if index == row.happened else 0.0 for index in range(len(said))]
    pairs = []
    for line in quantity.lines:
        probability = read_line(kind, _format_values(kind, said), line)
        happened = read_line(kind, _format_values(kind, point), line)
        if kind == "count" and line.startswith("over=") and float(line[5:]) >= len(said) - 1:
            continue  # beyond what k matches can hold
        pairs.append((probability, bool(round(happened))))
    return pairs


def judge(results: dict) -> dict:
    research = results["research"]
    verdict = research["verdict"]
    alarm = research["alarm"]["windows"]
    needed = (int(verdict["min_seasons_improved"]) if results["stage"] == "development"
              else len(results["seasons"]))  # fmt: skip
    repetitions, seed = int(verdict["bootstrap_repetitions"]), int(verdict["seed"])
    tolerance = float(verdict["calibration_tolerance"])
    definitions = load_quantities()
    judged = {}
    for key in sorted({row.key for row in results["rows"]}):
        for scope in ("leagues", "cups"):
            rows = [row for row in results["rows"] if row.key == key and row.scope == scope]
            if not rows:
                continue
            scores = {method: [score_log_loss(row.distributions[method], row.happened)
                               for row in rows] for method in METHODS}  # fmt: skip
            difference = [own - other for own, other
                          in zip(scores["vesta"], scores["reference"], strict=True)]  # fmt: skip
            blocks = [row.block for row in rows]
            mean, low, high = bootstrap_mean(difference, blocks, repetitions, seed)
            by_season: dict[int, float] = {}
            for value, row in zip(difference, rows, strict=True):
                by_season[row.season] = by_season.get(row.season, 0.0) + value
            improved = sum(1 for total in by_season.values() if total < 0)
            calibration = {method: measure_calibration_error(
                [pair for row in rows for pair in _collect_line_pairs(row, method, definitions)])
                for method in METHODS}  # fmt: skip
            split = {}
            for label, wanted in (("first", True), ("later", False)):
                chosen = [value for value, row in zip(difference, rows, strict=True)
                          if row.first == wanted]  # fmt: skip
                if chosen:
                    split[label] = (len(chosen), float(np.mean(chosen)))
            per_checkpoint, warnings, stops = {}, [], []
            for checkpoint in sorted({row.checkpoint for row in rows}, key=_order):
                chosen = [index for index, row in enumerate(rows) if row.checkpoint == checkpoint]
                realised = [scores["vesta"][index] for index in chosen]
                expected = [compute_entropy(rows[index].distributions["vesta"]) for index in chosen]
                gap, gap_low, gap_high = bootstrap_mean(
                    [own - said for own, said in zip(realised, expected, strict=True)],
                    [blocks[index] for index in chosen], repetitions, seed)  # fmt: skip
                error = (gap_high - gap_low) / (2 * 1.959964) or float("inf")
                in_errors = -gap / error
                if in_errors > float(alarm["stop_standard_errors"]):
                    stops.append(f"{key} ({scope}) {checkpoint}: {in_errors:.1f} standard errors"
                                 " below the forecasts' entropy")  # fmt: skip
                elif in_errors > float(alarm["warn_standard_errors"]):
                    warnings.append(f"{key} ({scope}) {checkpoint}: {in_errors:.1f} standard"
                                    " errors below the forecasts' entropy (warning)")  # fmt: skip
                per_checkpoint[checkpoint] = (len(chosen), float(np.mean(realised)),
                                              float(np.mean([scores["reference"][index]
                                                             for index in chosen])),
                                              float(np.mean(expected)), in_errors)  # fmt: skip
            passed = (high < 0 and improved >= needed and not stops
                      and calibration["vesta"] <= calibration["reference"] + tolerance)  # fmt: skip
            judged[(key, scope)] = {
                "n": len(rows), "blocks": len(set(blocks)),
                "scores": {method: float(np.mean(values)) for method, values in scores.items()},
                "difference": (mean, low, high), "improved": improved,
                "seasons": len(by_season), "needed": needed, "calibration": calibration,
                "split": split, "per_checkpoint": per_checkpoint, "warnings": warnings,
                "stops": stops, "passed": passed,
            }
    return judged


def _order(checkpoint: str) -> tuple:
    return (0, int(checkpoint.split()[1])) if checkpoint.startswith("after") else (1, checkpoint)


def report(results: dict, judged: dict, world_path: Path, world_sha: str, commit: str,
           seconds: float) -> str:  # fmt: skip
    stage = results["stage"]
    lines = [f"# Streaks, the next k matches and windows: the {stage} verdict (T3, T12, T13)", "",
             f"World: `{world_path}` (sha256 {world_sha}). Code: `{commit}`. Seasons:"
             f" {', '.join(map(str, results['seasons']))}; {results['draws']} draws per"
             " checkpoint. Written by `python -m vesta.season.windows_verdict`; the plan"
             " declared before the run (f032c285, amendments"
             f" cbbdf3a9 and f5495b96), the alarm `[alarm.windows]`. {seconds:.0f} seconds.", "",
             "The reference ignores the opponent and the venue by design (the per-match"
             " ppg_shrunk). T13 is judged for the five leagues' clubs only: a club outside them"
             " has its domestic matches outside the world.", "",
             "| key | scope | n | blocks | Vesta | reference | Vesta - ref [95%] | seasons |"
             " ECE Vesta | ECE ref | first checkpoint | later | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for (key, scope), result in judged.items():
        mean, low, high = result["difference"]
        split = result["split"]
        first = f"{split['first'][1]:+.4f} ({split['first'][0]})" if "first" in split else "-"
        later = f"{split['later'][1]:+.4f} ({split['later'][0]})" if "later" in split else "-"
        lines.append(f"| {key} | {scope} | {result['n']} | {result['blocks']} |"
                     f" {result['scores']['vesta']:.4f} | {result['scores']['reference']:.4f} |"
                     f" {mean:+.4f} [{low:+.4f}; {high:+.4f}] | {result['improved']}/"
                     f"{result['seasons']} (needed {result['needed']}) |"
                     f" {result['calibration']['vesta']:.4f} |"
                     f" {result['calibration']['reference']:.4f} | {first} | {later} |"
                     f" {'PASSED' if result['passed'] else 'NOT PASSED'} |")  # fmt: skip
    lines += ["", "## The alarm and each checkpoint", "",
              "| key | scope | checkpoint | n | Vesta | reference | entropy | gap (SE) |",
              "|---|---|---|---|---|---|---|---|"]
    for (key, scope), result in judged.items():
        for checkpoint, (count, own, other, said, errors) in result["per_checkpoint"].items():
            lines.append(f"| {key} | {scope} | {checkpoint} | {count} | {own:.4f} | {other:.4f} |"
                         f" {said:.4f} | {errors:+.2f} |")
    alarms = [stop for result in judged.values() for stop in result["stops"]]
    warnings = [warning for result in judged.values() for warning in result["warnings"]]
    lines += ["", "## Alarms and warnings", ""]
    lines += [f"- ALARM: {stop}" for stop in alarms] + [f"- {warning}" for warning in warnings]
    if not alarms and not warnings:
        lines.append("- none")
    return "\n".join(lines)


def main(arguments: list[str]) -> int:
    from vesta.research.repository import run_git

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("world", type=Path)
    parser.add_argument("--stage", choices=("development", "blind"), default="development")
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--scope", choices=("leagues", "cups"), default=None,
                        help="one scope only (default both)")
    parser.add_argument("--key", action="append", default=None,
                        help="judge only these keys (default all)")
    options = parser.parse_args(arguments)
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    draws = options.draws or int(spine["season"]["simulations"])
    started = time.monotonic()
    commit = run_git(ROOT, "rev-parse", "HEAD").strip()  # the code the run starts from
    scopes = (options.scope,) if options.scope else ("leagues", "cups")
    results = run(options.world, options.stage, draws, scopes=scopes)
    if options.key:
        results["rows"] = [row for row in results["rows"] if row.key in options.key]
    judged = judge(results)
    digest = hashlib.sha256()
    with options.world.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 24), b""):
            digest.update(chunk)
    text = report(results, judged, options.world, digest.hexdigest(), commit,
                  time.monotonic() - started)  # fmt: skip
    out = options.out or (ROOT / "docs/research"
                          / f"{date.today().isoformat()}-season-family-windows-{options.stage}.md")
    out.write_text(text + "\n", encoding="utf-8")
    for (key, scope), result in judged.items():
        print(key, scope, "PASSED" if result["passed"] else "NOT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
