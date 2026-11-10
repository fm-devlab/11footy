"""The UEFA cups' progression verdict (T4.round_reached, T4.phase_place).

    python -m vesta.season.cups_verdict <football-world.db> [--stage development|blind]
                                        [--draws N] [--out report.md] [--records records.json]

The plan and its rule: the season family's cups plan (approved with five conditions), the alarm
`[alarm.progression]`. For every cup
of the stage's seasons (the Champions League and the Europa League; the Conference League from
2021-22) and every checkpoint (C0 the eve of the phase, C1 half its matchdays played, C2 the phase
over, C3 after the round of 16, C4 after the quarter-finals; each the end of the UTC day of the
step's last match), the cup is simulated from what was known then (`vesta.season.cup_progression`)
three times on the same seed:

- **Vesta:** the cups model's strength at the checkpoint;
- **the reference (deciding):** equal strength, the mean score of the cups' earlier seasons;
- **frozen C0 (information):** the cups model's strength at the cup's C0, the in-season matches
  left out.

Every club whose outcome the facts have not settled is scored: the log score of what happened
(probabilities held to [0.0001, 0.9999]), the ranked probability score beside it. A key passes
when the paired block bootstrap interval of Vesta minus the reference (blocks: the
competition-seasons) lies below zero, Vesta is better in at least `min_seasons_improved` of the
development seasons (every season, blind), and the expected calibration error of its lines is no
worse than the reference's by more than `calibration_tolerance`. The alarm compares, at each
checkpoint, the realised mean log score with the mean entropy of Vesta's own forecasts.
"""

import argparse
import hashlib
import json
import math
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from vesta.forecast_reading import read_line
from vesta.quantities import load_quantities
from vesta.research.bootstrap import bootstrap_mean
from vesta.research.calibration import measure_calibration_error
from vesta.season import cup_qualifiers
from vesta.season.cup_progression import (
    CupSeason,
    build_phase_tables,
    find_clubs_still_in,
    find_known_knockout,
    find_outcomes_reached,
    find_tie_winner_disagreements,
    is_phase_over,
    rank_final_phase,
    read_cup_season,
    simulate_cup,
)
from vesta.season.cup_qualifiers import draw_dates, read_qualifying_season
from vesta.season.cups_strength import (
    build_cups_strength,
    build_equal_strength,
    compute_mean_score,
    fit_cups_mapping,
)
from vesta.season.league import place_bounds
from vesta.spine.cups import CupsParameters
from vesta.spine.ties import TieParameters, read_tie_rule
from vesta.spine.world import open_world, read_club_matches

ROOT = Path(__file__).resolve().parents[3]
ROUND, PLACE, QUALIFIES = "T4.round_reached", "T4.phase_place", "T4.qualifies.team"
CHECKPOINTS = ("C0", "C1", "C2", "C3", "C4", "Q3", "PO")
PLACE_CHECKPOINTS = ("C0", "C1")
METHODS = ("vesta", "reference", "frozen")
FLOOR = 1e-4


def list_cups(season: int) -> tuple[str, ...]:
    return ("CL", "EL", "ECL") if season >= 2021 else ("CL", "EL")


# ---- checkpoints ---------------------------------------------------------------------------


def find_checkpoint_instants(cup: CupSeason) -> dict[str, str]:
    """The instants of the checkpoints a cup-season has (a round it never played has none)."""
    phase_days = sorted({match.kickoff[:10] for match in cup.phase})
    played: dict[int, list[str]] = {}  # the world holds no matchday: each club's own matches
    for match in cup.phase:
        for team in (match.home, match.away):
            played.setdefault(team, []).append(match.kickoff[:10])
    counts = {len(days) for days in played.values()}
    if len(counts) != 1 or counts.pop() not in (6, 8):
        raise ValueError(f"{cup.code} {cup.season}: the clubs do not all play 6 or 8 phase matches")
    half_played = max(sorted(days)[len(days) // 2 - 1] for days in played.values())
    instants = {"C0": f"{phase_days[0]}T00:00:00Z",
                "C1": _add_one_day(half_played),  # every club has played half its matches
                "C2": _add_one_day(phase_days[-1])}  # fmt: skip
    for checkpoint, name in (("C3", "round_of_16"), ("C4", "quarter_finals")):
        days = [match.kickoff[:10] for match in cup.knockout if match.round_name == name]
        if days:
            instants[checkpoint] = _add_one_day(max(days))
    return instants


def _add_one_day(day: str) -> str:
    return f"{(date.fromisoformat(day) + timedelta(days=1)).isoformat()}T00:00:00Z"


# ---- one run -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    key: str
    code: str
    season: int
    checkpoint: str
    team: int
    happened: int
    distributions: dict[str, list[float]]  # method -> the shares of each outcome

    @property
    def block(self) -> str:
        return f"{self.code}|{self.season}"


def run(world_path: Path, stage: str, draws: int, say=print,
        seasons: tuple[int, ...] | None = None, codes: tuple[str, ...] | None = None) -> dict:
    """The rows of the main stage's keys; `seasons` and `codes` replace the stage's only for the
    warm-up seasons candidate 2's temperature is fitted on (CL 2016-2018), never judged."""
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    research = tomllib.loads((ROOT / "config/vesta/research.toml").read_text(encoding="utf-8"))
    seasons = seasons or research["seasons"]["development" if stage == "development" else "closed"]
    competitions = (lambda _season: codes) if codes else list_cups
    parameters = CupsParameters.from_config(spine)
    ties = TieParameters.from_config(spine)
    seed = int(research["verdict"]["seed"])
    rows: list[Row] = []
    notes: list[str] = []
    refused: list[str] = []
    with open_world(world_path) as world:
        matches = read_club_matches(world, list(parameters.leagues), list(parameters.cups),
                                    parameters.first_league_season, parameters.first_cup_season,
                                    max(seasons))  # fmt: skip
        cups = {(code, season): read_cup_season(world, code, season)
                for season in seasons for code in competitions(season)}  # fmt: skip
    for season in seasons:
        mapping = fit_cups_mapping(matches, parameters, season)
        reference = build_equal_strength(compute_mean_score(matches, season))
        for code in competitions(season):
            cup = cups[(code, season)]
            if not cup.table_rule.publishable:
                refused.append(f"{code} {season}: its tie-break rule is not verified - can't pass"
                               " yet")  # fmt: skip
                continue
            source = (cups[(cup.form.drop_in.source, season)]
                      if cup.form.drop_in is not None else None)  # fmt: skip
            order = rank_final_phase(cup, seed)
            drop_ins = ()
            if source is not None:
                source_order = rank_final_phase(source, seed)
                drop_ins = tuple(source_order[group][cup.form.drop_in.place - 1]
                                 for group in sorted(source_order))  # fmt: skip
            happened = find_outcomes_reached(cup, order, drop_ins)
            notes += [f"{code} {season} {name}: the world's tie {tie} names {world_winner}, the"
                      f" goals {ours} (who plays on)"
                      for name, tie, world_winner, ours in find_tie_winner_disagreements(cup)]
            instants = find_checkpoint_instants(cup)
            frozen = build_cups_strength(matches, parameters, mapping, instants["C0"])
            for index, (checkpoint, instant) in enumerate(instants.items()):
                started = time.monotonic()
                model = build_cups_strength(matches, parameters, mapping, instant)
                run_seed = seed + 1000 * season + 100 * ("CL", "EL", "ECL").index(code) + index
                simulated = {method: simulate_cup(cup, source, instant, strength, ties, draws,
                                                  run_seed)
                             for method, strength in (("vesta", model), ("reference", reference),
                                                      ("frozen", frozen))}  # fmt: skip
                _assert_known_before(cup, source, instant)
                tables = build_phase_tables(cup, instant, reference)
                final_order = order if is_phase_over(tables) else None
                state = find_known_knockout(cup, instant, final_order)
                notes += [f"{code} {season} {checkpoint}: a tie awarded without a match, {winner}"
                          f" through against {loser}" for winner, loser in state.awarded]
                alive = find_clubs_still_in(cup, tables, state)
                if state.last_round is None and source is not None and is_phase_over(
                        build_phase_tables(source, instant, reference)):  # fmt: skip
                    alive |= set(drop_ins)
                outcomes = len(cup.form.outcomes)
                for team in sorted(alive):
                    rows.append(Row(ROUND, code, season, checkpoint, team, happened[team],
                                    {method: _count_shares(result.reached[team], outcomes)
                                     for method, result in simulated.items()}))  # fmt: skip
                if checkpoint in PLACE_CHECKPOINTS:
                    final_place = {team: place + 1 for teams in order.values()
                                   for place, team in enumerate(teams)}  # fmt: skip
                    for table in tables.values():
                        best, worst = place_bounds(table)
                        size = len(table.team_ids)
                        for column, team in enumerate(table.team_ids):
                            if best[column] == worst[column]:
                                continue
                            rows.append(Row(PLACE, code, season, checkpoint, team,
                                            final_place[team] - 1,
                                            {method: _count_shares(result.places[team] - 1, size)
                                             for method, result in simulated.items()}))  # fmt: skip
                say(f"{code} {season} {checkpoint} {instant}: {len(alive)} clubs in,"
                    f" {time.monotonic() - started:.0f} s")  # fmt: skip
    return {"stage": stage, "seasons": seasons, "rows": rows, "notes": notes, "refused": refused,
            "research": research, "draws": draws}


def run_qualifiers(world_path: Path, stage: str, draws: int, say=print) -> dict:
    """`T4.qualifies.team` at the eves of each cup's third qualifying round and play-off round
    whose path the calendar shows drawn by then (`vesta.season.cup_qualifiers`). 2020-21 is left
    out: its rounds were single matches the world does not hold as ties."""
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    research = tomllib.loads((ROOT / "config/vesta/research.toml").read_text(encoding="utf-8"))
    seasons = [season for season in
               research["seasons"]["development" if stage == "development" else "closed"]
               if season != 2020]  # fmt: skip
    parameters = CupsParameters.from_config(spine)
    ties = TieParameters.from_config(spine)
    seed = int(research["verdict"]["seed"])
    dates = draw_dates()
    rows: list[Row] = []
    notes: list[str] = []
    with open_world(world_path) as world:
        matches = read_club_matches(world, list(parameters.leagues), list(parameters.cups),
                                    parameters.first_league_season, parameters.first_cup_season,
                                    max(seasons))  # fmt: skip
        qualifying = {season: read_qualifying_season(world, season) for season in seasons}
    for season in seasons:
        mapping = fit_cups_mapping(matches, parameters, season)
        reference = build_equal_strength(compute_mean_score(matches, season, qualifying=True))
        rules = {cup: read_tie_rule(f"{cup}Q" if season >= 2020 else cup, season)
                 for cup in list_cups(season)}  # fmt: skip
        state = qualifying[season]
        for code in list_cups(season):
            instants = cup_qualifiers.find_checkpoints(state, code, dates)
            for name in ("Q3", "PO"):
                if name not in instants:
                    notes.append(f"{code} {season} {name}: not taken - the calendar does not show"
                                 " its path drawn before its first match")  # fmt: skip
            if not instants:
                continue
            frozen = build_cups_strength(
                matches, parameters, mapping, next(iter(instants.values())), qualifying=True
            )
            for index, (name, instant) in enumerate(instants.items()):
                started = time.monotonic()
                linked = cup_qualifiers.link_ties(
                    cup_qualifiers.find_known_ties(state, instant, dates)
                )
                clubs = cup_qualifiers.find_subject_clubs(linked, code, name)
                model = build_cups_strength(matches, parameters, mapping, instant, qualifying=True)
                run_seed = seed + 1000 * season + 100 * ("CL", "EL", "ECL").index(code) + 50 + index
                simulated = {method: cup_qualifiers.simulate_qualifiers(
                    linked, instant, strength, rules, ties, draws, run_seed)
                    for method, strength in (("vesta", model), ("reference", reference),
                                             ("frozen", frozen))}  # fmt: skip
                for team in sorted(clubs):
                    happened = int(team in state.main_stage.get(code, frozenset()))
                    distributions = {}
                    for method, through in simulated.items():
                        reached = through.get((team, code))
                        share = float(reached.mean()) if reached is not None else 0.0
                        distributions[method] = [1.0 - share, share]
                    rows.append(Row(QUALIFIES, code, season, name, team, happened, distributions))
                say(f"{code} {season} {name} {instant}: {len(clubs)} clubs,"
                    f" {time.monotonic() - started:.0f} s")  # fmt: skip
    return {"stage": stage, "seasons": seasons, "rows": rows, "notes": notes, "refused": [],
            "research": research, "draws": draws}


# ---- candidate 2: a temperature per stage ---------------------------------------------------

STAGES = {"phase": ("C0", "C1"), "knockout": ("C2", "C3", "C4")}
TAU_GRID = tuple(round(0.50 + 0.05 * step, 2) for step in range(31))  # 0.50 to 2.00
WARM_UP = (2016, 2017, 2018)


def temper_shares(shares: list[float], tau: float) -> list[float]:
    """p_i^tau / sum_j p_j^tau: tau > 1 sharpens, tau < 1 flattens; an impossible outcome stays
    impossible."""
    powered = np.array([share ** tau if share > 0 else 0.0 for share in shares])
    return (powered / powered.sum()).tolist()


def fit_tau(rows: list["Row"]) -> float:
    """The grid's tau with the lowest mean log score of Vesta's forecasts on `rows`."""
    if not rows:
        return 1.0
    scores = {
        tau: math.fsum(
            score_log_loss(temper_shares(row.distributions["vesta"], tau), row.happened)
            for row in rows
        )
        for tau in TAU_GRID
    }
    return min(TAU_GRID, key=lambda tau: (scores[tau], abs(tau - 1.0)))


def apply_temperature(rows: list["Row"], earlier: list["Row"]) -> tuple[list["Row"], dict]:
    """Candidate 2: every row's Vesta forecast tempered by the tau of its key and stage, fitted on
    the rows of the seasons before its own (in `earlier` and `rows`); candidate 1's forecast kept
    as "candidate_1". Returns the rows and the taus by (season, key, stage)."""
    pool = earlier + rows
    taus: dict[tuple[int, str, str], float] = {}
    tempered_rows = []
    for row in rows:
        stage = next(name for name, checkpoints in STAGES.items() if row.checkpoint in checkpoints)
        key = (row.season, row.key, stage)
        if key not in taus:
            taus[key] = fit_tau([other for other in pool if other.key == row.key
                                 and other.season < row.season
                                 and other.checkpoint in STAGES[stage]])  # fmt: skip
        distributions = dict(row.distributions)
        distributions["candidate_1"] = row.distributions["vesta"]
        distributions["vesta"] = temper_shares(row.distributions["vesta"], taus[key])
        tempered_rows.append(Row(row.key, row.code, row.season, row.checkpoint, row.team,
                                 row.happened, distributions))  # fmt: skip
    return tempered_rows, taus


def bin_calibration_bands(rows: list["Row"],
                          method: str) -> list[tuple[float, float, int, float, float]]:
    """(low, high, n, said, happened) of the key's lines, by band of the said probability."""
    definitions = load_quantities()
    bands = (0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95, 1.0001)
    pairs = [pair for row in rows for pair in _collect_line_pairs(row, method, definitions)]
    found = []
    for low, high in zip(bands[:-1], bands[1:], strict=True):
        chosen = [(said, happened) for said, happened in pairs if low <= said < high]
        if chosen:
            found.append((low, min(high, 1.0), len(chosen),
                          math.fsum(said for said, _ in chosen) / len(chosen),
                          sum(happened for _, happened in chosen) / len(chosen)))  # fmt: skip
    return found


def _count_shares(values: np.ndarray, count: int) -> list[float]:
    return (np.bincount(values, minlength=count)[:count] / len(values)).tolist()


def _assert_known_before(cup: CupSeason, source: CupSeason | None, instant: str) -> None:
    """The run's own check: every result a state reads kicked off before the checkpoint (the
    filters of `vesta.season.cup_progression` say so; this asserts it on the data)."""
    for season in (cup, source):
        if season is None:
            continue
        tables = build_phase_tables(season, instant, lambda *_: (np.ones((11, 11)), 1, 1))
        kickoffs = {(match.home, match.away): match.kickoff for match in season.phase}
        for table in tables.values():
            for home, away, *_ in table.played:
                if kickoffs[(home, away)] >= instant:
                    raise AssertionError(f"{season.code} {season.season}: a result after {instant}")


# ---- scores and the verdict ----------------------------------------------------------------


def score_log_loss(shares: list[float], happened: int) -> float:
    return -math.log(min(max(shares[happened], FLOOR), 1 - FLOOR))


def compute_entropy(shares: list[float]) -> float:
    return -math.fsum(share * math.log(min(max(share, FLOOR), 1 - FLOOR))
                      for share in shares if share > 0)  # fmt: skip


def score_ranked_probability(shares: list[float], happened: int) -> float:
    cumulative = np.cumsum(shares)
    observed = (np.arange(len(shares)) >= happened).astype(float)
    return float(((cumulative - observed)[:-1] ** 2).sum() / max(len(shares) - 1, 1))


def _collect_line_pairs(row: Row, method: str, definitions) -> list[tuple[float, bool]]:
    """The key's lines a club's table or format can reach, said and happened."""
    kind = definitions[row.key].quantity.distribution_kind
    shares = row.distributions[method]
    if kind == "binary":
        return [(shares[1], row.happened == 1)]
    size = len(shares)
    pairs = []
    for line in definitions[row.key].quantity.lines:
        if kind == "rank" and line.startswith("top=") and int(line.split("=")[1]) >= size:
            continue  # a group of four: its top 8 holds every club
        if kind == "rank" and line.startswith("place=") and int(line.split("=")[1]) > size:
            continue
        if kind != "rank" and not _is_in_format(row, line):
            continue  # a format without a round of 16: the line repeats the next one
        said = read_line(kind, _format_values(row, shares, kind), line)
        point = [1.0 if index == row.happened else 0.0 for index in range(size)]
        pairs.append((said, bool(round(read_line(kind, _format_values(row, point, kind), line)))))
    return pairs


def _is_in_format(row: Row, line: str) -> bool:
    """Whether every outcome a line of named outcomes names is one of the row's format."""
    from vesta.season.tournament import read_tournament_format

    outcomes = read_tournament_format(row.code, row.season).outcomes
    return all(name in outcomes for name in line.split("=", 1)[1].split(","))


def _format_values(row: Row, shares: list[float], kind: str):
    if kind == "rank":
        return list(shares)
    from vesta.season.tournament import read_tournament_format

    outcomes = read_tournament_format(row.code, row.season).outcomes
    return [[name, share] for name, share in zip(outcomes, shares, strict=True)]


def _order_checkpoint(checkpoint: str) -> tuple:
    return (CHECKPOINTS.index(checkpoint) if checkpoint in CHECKPOINTS else len(CHECKPOINTS),
            checkpoint)  # fmt: skip


def judge(results: dict, keys: tuple[str, ...] = (ROUND, PLACE, QUALIFIES),
          alarm_name: str = "progression", information: str = "frozen",
          groups: tuple[str, ...] = ("CL", "EL", "ECL")) -> dict:  # fmt: skip
    """The verdict of each key; `information` the method reported beside the reference (its
    results under "frozen"), `groups` the competitions reported apart."""
    research = results["research"]
    verdict = research["verdict"]
    alarm = research["alarm"][alarm_name]
    methods = ("vesta", "reference", information)
    needed = (int(verdict["min_seasons_improved"]) if results["stage"] == "development"
              else len(results["seasons"]))  # fmt: skip
    definitions = load_quantities()
    judged = {}
    for key in keys:
        rows = [row for row in results["rows"] if row.key == key]
        if not rows:
            continue
        scores = {method: [score_log_loss(row.distributions[method], row.happened) for row in rows]
                  for method in methods}  # fmt: skip
        difference = [own - other for own, other in zip(scores["vesta"], scores["reference"],
                                                        strict=True)]  # fmt: skip
        frozen = [own - other for own, other in zip(scores["vesta"], scores[information],
                                                     strict=True)]  # fmt: skip
        blocks = [row.block for row in rows]
        repetitions, seed = int(verdict["bootstrap_repetitions"]), int(verdict["seed"])
        mean, low, high = bootstrap_mean(difference, blocks, repetitions, seed)
        by_season: dict[int, float] = {}
        for value, row in zip(difference, rows, strict=True):
            by_season[row.season] = by_season.get(row.season, 0.0) + value
        improved = sum(1 for total in by_season.values() if total < 0)
        calibration = {
            ("frozen" if method == information else method): measure_calibration_error(
                [pair for row in rows for pair in _collect_line_pairs(row, method, definitions)]
            )
            for method in methods
        }
        rps = {("frozen" if method == information else method):
               float(np.mean([score_ranked_probability(row.distributions[method], row.happened)
                              for row in rows])) for method in methods}  # fmt: skip
        per_cup = {}
        for code in groups:
            chosen = [index for index, row in enumerate(rows) if row.code == code]
            if chosen:
                per_cup[code] = (len(chosen), *bootstrap_mean(
                    [difference[index] for index in chosen], [blocks[index] for index in chosen],
                    repetitions, seed))  # fmt: skip
        per_checkpoint = {}
        warnings, stops = [], []
        for checkpoint in sorted({row.checkpoint for row in rows}, key=_order_checkpoint):
            chosen = [index for index, row in enumerate(rows) if row.checkpoint == checkpoint]
            if not chosen:
                continue
            realised = [scores["vesta"][index] for index in chosen]
            expected = [compute_entropy(rows[index].distributions["vesta"]) for index in chosen]
            gap_values = [own - said for own, said in zip(realised, expected, strict=True)]
            gap, gap_low, gap_high = bootstrap_mean(
                gap_values, [blocks[index] for index in chosen], repetitions, seed)  # fmt: skip
            error = (gap_high - gap_low) / (2 * 1.959964) or float("inf")
            in_errors = -gap / error  # positive: realised below the forecasts' own entropy
            if in_errors > float(alarm["stop_standard_errors"]):
                stops.append(f"{key} {checkpoint}: the realised log score is"
                             f" {in_errors:.1f} standard errors below the forecasts' entropy")
            elif in_errors > float(alarm["warn_standard_errors"]):
                warnings.append(f"{key} {checkpoint}: {in_errors:.1f} standard errors below the"
                                " forecasts' entropy (warning)")  # fmt: skip
            per_checkpoint[checkpoint] = {
                "n": len(chosen),
                "vesta": float(np.mean(realised)),
                "reference": float(np.mean([scores["reference"][index] for index in chosen])),
                "frozen": float(np.mean([scores[information][index] for index in chosen])),
                "difference": float(np.mean([difference[index] for index in chosen])),
                "frozen_difference": float(np.mean([frozen[index] for index in chosen])),
                "entropy": float(np.mean(expected)),
                "standard_errors": in_errors,
            }
        tolerance = float(verdict["calibration_tolerance"])
        passed = (high < 0 and improved >= needed and not stops
                  and calibration["vesta"] <= calibration["reference"] + tolerance)  # fmt: skip
        held_out = [code for code, (_, _, cup_low, _) in per_cup.items() if cup_low > 0]
        judged[key] = {
            "n": len(rows), "blocks": len(set(blocks)),
            "scores": {("frozen" if method == information else method): float(np.mean(values))
                       for method, values in scores.items()},
            "difference": (mean, low, high),
            "frozen_difference": bootstrap_mean(frozen, blocks, repetitions, seed),
            "improved": improved, "seasons": len(by_season), "needed": needed,
            "by_season": by_season, "calibration": calibration, "rps": rps,
            "per_cup": per_cup, "per_checkpoint": per_checkpoint, "warnings": warnings,
            "stops": stops, "passed": passed, "held_out": held_out if passed else [],
        }
    return judged


def report(results: dict, judged: dict, world_path: Path, world_sha: str, commit: str,
           seconds: float) -> str:  # fmt: skip
    stage = results["stage"]
    lines = [f"# The UEFA cups' progression: the {stage} verdict (T4)", "",
             f"World: `{world_path}` (sha256 {world_sha}). Code: `{commit}`. Seasons:"
             f" {', '.join(map(str, results['seasons']))}; {results['draws']} simulations per"
             " cup, checkpoint and method. Written by `python -m vesta.season.cups_verdict`; the"
             " plan declared before the run (1fdd31aa, approved"
             " with five conditions), the alarm `[alarm.progression]` (3506c771)."
             f" {seconds:.0f} seconds.", ""]  # fmt: skip
    for key, result in judged.items():
        mean, low, high = result["difference"]
        lines += [f"## {key}: {'PASSED' if result['passed'] else 'NOT PASSED'}", "",
                  f"- {result['n']} (club, checkpoint) scored, in {result['blocks']} blocks"
                  " (competition-seasons): the bootstrap's interval is as coarse as that.",
                  f"- Mean log score: Vesta {result['scores']['vesta']:.4f}, reference (equal"
                  f" strength) {result['scores']['reference']:.4f}, frozen C0"
                  f" {result['scores']['frozen']:.4f}.",
                  f"- Vesta minus the reference: {mean:+.4f} [{low:+.4f}; {high:+.4f}]"
                  f" ({'below zero' if high < 0 else 'not below zero'}).",
                  f"- Seasons improved: {result['improved']} of {result['seasons']} (needed"
                  f" {result['needed']}): "
                  + ", ".join(f"{season} {total:+.2f}" for season, total
                              in sorted(result["by_season"].items())) + ".",
                  f"- Calibration (expected calibration error of the lines): Vesta"
                  f" {result['calibration']['vesta']:.4f}, reference"
                  f" {result['calibration']['reference']:.4f}, frozen C0"
                  f" {result['calibration']['frozen']:.4f}.",
                  f"- Ranked probability score: Vesta {result['rps']['vesta']:.4f}, reference"
                  f" {result['rps']['reference']:.4f}, frozen C0 {result['rps']['frozen']:.4f}.",
                  "- Vesta minus frozen C0 (information): {:+.4f} [{:+.4f}; {:+.4f}].".format(
                      *result["frozen_difference"])]  # fmt: skip
        lines += [f"- ALARM: {stop}" for stop in result["stops"]]
        lines += [f"- Warning: {warning}" for warning in result["warnings"]]
        if result["held_out"]:
            lines.append("- Held out (their own interval above zero): "
                         + ", ".join(result["held_out"]))
        lines += ["", "**Per cup** (Vesta minus the reference)", "",
                  "| cup | n | mean [95%] |", "|---|---|---|"]
        for code, (count, cup_mean, cup_low, cup_high) in result["per_cup"].items():
            lines.append(f"| {code} | {count} | {cup_mean:+.4f}"
                         f" [{cup_low:+.4f}; {cup_high:+.4f}] |")
        lines += ["", "**Per checkpoint**: mean log scores; the sharpness check (the realised log"
                  " score against the mean entropy of Vesta's forecasts, the gap in bootstrap"
                  " standard errors, positive when the forecasts did better than they said)", "",
                  "| checkpoint | n | Vesta | reference | frozen C0 | Vesta - ref |"
                  " Vesta - frozen |"
                  " entropy | gap (SE) |", "|---|---|---|---|---|---|---|---|---|"]
        for checkpoint, row in result["per_checkpoint"].items():
            lines.append(f"| {checkpoint} | {row['n']} | {row['vesta']:.4f} |"
                         f" {row['reference']:.4f} | {row['frozen']:.4f} |"
                         f" {row['difference']:+.4f} | {row['frozen_difference']:+.4f} |"
                         f" {row['entropy']:.4f} | {row['standard_errors']:+.2f} |")
        lines.append("")
    if results["refused"]:
        lines += ["## Not judged", ""] + [f"- {reason}" for reason in results["refused"]] + [""]
    lines += ["## Notes", "",
              "- The world's `match_ties.winner_team_id` disagrees with the ties' goals (and the"
              " club that played on) on the ties below; the verdict settles every tie from its"
              " goals, the world's winner read only for a tie level after its legs. Reported on"
              " 31 October."]  # fmt: skip
    lines += [f"  - {note}" for note in results["notes"] if "the world's tie" in note]
    lines += [f"- {note}" for note in results["notes"] if "awarded" in note]
    lines += ["- The 2019-20 round of 16 of the Europa League (two ties single matches in August"
              " 2020) is simulated with two legs; the truth reads only the round reached.",
              "- Same-association protection in the draws is not held (as every bracket rule).",
              "- The reference pools the three cups' earlier seasons (the Conference League's"
              " first season has none of its own)."]
    return "\n".join(lines)


def format_temperature_report(results: dict) -> str:
    """The taus by season and stage, and the calibration by band before and after them."""
    lines = ["## Candidate 2: the temperature per stage", "",
             "tau fitted on the seasons before each season judged (grid 0.50-2.00); 2019's on the"
             " warm-up CL 2016-2018 alone (EL before 2019 has no verified format).", "",
             "| season | key | stage | tau |", "|---|---|---|---|"]
    for (season, key, stage), tau in sorted(results["taus"].items()):
        lines.append(f"| {season} | {key} | {stage} | {tau:.2f} |")
    for key in results["candidate_2_bands"]:
        lines += ["", f"**{key}: calibration by band, candidate 1 then candidate 2**", "",
                  "| band | n | said (1) | happened | said (2) | happened |",
                  "|---|---|---|---|---|---|"]
        after = {(low, high): (count, said, happened)
                 for low, high, count, said, happened in results["candidate_2_bands"][key]}
        for low, high, count, said, happened in results["candidate_1_bands"][key]:
            second = after.get((low, high))
            lines.append(f"| {low:.2f}-{high:.2f} | {count} | {said:.3f} | {happened:.3f} | "
                         + (f"{second[1]:.3f} | {second[2]:.3f} |" if second else "- | - |"))
    return "\n".join(lines)


def main(arguments: list[str]) -> int:
    from vesta.research.repository import run_git

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("world", type=Path)
    parser.add_argument("--stage", choices=("development", "blind"), default="development")
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--records", type=Path, default=None)
    parser.add_argument("--keys", choices=("main", "qualifiers"), default="main",
                        help="the main stage (round reached, phase place) or the qualifiers")
    parser.add_argument("--candidate", type=int, choices=(1, 2), default=1,
                        help="2: the temperature per stage, fitted walk-forward (main stage)")
    parser.add_argument("--key", action="append", choices=(ROUND, PLACE, QUALIFIES),
                        default=None,
                        help="judge only these keys (blind: what development passed)")
    options = parser.parse_args(arguments)
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    draws = options.draws or int(spine["season"]["simulations"])
    started = time.monotonic()
    commit = run_git(ROOT, "rev-parse", "HEAD").strip()  # the code the run starts from
    results = (run if options.keys == "main" else run_qualifiers)(options.world, options.stage,
                                                                   draws)  # fmt: skip
    if options.key:
        results["rows"] = [row for row in results["rows"] if row.key in options.key]
    banded = tuple(key for key in (ROUND, PLACE) if not options.key or key in options.key)
    if options.candidate == 2:
        earlier = run(options.world, "development", draws, seasons=WARM_UP, codes=("CL",))["rows"]
        if options.stage == "blind":
            earlier += run(options.world, "development", draws)["rows"]
        results["candidate_1_bands"] = {key: bin_calibration_bands(
            [row for row in results["rows"] if row.key == key], "vesta") for key in banded}
        results["rows"], results["taus"] = apply_temperature(results["rows"], earlier)
        results["candidate_2_bands"] = {key: bin_calibration_bands(
            [row for row in results["rows"] if row.key == key], "vesta") for key in banded}
    judged = judge(results)
    digest = hashlib.sha256()
    with options.world.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 24), b""):
            digest.update(chunk)
    text = report(results, judged, options.world, digest.hexdigest(), commit,
                  time.monotonic() - started)  # fmt: skip
    if options.candidate == 2:
        text += "\n\n" + format_temperature_report(results)
    out = options.out or (ROOT / "docs/research"
                          / f"{date.today().isoformat()}-season-family-cups"
                            f"{'' if options.keys == 'main' else '-qualifiers'}"
                            f"{'-candidate-2' if options.candidate == 2 else ''}"
                            f"-{options.stage}.md")
    out.write_text(text + "\n", encoding="utf-8")
    if options.records:
        options.records.write_text(json.dumps([row.__dict__ for row in results["rows"]]),
                                   encoding="utf-8")  # fmt: skip
    for key, result in judged.items():
        print(key, "PASSED" if result["passed"] else "NOT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
