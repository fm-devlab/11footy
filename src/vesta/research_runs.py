"""The runs of registered hypotheses: each judged once, on the development seasons, by the rules.

    python -m vesta.research_runs <H1|H2|H3|H4> <football-world.db>

The entry is read from the committed registry (`config/vesta/hypotheses.toml`, refused while it
has uncommitted changes); the published model and the candidate forecast the same forecasts of
the development seasons; the leakage alarm of the entry's kind runs first
(`config/vesta/research.toml` [alarm.<kind>]); the verdict (`vesta.research.verdicts.judge`) is
written once to `docs/research/verdicts/<id>.md`.
- H1, ties: extra time's pace and the shoot-out's home share fitted per season on the ties before
  it (`vesta.spine.ties_check.fitted_parameters`), against the config's 1.0 and 0.5; every tie's
  chance of going through for the first leg's home side, its log loss.
- H2, goals in time: the minute of a goal within its half by the live chain's profile of the same
  seasons, against the uniform spread; the M8 band lines, their log loss.
- H3, goals in time: each side's first-half share by the band of its expected goals, against one
  share; M10.highest_scoring_half, its log loss.
- H4, the spine at lineup time: the player ratings read the match's announced eleven instead of
  each side's last one, both forecast at kick-off minus 60 minutes
  (`vesta.spine.announced_eleven`); M1, its ranked probability score. Reported beside it,
  deciding nothing: M2, M3 and M4, and M1 on the matches where a side's eleven changed by three
  or more players from its last one.
"""

import hashlib
import math
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np

from vesta.quantities import load_quantities
from vesta.research.bootstrap import assign_block
from vesta.research.hypotheses import read_committed_hypothesis
from vesta.research.leakage import find_alarms
from vesta.research.repository import run_git
from vesta.research.scores import score
from vesta.research.verdicts import PairedScore, VerdictRules, judge, write_verdict
from vesta.spine.announced_eleven import (
    SHEET_OUT,
    AnnouncedElevenFamily,
    count_changed_starters,
    find_announced_elevens,
)
from vesta.spine.champion import forecast_world_with_champion, read_champion_inputs
from vesta.spine.cups import CupsParameters, forecast_cups
from vesta.spine.dixon_coles import DixonColesParameters
from vesta.spine.goal_minutes import TimingSeasons, fit_profile, measure_added_minutes
from vesta.spine.goal_minutes import read_goal_history as read_history
from vesta.spine.goals_in_time import (
    GoalsInTimeParameters,
    TimeShares,
    fit_first_half_shares,
    fit_method_shares,
    fit_shares_by_band,
    simulate,
    split_regular_minutes,
)
from vesta.spine.goals_in_time_publication import GOAL_BANDS, build_goals_in_time_distributions
from vesta.spine.goals_in_time_verdict import find_outcomes
from vesta.spine.player_ratings import PlayerRatingsParameters
from vesta.spine.score_matrix import SIZE
from vesta.spine.ties import TieParameters
from vesta.spine.ties_check import (
    fit_tie_parameters,
    forecast_through_chance,
    read_extra_time_goals,
    read_ties,
)
from vesta.spine.timing import TimeRules
from vesta.spine.walk_forward import WalkForwardSettings, forecast_with_champion
from vesta.spine.world import (
    open_world,
    read_club_matches,
    read_goal_methods,
    read_half_time_scores,
    read_match_stories,
)

ROOT = Path(__file__).resolve().parents[2]
SMALLEST = 1e-15
H2_KEYS = tuple(f"M8.goal_in_band.band={start}-{end}" for start, end in GOAL_BANDS)
H3_KEYS = ("M10.highest_scoring_half",)


def build_h1_pairs(world: Path, seasons: tuple[int, ...], spine: dict):
    """(pairs, published lines, candidate lines) of H1."""
    cups = CupsParameters.from_config(spine)
    base = TieParameters.from_config(spine)
    with open_world(world) as connection:
        matches = read_club_matches(connection, list(cups.leagues), list(cups.cups),
                                    cups.first_league_season, cups.first_cup_season,
                                    max(seasons))  # fmt: skip
        ties = read_ties(connection, list(cups.cups))
        extra = read_extra_time_goals(connection)
    span = tuple(range(cups.first_cup_season + 1, max(seasons) + 1))
    forecasts = {forecast.match_id: forecast for forecast in forecast_cups(matches, cups, span)}
    fitted = fit_tie_parameters(ties, forecasts, extra, seasons, base)
    pairs, published_lines, candidate_lines = [], [], []
    for tie in ties:
        if tie.season not in seasons or tie.first_leg not in forecasts:
            continue
        if tie.second_leg is not None and tie.second_leg not in forecasts:
            continue
        happened = tie.winner == tie.first_home
        before = forecast_through_chance(tie, forecasts, base)
        after = forecast_through_chance(tie, forecasts, fitted[tie.season])
        pairs.append(
            PairedScore(
                f"tie:{tie.tie_id}",
                tie.season,
                assign_block(tie.competition, tie.kickoff_utc),
                _score_log_loss(before, happened),
                _score_log_loss(after, happened),
            )
        )
        published_lines.append((before, happened))
        candidate_lines.append((after, happened))
    return pairs, published_lines, candidate_lines


def build_goals_in_time_pairs(
    world: Path, seasons: tuple[int, ...], spine: dict, keys: tuple[str, ...], candidate: str
):
    """(pairs, published lines, candidate lines) of H2 or H3 on `keys`."""
    timing = TimingSeasons.from_config(spine)
    competitions = list(spine["walk_forward"]["competitions"])
    parameters = GoalsInTimeParameters.from_config(spine)
    definitions = load_quantities()
    first = int(spine["live_chain"]["first_effects_season"])
    forecasts = forecast_world_with_champion(world, tuple(range(first, max(seasons) + 1)))
    with open_world(world) as connection:
        history = read_history(connection, competitions, timing.first_season, max(seasons))
        shares, candidates, added = {}, {}, {}
        halves = _read_half_times(connection, competitions, first, max(seasons))
        for season in seasons:
            before = history.season < season
            profile = fit_profile(
                history.select(before),
                history.select(before & (history.season >= timing.find_recent_seasons(season)[0])),
                timing.smoothing_minutes,
            )
            added[season] = measure_added_minutes(profile)
            scores = read_half_time_scores(connection, competitions, timing.first_season,
                                           season - 1)  # fmt: skip
            home, away = fit_first_half_shares(
                *(np.array(column) for column in (scores.home_goals, scores.away_goals,
                                                  scores.home_half_time, scores.away_half_time)),
                parameters.first_half_share_without_data,
            )  # fmt: skip
            methods = read_goal_methods(connection, competitions, timing.first_season,
                                        season - 1)  # fmt: skip
            shares[season] = TimeShares(home, away, fit_method_shares(methods))
            if candidate == "H2":
                candidates[season] = replace(shares[season],
                                             regular_minutes=split_regular_minutes(profile))
            else:
                candidates[season] = replace(shares[season], shares_by_band=_fit_bands(
                    halves, forecasts, season, parameters.first_half_share_without_data))
        stories = read_match_stories(connection, {forecast.match_id for forecast in forecasts
                                                  if forecast.season in seasons})  # fmt: skip
    pairs, published_lines, candidate_lines = [], [], []
    for forecast in forecasts:
        if forecast.season not in seasons or forecast.match_id not in stories:
            continue
        story = stories[forecast.match_id]
        happened = find_outcomes(story)
        used = replace(parameters, added_time_minutes=added[forecast.season])
        published = _simulate_values(forecast, shares[forecast.season], used, story)
        tried = _simulate_values(forecast, candidates[forecast.season], used, story)
        for key in keys:
            if (key, "match") not in happened:
                continue
            kind = definitions[key].quantity.distribution_kind
            outcome = happened[(key, "match")]
            before = score(kind, published[key], outcome)["log_loss"]
            after = score(kind, tried[key], outcome)["log_loss"]
            pairs.append(PairedScore(f"{forecast.match_id}:{key}", forecast.season,
                                     assign_block(forecast.competition, forecast.forecast_time),
                                     before, after))  # fmt: skip
            published_lines += _make_lines(kind, published[key], outcome)
            candidate_lines += _make_lines(kind, tried[key], outcome)
    return pairs, published_lines, candidate_lines


def build_h4_pairs(world: Path, seasons: tuple[int, ...], spine: dict):
    """(pairs, published lines, candidate lines, the candidate's accuracy, what is reported
    beside the verdict) of H4."""
    matches, facts = read_champion_inputs(world, seasons)
    settings = WalkForwardSettings.from_config(spine)
    ratings = PlayerRatingsParameters.from_config(spine)
    common = (matches, facts, TimeRules.from_config(spine),
              DixonColesParameters.from_config(spine, "dixon_coles_expected_goals"), ratings,
              int(spine["champion_calibration"]["seasons"]), settings, seasons)  # fmt: skip
    published = {item.match_id: item for item in forecast_with_champion(*common, earlier=SHEET_OUT)}
    tried = {item.match_id: item for item in forecast_with_champion(
        *common, ratings=AnnouncedElevenFamily(ratings, settings.competitions,
                                               find_announced_elevens(facts)),
        earlier=SHEET_OUT)}  # fmt: skip
    goals = {match.match_id: (match.home_goals, match.away_goals) for match in matches}
    changed = count_changed_starters(facts, matches)
    sides = {match.match_id: (match.home_team_id, match.away_team_id) for match in matches}
    pairs, published_lines, candidate_lines = [], [], []
    hits = 0
    beside: dict[str, list[float]] = {}
    for match_id in sorted(set(published) & set(tried)):
        before, after = published[match_id], tried[match_id]
        if before.season not in seasons:
            continue
        values = [[name, probability] for name, probability
                  in zip(("home", "draw", "away"), before.probabilities, strict=True)]
        candidate = [[name, probability] for name, probability
                     in zip(("home", "draw", "away"), after.probabilities, strict=True)]
        rps = (score("ordered", values, before.result)["ranked_probability_score"],
               score("ordered", candidate, before.result)["ranked_probability_score"])
        pairs.append(PairedScore(f"{match_id}:M1.result", before.season,
                                 assign_block(before.competition, before.forecast_time), *rps))
        for name, said, tried_said in zip(("home", "draw", "away"), before.probabilities,
                                          after.probabilities, strict=True):  # fmt: skip
            published_lines.append((said, name == before.result))
            candidate_lines.append((tried_said, name == before.result))
        likeliest = max(range(3), key=lambda index: after.probabilities[index])
        hits += ("home", "draw", "away")[likeliest] == before.result
        home, away = goals[match_id]
        for arm, item in (("published", before), ("candidate", after)):
            matrix = item.matrix
            size = len(matrix)
            joint = {"matrix": matrix.tolist(), "tail": max(0.0, 1.0 - float(matrix.sum()))}
            totals = [float(sum(matrix[i][j] for i in range(size) for j in range(size)
                                if i + j == total)) for total in range(size)]  # fmt: skip
            count = {"probabilities": totals[:-1], "tail": max(0.0, 1.0 - sum(totals[:-1])),
                     "tail_from": size - 1}  # fmt: skip
            both = {"yes": float(matrix[1:, 1:].sum())}
            beside.setdefault(f"M2.score {arm}", []).append(
                score("joint_count", joint, (home, away))["log_loss"])
            beside.setdefault(f"M3.goals {arm}", []).append(
                score("count", count, home + away)["log_loss"])
            beside.setdefault(f"M4.both_score {arm}", []).append(
                score("binary", both, home > 0 and away > 0)["log_loss"])
        if max(changed.get((match_id, team), 0) for team in sides[match_id]) >= 3:
            beside.setdefault("M1 eleven changed by 3+ published", []).append(rps[0])
            beside.setdefault("M1 eleven changed by 3+ candidate", []).append(rps[1])
    accuracy = hits / max(len(pairs), 1)
    reported = {name: (len(items), math.fsum(items) / len(items)) for name, items in
                beside.items()}  # fmt: skip
    return pairs, published_lines, candidate_lines, accuracy, reported


def _simulate_values(forecast, shares, parameters, story) -> dict:
    population = simulate(forecast.matrix, shares, parameters, forecast.match_id)
    return {item.quantity_key: item.values for item in build_goals_in_time_distributions(
        population, story.match_id, story.home_team_id, story.away_team_id, "run")}  # fmt: skip


def _make_lines(kind: str, values, outcome) -> list[tuple[float, bool]]:
    if kind == "binary":
        return [(float(values["yes"]), bool(outcome))]
    return [(float(probability), name == outcome) for name, probability in values.items()]


def _read_half_times(connection, competitions, first_season, last_season) -> dict:
    marks = ", ".join("?" for _ in competitions)
    rows = connection.execute(
        f"""
        SELECT m.match_id, s.season, m.home_goals, m.away_goals, m.home_goals_half_time,
               m.away_goals_half_time
        FROM matches m JOIN seasons s USING (season_id) JOIN competitions c USING (competition_id)
        WHERE c.code IN ({marks}) AND s.season BETWEEN ? AND ? AND m.status = 'played'
          AND m.home_goals IS NOT NULL AND m.home_goals_half_time IS NOT NULL
        """,
        (*competitions, first_season, last_season),
    ).fetchall()
    return {row[0]: row[1:] for row in rows}


def _fit_bands(halves: dict, forecasts: list, season: int, without_data: float):
    """H3's shares by band, fitted on the training matches that hold a pre-match forecast."""
    rows = []
    goals = np.arange(SIZE)
    for forecast in forecasts:
        if forecast.season >= season or forecast.match_id not in halves:
            continue
        matrix = forecast.matrix
        _, home, away, home_half, away_half = halves[forecast.match_id]
        rows.append((home, away, home_half, away_half, float(matrix.sum(axis=1) @ goals),
                     float(matrix.sum(axis=0) @ goals)))  # fmt: skip
    columns = [np.array(column) for column in zip(*rows, strict=True)]
    return fit_shares_by_band(*columns, without_data)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _score_log_loss(probability: float, happened: bool) -> float:
    return -math.log(max(probability if happened else 1.0 - probability, SMALLEST))


def run(hypothesis_id: str, world: Path) -> Path:
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    research = tomllib.loads((ROOT / "config/vesta/research.toml").read_text(encoding="utf-8"))
    hypothesis = read_committed_hypothesis(
        ROOT, hypothesis_id, int(research["verdict"]["grid_max"])
    )
    seasons = tuple(research["seasons"]["development"])
    accuracy, reported = None, {}
    if hypothesis_id == "H1":
        pairs, published, candidate = build_h1_pairs(world, seasons, spine)
    elif hypothesis_id == "H4":
        pairs, published, candidate, accuracy, reported = build_h4_pairs(world, seasons, spine)
    elif hypothesis_id in ("H2", "H3"):
        keys = H2_KEYS if hypothesis_id == "H2" else H3_KEYS
        pairs, published, candidate = build_goals_in_time_pairs(world, seasons, spine, keys,
                                                                hypothesis_id)  # fmt: skip
    else:
        raise ValueError(f"no run is written for {hypothesis_id}")
    gain = math.fsum(pair.published - pair.candidate for pair in pairs) / max(len(pairs), 1)
    findings = find_alarms(hypothesis.distribution_kind, research.get("alarm", {}), gain,
                           accuracy=accuracy)  # fmt: skip
    provenance = {"code_commit": run_git(ROOT, "rev-parse", "HEAD").strip(),
                  "world_build": f"{world} (sha256 {_hash_file(world)})",
                  "command": f"python -m vesta.research_runs {hypothesis_id} <world>"}  # fmt: skip
    verdict = judge(hypothesis, pairs, published, candidate, VerdictRules.from_config(research),
                    findings, provenance, ROOT)  # fmt: skip
    path = write_verdict(verdict, ROOT)
    if reported:
        lines = ["", "## Reported beside it, deciding nothing", "",
                 f"The candidate's accuracy on M1: {accuracy:.4f}.", "",
                 "| What | Forecasts | Mean |", "|---|---|---|"]  # fmt: skip
        lines += [f"| {name} | {count} | {mean:.5f} |" for name, (count, mean)
                  in sorted(reported.items())]  # fmt: skip
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines) + "\n")
    return path


def main(arguments: list[str]) -> int:
    if len(arguments) != 2:
        print(__doc__)
        return 2
    print(f"written {run(arguments[0], Path(arguments[1]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
