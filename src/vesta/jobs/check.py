"""The check job: after each dataset build, what happened to every forecast of record, and the
track record's figures (`docs/FORECAST_LOG.md` §8-§9).

    python -m vesta.jobs.check <football-world.db> <forecast-log.db> [--now <instant>]

First the forecasts of record: every match whose pre-match forecasts the log holds, that has
none yet, and that the world shows kicked off by `now` (played, awarded or abandoned; cancelled,
at its time, so it is voided) gets them, the last pre-match publication of each quantity before
its kick-off (`record_forecasts_of_record`; a forecast published after it never counts). A match
recorded once is never recorded again, so the job can be run again. Then, for every match with
forecasts of record, Vesta's and any earlier record alike, each figure
grouped by its version:
- a match cancelled, abandoned or awarded: a void for every quantity of the match, once. An
  awarded result is a decision off the pitch, and the score on the pitch is not the result the
  forecast was about once the match was awarded (28 October 2026); a
  match checked as played and awarded later is voided then, and leaves the figures;
- a match played: what happened to each quantity of record, in the shape its kind is scored on,
  written to `outcomes` (once; again with its reason when a later build of the world says
  otherwise), then the check of every line of the forecast of record in `line_checks` (again
  only when its answer changes). The goals in time read the match's story (its goals in order,
  `vesta.spine.world.read_match_stories`); a match whose goal events do not add up to its score
  has none, and its goals-in-time quantities stay pending;
- anything else: pending.
Then the figures, computed again from every counted forecast of record (neither void nor
withdrawn) and written with one `computed_at`: per probability kind, model version, competition
(and 0, every competition) and period (`season:<year>`, `month:<YYYY-MM>`, `all`), the scores of
§9 beside the same scores of the naive reference (the frequencies of the competition's seasons
before, from the first training season, smoothed by half a case: the reference of the verdicts),
and the calibration of every line by tenths. A pre-match publication the world does not link (a
Vesta v1 forecast left "unmatched") is counted beside the figures (`unmatched_count`, every
competition together: its competition is not known), by its key's probability kind, the month of
its kickoff and `all`, never dropped.
"""

import argparse
import json
import math
import sqlite3
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

from vesta.forecast_log.store import open_log
from vesta.forecast_log.writing import (
    KeyScore,
    TrackRecordHits,
    TrackRecordScore,
    record_calibration,
    record_forecasts_of_record,
    record_hits,
    record_key_scores,
    record_line_check,
    record_outcome,
    record_scores,
    record_void,
)
from vesta.forecast_reading import read_line, round_to_whole_percentages
from vesta.jobs.fixtures import format_now_utc
from vesta.players.model import read_happened
from vesta.players.thresholds import COUNTS, read_counts
from vesta.players.thresholds import KEYS as THRESHOLDS
from vesta.quantities import load_quantities
from vesta.research.calibration import bucket_lines
from vesta.research.scores import METRICS_BY_KIND, OutcomeError, score
from vesta.spine.goals_in_time_verdict import (
    OUTCOME_NAMES,
    _has_line_happened,
    _smooth_frequencies,
    find_outcomes,
)
from vesta.spine.world import NINETY_AWAY, NINETY_HOME, MatchStory, open_world, read_match_stories
from vesta.team_counts.model import FAMILY_A_KEYS as FAMILY_A_TEAM_COUNT_KEYS
from vesta.team_counts.model import KEYS as TEAM_COUNT_KEYS
from vesta.team_counts.outcomes import read_match_counts
from vesta.team_counts.verdict import smooth_frequencies

ROOT = Path(__file__).resolve().parents[3]
FINAL = ("played",)
VOID = ("cancelled", "abandoned", "awarded")
SPINE_NAMES = {"M1.result": ("home", "draw", "away")}
BASELINE = "the competition's frequencies of the seasons before"
SKILL = "skill"  # the share of the reference's loss saved, averaged across the kind's keys
ALL_VERSIONS = "all"  # every model version together, in the hits and the scores (no version)


def settle_spine_outcomes(home: int, away: int) -> dict[tuple[str, str], object]:
    """What happened to the spine's quantities, by (key, subject: match, home or away)."""
    result = "home" if home > away else "away" if away > home else "draw"
    return {("M1.result", "match"): result, ("M2.score", "match"): (home, away),
            ("M3.goals", "match"): home + away, ("M4.both_score", "match"): home > 0 and away > 0,
            ("M5.goals.team", "home"): home, ("M5.goals.team", "away"): away}  # fmt: skip


def settle_match_outcomes(home_goals: int, away_goals: int, story: MatchStory | None,
                          counts: dict | None = None) -> dict[tuple[str, str], object]:  # fmt: skip
    happened = settle_spine_outcomes(home_goals, away_goals)
    if story is not None:
        happened.update(find_outcomes(story))
    if counts:
        happened.update(counts)
    return happened


def read_team_count_outcomes(world: sqlite3.Connection, match_ids: set[int]) -> dict[int, dict]:
    """Per match, what happened to the team-count keys of every count (corners, cards, shots,
    shots on target, fouls, offsides), by family A's one rule (`vesta.team_counts.outcomes`): the
    layer of record (leagues: the report, else the feed; cups: the feed), the feed's zeros that
    mean unknown left pending; each side folded at its key's size as the distributions are (the last
    cell is that many or more). A count left pending has no outcome."""
    config = tomllib.loads((ROOT / "config/vesta/team_counts.toml").read_text(encoding="utf-8"))
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    if not match_ids:
        return {}
    scopes = {"leagues": set(spine["walk_forward"]["competitions"]),
              "cups": set(spine["cups"]["cups"])}  # fmt: skip
    marks = ", ".join("?" for _ in match_ids)
    by_scope: dict[str, set[int]] = defaultdict(set)
    for match_id, code in world.execute(
            f"SELECT m.match_id, c.code FROM matches m JOIN seasons s ON s.season_id ="
            f" m.season_id JOIN competitions c ON c.competition_id = s.competition_id"
            f" WHERE m.match_id IN ({marks})", tuple(match_ids)):  # fmt: skip
        for scope, codes in scopes.items():
            if code in codes:
                by_scope[scope].add(match_id)
    found: dict[int, dict] = defaultdict(dict)
    for scope, chosen in by_scope.items():
        for count, (total_key, team_key, sides_key) in {**TEAM_COUNT_KEYS,
                                                        **FAMILY_A_TEAM_COUNT_KEYS}.items():
            top = int(config["counts"][count]["size"]) - 1
            for match_id, (home_count, away_count, _) in read_match_counts(
                    world, chosen, count, scope).items():  # fmt: skip
                happened = {(total_key, "match"): min(home_count + away_count, 2 * top),
                            (team_key, "home"): min(home_count, top),
                            (team_key, "away"): min(away_count, top)}
                if sides_key is not None:
                    happened[(sides_key, "match")] = (min(home_count, top), min(away_count, top))
                found[match_id].update(happened)
    return dict(found)


LINEUP_KEYS = ("L1.starts", "L6.plays", "L6.minutes", "L2.formation.team", "L9.position")
SCORER_KEYS = ("P3.scores", "P3.goals", "P3.first_scorer", "P5.assists", "P5.scores_or_assists")
THRESHOLD_KEYS = tuple(THRESHOLDS)
PLAYER_KEYS = SCORER_KEYS + THRESHOLD_KEYS


def read_lineup_sheets(world: sqlite3.Connection, match_ids: set[int]) -> dict[int, dict]:
    """Per match with a sheet of record: each player's (started, minutes, position if he
    started) and each team's formation."""
    if not match_ids:
        return {}
    marks = ", ".join("?" for _ in match_ids)
    found: dict[int, dict] = {}
    for match_id, team_id, formation in world.execute(
            f"SELECT match_id, team_id, formation FROM match_team_sheets"
            f" WHERE match_id IN ({marks})", tuple(match_ids)):  # fmt: skip
        found.setdefault(match_id, {"players": {}, "formations": {}})["formations"][
            team_id] = formation
    for match_id, player_id, started, minutes, code in world.execute(
            f"SELECT match_id, player_id, started, minutes_played, position_code"
            f" FROM match_players WHERE match_id IN ({marks})", tuple(match_ids)):  # fmt: skip
        if match_id in found:
            found[match_id]["players"][player_id] = (bool(started), int(minutes or 0),
                                                     code if started else None)  # fmt: skip
    return found


def read_team_sheets(world: sqlite3.Connection, match_ids: set[int]) -> dict:
    """(match, team) -> player -> (started, minutes, position if he started): a keeper's
    window for his saves (`vesta.players.thresholds.read_counts`)."""
    found: dict = defaultdict(dict)
    chosen = sorted(match_ids)
    for start in range(0, len(chosen), 500):
        part = chosen[start:start + 500]
        for match_id, team_id, player_id, started, minutes, code in world.execute(
                "SELECT match_id, team_id, player_id, started, minutes_played, position_code"
                f" FROM match_players WHERE match_id IN ({', '.join('?' * len(part))})",
                part):  # fmt: skip
            found[(match_id, team_id)][player_id] = (bool(started), int(minutes or 0),
                                                     code if started else None)  # fmt: skip
    return dict(found)


def settle_lineup_outcome(
    key: str, subject: int, sheet: dict, definitions: dict
) -> tuple[bool, object]:
    """(settled, what happened) of a lineups key: a player not on the sheet did not start, did
    not play, played 0 minutes; his position is void when he did not start (the key is "if he
    starts"), returned as (True, None); a formation the sheet lacks stays pending (False)."""
    lines = definitions[key].quantity.lines
    if key == "L2.formation.team":
        formation = sheet["formations"].get(subject)
        if not formation:
            return False, None
        return True, formation if f"outcome={formation}" in lines else "other"
    started, minutes, code = sheet["players"].get(subject, (False, 0, None))
    if key == "L1.starts":
        return True, started
    if key == "L6.plays":
        return True, started or minutes > 0
    if key == "L6.minutes":
        return True, min(minutes, 90)
    if not started or not code:
        return True, None
    return True, code if f"outcome={code}" in lines else "other"


def settle_scorer_outcome(key: str, subject: int, sheet: dict, did) -> object:
    """What happened to a scorers key ("if he plays"): None when the player did not play (not
    on the sheet, or 0 minutes from the bench); else his goals, assists or first goal from the
    goals of record (`vesta.players.model.read_happened`, `did`: his, or None for none)."""
    started, minutes, _ = sheet["players"].get(subject, (False, 0, None))
    if not (started or minutes > 0):
        return None
    goals, assists, first = (did.goals, did.assists, did.first) if did else (0, 0, False)
    return {"P3.scores": goals > 0, "P3.goals": goals, "P3.first_scorer": first,
            "P5.assists": assists > 0, "P5.scores_or_assists": goals > 0 or assists > 0}[key]


def settle_threshold_outcome(key: str, subject: int, sheet: dict, counts) -> tuple[bool, object]:
    """(settled, what happened) of a player threshold ("if he plays"): (True, None) when he did
    not play; his count of the feed's player statistics (`counts`: his, or None), booked as yes or
    no; (False, None), pending, when the source holds no count for him."""
    started, minutes, _ = sheet["players"].get(subject, (False, 0, None))
    if not (started or minutes > 0):
        return True, None
    if counts is None:
        return False, None
    threshold = THRESHOLDS[key]
    value = counts[0][COUNTS.index(threshold.count)]
    if math.isnan(value):
        return False, None
    return True, bool(value >= 1) if threshold.kind == "binary" else int(value)


def run(world_path: Path, log_path: Path, now: str, say=print) -> dict:
    definitions = load_quantities()
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    first_training = int(spine["walk_forward"]["first_training_season"])
    build = world_path.name
    log = open_log(log_path)
    with open_world(world_path) as world:
        recorded = record_kicked_off_matches(log, world, now)
    world.close()
    say(f"forecasts of record: {recorded[0]} matches, {recorded[1]} forecasts written")
    records = log.execute(
        "SELECT r.match_id, r.quantity_key, r.subject_id, r.publication_id, d.model_version,"
        " d.window_key, d.distribution_values, r.real_kickoff_utc"
        " FROM forecasts_of_record r JOIN distributions d ON d.publication_id = r.publication_id"
        " AND d.quantity_key = r.quantity_key AND d.subject_id = r.subject_id").fetchall()
    published = {row[0] for row in log.execute("SELECT DISTINCT quantity_key FROM"
                                                " published_quantities")}  # fmt: skip
    records = keep_checkable(records, definitions, published, say)
    match_ids = {row[0] for row in records}
    counts = {"matches": len(match_ids), "final": 0, "void": 0, "pending": 0, "outcomes": 0,
              "line_checks": 0}  # fmt: skip
    with open_world(world_path) as world:
        state = _read_matches(world, match_ids)
        stories = read_match_stories(world, {match for match, row in state.items()
                                             if row["status"] == "played"})  # fmt: skip
        counted = read_team_count_outcomes(world, {match for match, row in state.items()
                                             if row["status"] == "played"})  # fmt: skip
        references = _build_references(world, state, definitions, first_training)
        sheets = read_lineup_sheets(world, {record[0] for record in records
                                            if record[1] in LINEUP_KEYS + PLAYER_KEYS
                                            and state.get(record[0], {}).get("status") in FINAL})
        scored, _ = read_happened(world, {record[0] for record in records
                                          if record[1] in SCORER_KEYS and record[0] in sheets})
        wanted = {record[0] for record in records
                  if record[1] in THRESHOLD_KEYS and record[0] in sheets}  # fmt: skip
        player_counts = read_counts(world, read_team_sheets(world, wanted), wanted)
    voided = {row[0] for row in log.execute("SELECT DISTINCT match_id FROM voids"
                                            " WHERE quantity_key IS NULL")}  # fmt: skip
    happened_of: dict[tuple[int, str, int], object] = {}
    for match_id, row in state.items():
        if row["status"] in VOID:
            counts["void"] += 1
            if match_id not in voided:
                record_void(log, row["status"], f"world build {build}: status {row['status']}",
                            now, match_id=match_id)  # fmt: skip
            continue
        if row["status"] not in FINAL:
            counts["pending"] += 1
            continue
        counts["final"] += 1
        story = stories.get(match_id)
        for (key, side), value in settle_match_outcomes(row["home"], row["away"], story,
                                                        counted.get(match_id)).items():  # fmt: skip
            subject = (match_id if side == "match" else
                       row["home_team_id"] if side == "home" else row["away_team_id"])  # fmt: skip
            happened_of[(match_id, key, subject)] = value
    player_voids = {(row[0], row[1], row[2]) for row in log.execute(
        "SELECT match_id, quantity_key, subject_id FROM voids WHERE quantity_key IS NOT NULL")}
    for match_id, key, subject, *_ in records:
        if key not in LINEUP_KEYS + PLAYER_KEYS or match_id not in sheets:
            continue
        if key in THRESHOLD_KEYS:
            settled, value = settle_threshold_outcome(key, subject, sheets[match_id],
                                                      player_counts.get((match_id, subject)))
            if not settled:
                continue
            if value is None:
                if (match_id, key, subject) not in player_voids:
                    record_void(log, "did_not_play", f"world build {build}: did not play, and"
                                f" {key} is if he plays (not on the pitch on the sheet of"
                                " record)", now, match_id=match_id, quantity_key=key,
                                subject_id=subject)  # fmt: skip
                    player_voids.add((match_id, key, subject))
                continue
            happened_of[(match_id, key, subject)] = value
            continue
        if key in SCORER_KEYS:
            value = settle_scorer_outcome(key, subject, sheets[match_id],
                                          scored.get((match_id, subject)))  # fmt: skip
            if value is None:
                if (match_id, key, subject) not in player_voids:
                    record_void(log, "did_not_play", f"world build {build}: did not play, and"
                                f" {key} is if he plays (not on the pitch on the sheet of"
                                " record)", now, match_id=match_id, quantity_key=key,
                                subject_id=subject)  # fmt: skip
                    player_voids.add((match_id, key, subject))
                continue
            happened_of[(match_id, key, subject)] = value
            continue
        settled, value = settle_lineup_outcome(key, subject, sheets[match_id], definitions)
        if not settled:
            continue
        if value is None:
            if (match_id, key, subject) not in player_voids:
                # the log's reasons hold no "did not start": a key "if he starts" is void under
                # did_not_play, its evidence saying which condition failed
                record_void(log, "did_not_play", f"world build {build}: did not start, and"
                            f" {key} is if he starts (not a starter on the sheet of record)",
                            now, match_id=match_id, quantity_key=key,
                            subject_id=subject)  # fmt: skip
                player_voids.add((match_id, key, subject))
            continue
        happened_of[(match_id, key, subject)] = value
    last_outcomes = _read_last_outcomes(log)
    last_checks = _read_last_checks(log)
    for match_id, key, subject, publication_id, _, window, values_text, _ in records:
        if (match_id, key, subject) not in happened_of or match_id not in state:
            continue
        value = happened_of[(match_id, key, subject)]
        stored = json.loads(json.dumps(value))
        previous = last_outcomes.get((key, subject, window, match_id))
        if previous != stored:
            record_outcome(log, key, subject, value, _describe_sources(key), build, now,
                           match_id=match_id, window_key=window,
                           correction_reason=None if previous is None else
                           f"the world build {build} says otherwise")  # fmt: skip
            counts["outcomes"] += 1
        kind = definitions[key].quantity.distribution_kind
        values = json.loads(values_text)
        for (line,) in log.execute(
                "SELECT line_key FROM probabilities WHERE publication_id = ? AND quantity_key = ?"
                " AND subject_id = ? AND window_key = ?",
                (publication_id, key, subject, window)):  # fmt: skip
            happened = _has_line_happened(kind, values, line, value)
            if last_checks.get((publication_id, key, subject, line)) != ("checked", happened):
                record_line_check(log, publication_id, key, subject, line, "checked", happened,
                                  now)  # fmt: skip
                counts["line_checks"] += 1
    figures = _rewrite_figures(log, records, state, happened_of, references, definitions, now)
    hits = count_hits(log, records, state, happened_of)
    if hits:
        record_hits(log, hits, now)
    log.close()
    say(f"{counts['matches']} matches of record: {counts['final']} final, {counts['void']} void,"
        f" {counts['pending']} pending; {counts['outcomes']} outcomes and"
        f" {counts['line_checks']} line checks written; {figures} figures")
    if counts["final"] and not figures:
        say("no figures: no checked forecast has a reference (the competition has no season "
            "before the one forecast in the world, from the first training season)")
    return counts


def keep_checkable(records: list, definitions: dict, published: set[str], say=print) -> list:
    """The forecasts of record of the keys the job knows (`config/vesta/quantities.csv`) and
    the log has published; the others are skipped, each key said once, never a stop."""
    kept, skipped = [], {}
    for record in records:
        key = record[1]
        if key not in definitions:
            skipped.setdefault(key, "not in config/vesta/quantities.csv")
        elif key not in published:
            skipped.setdefault(key, "never in published_quantities")
        else:
            kept.append(record)
    for key, reason in sorted(skipped.items()):
        say(f"skipped {key}: {reason}")
    return kept


HITS_KEY = "M1.result"


def find_likeliest_outcome(values: object) -> str | None:
    """The result's likeliest outcome as the page shows it, or None when the two largest shown
    figures are equal (a tie: the forecast did not choose). The three are shown together by the
    percentages rule (`vesta.forecast_reading.round_to_whole_percentages`, since 4 November 2026;
    before, each value was rounded alone)."""
    outcomes = ("home", "draw", "away")
    shown = round_to_whole_percentages([read_line("ordered", values, f"outcome={outcome}")
                               for outcome in outcomes])  # fmt: skip
    ranked = sorted(zip(shown, outcomes, strict=True), key=lambda item: -item[0])
    return None if ranked[0][0] == ranked[1][0] else ranked[0][1]


def count_hits(log, records, state, happened_of) -> list[TrackRecordHits]:
    """Per competition season and model version (and 'all' versions): the counted matches
    whose result forecast of record chose one outcome, how many it happened in, the ties, and the
    void, withdrawn and pending matches left out (FORECAST_LOG §9)."""
    withdrawn = {row[0] for row in log.execute("SELECT publication_id FROM withdrawals")}
    void_matches = {row[0] for row in log.execute("SELECT match_id FROM voids"
                                                  " WHERE quantity_key IS NULL")}  # fmt: skip
    tallies: dict[tuple[str, int, int], list[int]] = defaultdict(lambda: [0] * 6)
    for match_id, key, subject, publication_id, version, _, values_text, _ in records:
        row = state.get(match_id)
        if key != HITS_KEY or row is None:
            continue
        groups = [(name, row["competition_id"], row["season_id"])
                  for name in (version, ALL_VERSIONS)]
        if match_id in void_matches or row["status"] in VOID:
            cell = 3
        elif publication_id in withdrawn:
            cell = 4
        elif (match_id, key, subject) not in happened_of:
            cell = 5
        else:
            chosen = find_likeliest_outcome(json.loads(values_text))
            cell = 2 if chosen is None else 0
            if chosen is not None and chosen == happened_of[(match_id, key, subject)]:
                for group in groups:
                    tallies[group][1] += 1
        for group in groups:
            tallies[group][cell] += 1
    return [TrackRecordHits(HITS_KEY, version, competition, season_id, *counts)
            for (version, competition, season_id), counts in sorted(tallies.items())]


def record_kicked_off_matches(log: sqlite3.Connection, world: sqlite3.Connection,
                              now: str) -> tuple[int, int]:  # fmt: skip
    """The forecasts of record of every match with pre-match forecasts and none of record yet
    that the world shows kicked off by `now`, at its kick-off there; returns (matches,
    forecasts) written."""
    waiting = {row[0] for row in log.execute(
        "SELECT DISTINCT p.match_id FROM publications p WHERE p.stage = 'pre_match'"
        " AND p.match_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM forecasts_of_record r"
        " WHERE r.match_id = p.match_id)")}  # fmt: skip
    marks = ", ".join("?" for _ in FINAL + VOID)
    kicked_off = [(match_id, kickoff) for match_id, kickoff in world.execute(
        f"SELECT match_id, kickoff_utc FROM matches WHERE status IN ({marks})"
        f" AND kickoff_utc <= ? ORDER BY kickoff_utc, match_id", (*FINAL, *VOID, now))
        if match_id in waiting]  # fmt: skip
    matches = forecasts = 0
    for match_id, kickoff in kicked_off:
        written = record_forecasts_of_record(log, match_id, kickoff, now)
        matches += bool(written)
        forecasts += written
    return matches, forecasts


def _read_matches(world: sqlite3.Connection, match_ids: set[int]) -> dict[int, dict]:
    """Each match's status, score after 90 minutes on the pitch, teams, competition and
    season."""
    state = {}
    for row in world.execute(
        f"""
        SELECT m.match_id, m.status, {NINETY_HOME}, {NINETY_AWAY}, m.home_team_id,
               m.away_team_id,
               c.competition_id, s.season, m.kickoff_utc, m.season_id
        FROM matches m JOIN seasons s ON s.season_id = m.season_id
        JOIN competitions c ON c.competition_id = s.competition_id
        """
    ):
        if row[0] in match_ids:
            state[row[0]] = dict(zip(("match_id", "status", "home", "away", "home_team_id",
                                      "away_team_id", "competition_id", "season", "kickoff",
                                      "season_id"),
                                     row, strict=True))  # fmt: skip
    return state


def _build_references(world, state, definitions, first_training) -> dict:
    """(competition, season) -> (key, side) -> the frequency distribution of the seasons
    before, from the first training season: the naive reference."""
    wanted = {(row["competition_id"], row["season"]) for row in state.values()}
    competitions = {competition for competition, _ in wanted}
    if not competitions:
        return {}
    marks = ", ".join("?" for _ in competitions)
    history = world.execute(
        f"SELECT m.match_id, c.competition_id, s.season,"
        f" {NINETY_HOME}, {NINETY_AWAY}, m.status FROM matches m"
        f" JOIN seasons s ON s.season_id = m.season_id JOIN competitions c"
        f" ON c.competition_id = s.competition_id WHERE c.competition_id IN ({marks})"
        f" AND m.status = 'played' AND s.season >= ?",
        (*competitions, first_training)).fetchall()  # fmt: skip
    stories = read_match_stories(world, {row[0] for row in history if row[5] == "played"})
    counted = read_team_count_outcomes(world, {row[0] for row in history if row[5] == "played"})
    seen: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for match_id, competition, season, home, away, _ in history:
        seen[(competition, season)].append(
            settle_match_outcomes(home, away, stories.get(match_id), counted.get(match_id))
        )
    references = {}
    for competition, season in wanted:
        training = [happened for (other, earlier), rows in seen.items()
                    if other == competition and earlier < season for happened in rows]  # fmt: skip
        keys = {key for happened in training for key in happened}
        reference = {}
        for key in keys:
            kind = definitions[key[0]].quantity.distribution_kind
            observed = [happened[key] for happened in training if key in happened]
            names = SPINE_NAMES.get(key[0])
            count = next((name for name, keys in TEAM_COUNT_KEYS.items() if key[0] in keys), None)
            if count is not None:  # at the model's sizes, so no outcome falls in an empty tail
                reference[key] = smooth_frequencies(
                    kind, observed, read_team_count_size(count), key[0] == TEAM_COUNT_KEYS[count][0]
                )
            elif names is not None:
                reference[key] = _estimate_named_frequency(names, observed)
            elif kind == "joint_count" or key[0] in OUTCOME_NAMES or kind != "categorical":
                reference[key] = _smooth_frequencies(key[0], kind, observed)
        references[(competition, season)] = reference
    return references


def _estimate_named_frequency(names: tuple[str, ...], observed: list) -> list:
    counts = {name: sum(seen == name for seen in observed) + 0.5 for name in names}
    total = sum(counts.values())
    return [[name, counts[name] / total] for name in names]


def _rewrite_figures(log, records, state, happened_of, references, definitions, now: str) -> int:
    """The scores and the calibration of every counted forecast of record, written again.

    Each quantity key is scored on its own, beside its reference (`track_record_key_scores`); a
    key's skill is the share of the reference's loss it saves, 1 - its loss / the reference's, on
    the deciding metric of its distribution kind. A probability kind groups keys of several
    distribution kinds, so its summary (`track_record_scores`, metric `skill`) is the skill
    averaged across its keys, each key weighing the same, never their raw losses averaged.

    Beside each model version's rows, the summary of every version together (`ALL_VERSIONS`,
    5 November 2026: Vesta is shown as one thing): each key's own and reference
    losses pooled over all its forecasts, whatever their version, its skill taken from those, then
    averaged across the kind's keys as above; void, withdrawn and unmatched summed. Never an
    average of the versions' summaries. It is written to `track_record_scores` only: the key
    scores and the calibration stay by version."""
    withdrawn = {row[0] for row in log.execute("SELECT publication_id FROM withdrawals")}
    void_matches = {row[0] for row in log.execute("SELECT match_id FROM voids"
                                                  " WHERE quantity_key IS NULL")}  # fmt: skip
    # (group, key) -> metric -> [(own, reference)], on the forecasts that have a reference
    scored = defaultdict(lambda: defaultdict(list))
    lines = defaultdict(list)
    tallies = defaultdict(lambda: {"void": 0, "withdrawn": 0, "unmatched": 0})
    for kickoff, key, version in log.execute(
            "SELECT MAX(p.scheduled_kickoff_utc), d.quantity_key, d.model_version"
            " FROM distributions d JOIN publications p ON p.publication_id = d.publication_id"
            " WHERE p.stage = 'pre_match' AND p.match_id IS NULL"
            " AND p.match_reference IS NOT NULL"
            " GROUP BY p.match_reference, d.quantity_key, d.subject_id,"
            " d.model_version"):  # fmt: skip
        if key in definitions:
            probability_kind = definitions[key].quantity.probability_kind
            for period in (f"month:{kickoff[:7]}", "all"):
                for counted in (version, ALL_VERSIONS):
                    tallies[(probability_kind, counted, 0, period)]["unmatched"] += 1
    for match_id, key, subject, publication_id, version, window, values_text, kickoff in records:
        row = state.get(match_id)
        if row is None:
            continue
        kind = definitions[key].quantity.distribution_kind
        probability_kind = definitions[key].quantity.probability_kind
        periods = (f"season:{row['season']}", f"month:{kickoff[:7]}", "all")
        groups = [(probability_kind, counted, competition, period)
                  for counted in (version, ALL_VERSIONS)
                  for competition in (row["competition_id"], 0) for period in periods]
        if match_id in void_matches or publication_id in withdrawn:
            for group in groups:
                tallies[group]["void" if match_id in void_matches else "withdrawn"] += 1
            continue
        if (match_id, key, subject) not in happened_of:
            continue
        happened = happened_of[(match_id, key, subject)]
        values = json.loads(values_text)
        side = ("match" if subject == match_id else
                "home" if subject == row["home_team_id"] else "away")  # fmt: skip
        reference = references.get((row["competition_id"], row["season"]), {}).get((key, side))
        try:
            own = score(kind, values, happened)
            naive = score(kind, reference, happened) if reference is not None else None
        except OutcomeError:
            continue
        if naive is not None:
            for group in groups:
                for metric, value in own.items():
                    scored[(group, key)][metric].append((value, naive[metric]))
        for (line, probability) in log.execute(
                "SELECT line_key, probability FROM probabilities WHERE publication_id = ? AND"
                " quantity_key = ? AND subject_id = ? AND window_key = ?",
                (publication_id, key, subject, window)):  # fmt: skip
            for group in groups:
                if group[1] == ALL_VERSIONS:
                    continue
                lines[group].append((probability, _has_line_happened(kind, values, line, happened)))
    key_scores, skills = [], defaultdict(list)
    for (group, key), metrics in scored.items():
        deciding = METRICS_BY_KIND[definitions[key].quantity.distribution_kind][0]
        for metric, pairs in sorted(metrics.items()):
            value = sum(own for own, _ in pairs) / len(pairs)
            baseline = sum(naive for _, naive in pairs) / len(pairs)
            skill = 1.0 - value / baseline if baseline > 0 else 0.0
            if group[1] != ALL_VERSIONS:
                key_scores.append(KeyScore(group[0], key, group[1], group[2], group[3], metric,
                                           value, baseline, skill, len(pairs)))  # fmt: skip
            if metric == deciding:
                skills[group].append((skill, len(pairs)))
    scores = []
    # every group of the round gets its row, so a group whose forecasts all became void says so
    # (count 0) instead of leaving an older round's figures as the latest
    for group in sorted(set(skills) | set(tallies)):
        per_key, tally = skills.get(group, []), tallies[group]
        scores.append(TrackRecordScore(
            probability_kind=group[0], model_version=group[1], competition_id=group[2],
            period=group[3], metric=SKILL,
            value=sum(skill for skill, _ in per_key) / len(per_key) if per_key else 0.0,
            baseline=BASELINE, baseline_value=0.0, count=sum(count for _, count in per_key),
            void_count=tally["void"], withdrawn_count=tally["withdrawn"],
            unmatched_count=tally["unmatched"]))  # fmt: skip
    if key_scores:
        record_key_scores(log, key_scores, now)
    if scores:
        record_scores(log, scores, now)
    for group, group_lines in lines.items():
        record_calibration(
            log, group[0], group[1], group[2], group[3], bucket_lines(group_lines), now
        )
    return len(scores) + len(key_scores) + len(lines)


def read_team_count_size(count: str) -> int:
    config = tomllib.loads((ROOT / "config/vesta/team_counts.toml").read_text(encoding="utf-8"))
    return int(config["counts"][count]["size"])


def _describe_sources(key: str) -> str:
    family = key.split(".")[0]
    if family in ("L1", "L2", "L6", "L9"):
        return "match_players, match_team_sheets"
    if family == "P14":
        return "shots (tracked, the opponent's saved shots in his minutes), match_players"
    if family in ("P1", "P2", "P7", "P8", "P9", "P16"):
        return "player_match_statistics (the feed), match_players"
    if family in ("P3", "P5"):
        return "match_events, shots, player_match_statistics, match_players"
    if family in ("M15", "M17", "M19", "M20", "M21"):
        return "match_team_statistics (leagues: the report, else the feed; cups: the feed)"
    spine = family in ("M1", "M2", "M3", "M4", "M5")
    return "matches" if spine else "matches, match_events, shots"


def _read_last_outcomes(log) -> dict:
    last = {}
    for key, subject, window, match_id, value in log.execute(
            "SELECT quantity_key, subject_id, window_key, match_id, outcome_value FROM outcomes"
            " WHERE match_id IS NOT NULL ORDER BY outcome_id"):  # fmt: skip
        last[(key, subject, window, match_id)] = json.loads(value)
    return last


def _read_last_checks(log) -> dict:
    last = {}
    for publication_id, key, subject, line, status, happened in log.execute(
            "SELECT publication_id, quantity_key, subject_id, line_key, status, happened"
            " FROM line_checks ORDER BY check_id"):  # fmt: skip
        last[(publication_id, key, subject, line)] = (status, None if happened is None
                                                      else bool(happened))  # fmt: skip
    return last


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("world", type=Path)
    parser.add_argument("log", type=Path)
    parser.add_argument("--now", default=None)
    options = parser.parse_args(arguments)
    run(options.world, options.log, options.now or format_now_utc())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
