"""The nations' tournaments verdict (T6.round_reached, T6.group_place.team).

    python -m vesta.season.nations_verdict <football-world.db> [--stage development|blind]
                                           [--draws N] [--out report.md] [--key KEY ...]

The plan: the season family's nations plan (eba056d2, amendments e365884e
and 76b181b1, approved), the alarm `[alarm.nations_progression]`. Each tournament the
stage's years hold, with its format and tie-break rule verified, is simulated from what was known
at each checkpoint (G0 the eve of its first match; G1 every team's first match played; G2 the
groups over; K1 the first knockout round played), by `vesta.season.cup_progression` on the
tournament's format:

- **Vesta:** the spine's nations model at the checkpoint (`vesta.season.strength`). A host plays
  at home in its own country: the world's venue and neutral flag for the group matches, and in a
  drawn knockout match where the whole tournament is in its country.
- **The reference (deciding), the Elo-thirds history:** a team's outcome shares among the teams
  still in, in the same third by Elo within the tournament at the same checkpoint, in the earlier
  editions of its competition (the editions the plan names, ended before the checkpoint). A cell
  under 30 team-tournaments pools the six competitions. The outcomes are matched by their distance
  from the title (the winner, the final, the semi-finals, ...), the deeper ones a format lacks
  folded into its group stage. Only the outcomes the team can still reach are kept.
- **Equal strength (information):** every pair the mean score of the six competitions' earlier
  final tournaments.

Each team whose outcome the facts have not settled is scored by the log score of what happened;
the rule, the blocks (the tournaments), the calibration and the alarm are those of the cups
verdict (`vesta.season.cups_verdict.judge`).
"""

import argparse
import hashlib
import sqlite3
import sys
import time
import tomllib
from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from vesta.research.calibration import measure_calibration_error
from vesta.season.cup_progression import (
    ROUND_OF_STAGE,
    build_phase_tables,
    find_outcomes_reached,
    rank_final_phase,
    read_cup_season,
    simulate_cup,
)
from vesta.season.cups_strength import MeanScore, build_equal_strength
from vesta.season.cups_verdict import Row, judge
from vesta.season.league import place_bounds
from vesta.season.strength import build_nations_strength
from vesta.spine.nations import NationsParameters, find_current_ratings
from vesta.spine.score_matrix import SIZE, sum_result_probabilities
from vesta.spine.ties import TieParameters
from vesta.spine.world import open_world, read_nation_matches

ROOT = Path(__file__).resolve().parents[3]
ROUND, PLACE = "T6.round_reached", "T6.group_place.team"
TOURNAMENTS = {
    "development": (("WC", 2022), ("EURO", 2020), ("COPA", 2019), ("COPA", 2020),
                    ("AFCON", 2019), ("AFCON", 2021), ("AFCON", 2023), ("ASIAN", 2019),
                    ("ASIAN", 2023), ("GOLD", 2019), ("GOLD", 2021), ("GOLD", 2023)),
    "blind": (("EURO", 2024), ("COPA", 2024), ("AFCON", 2025), ("GOLD", 2025)),
}
HISTORY_FROM = {"WC": 2006, "EURO": 2016, "AFCON": 2019, "ASIAN": 2019, "COPA": 2015, "GOLD": 2013}
CHECKPOINTS = ("G0", "G1", "G2", "K1")
PLACE_CHECKPOINTS = ("G0", "G1")
THIN = 30
# the venues' country names that differ from the national team's name
COUNTRY_OF_VENUE = {"USA": "United States"}
KNOCKOUT_ORDER = ("round_of_32", "round_of_16", "quarter_finals", "semi_finals", "final")


# ---- an edition read for the history ------------------------------------------------------


@dataclass(frozen=True)
class Edition:
    """An edition as the reference reads it: its teams, their deepest outcome as a distance from
    the title (0 the winner, 1 the final, ...; the group stage the last), and its checkpoints."""

    competition: str
    season: int
    teams: tuple[int, ...]
    groups: dict[str, tuple[int, ...]]
    place: dict[int, int]  # the final place in its group (0 the first), by points, goal
    # difference and goals (the edition's own tie-breaks not read: the history's approximation)
    distance: dict[int, int]
    alive: dict[str, frozenset[int]]  # checkpoint -> the teams still in
    instants: dict[str, str]


def read_edition(world: sqlite3.Connection, competition: str, season: int) -> Edition | None:
    rows = world.execute(
        """
        SELECT LOWER(COALESCE(m.stage, '')), m.kickoff_utc, m.home_team_id, m.away_team_id,
               m.home_goals, m.away_goals, m.winner_team_id, COALESCE(m.group_name, '')
        FROM matches m JOIN seasons s ON s.season_id = m.season_id
        JOIN competitions c ON c.competition_id = s.competition_id
        WHERE c.code = ? AND s.season = ? AND m.status IN ('played', 'awarded')
          AND m.home_goals IS NOT NULL
        ORDER BY m.kickoff_utc
        """,
        (competition, season),
    ).fetchall()
    group = [row for row in rows if row[0].startswith("group")]
    knockout = [row for row in rows if row[0] in ROUND_OF_STAGE]
    if not group or not knockout:
        return None
    teams = tuple(sorted({team for row in group for team in row[2:4]}))
    members: dict[str, set[int]] = {}
    table: dict[int, list[int]] = {team: [0, 0, 0] for team in teams}
    for _, _, home, away, home_goals, away_goals, _, name in group:
        members.setdefault(name, set()).update((home, away))
        for team, scored, conceded in ((home, home_goals, away_goals),
                                       (away, away_goals, home_goals)):  # fmt: skip
            table[team][0] += 3 if scored > conceded else int(scored == conceded)
            table[team][1] += scored - conceded
            table[team][2] += scored
    groups = {name: tuple(sorted(clubs)) for name, clubs in members.items()}
    place = {team: index for clubs in groups.values()
             for index, team in enumerate(sorted(
                 clubs, key=lambda team_id: [-value for value in table[team_id]]))}
    rounds = [name for name in KNOCKOUT_ORDER
              if any(ROUND_OF_STAGE[row[0]] == name for row in knockout)]  # fmt: skip
    depth = dict.fromkeys(teams, 0)
    for row in knockout:
        for team in row[2:4]:
            depth[team] = max(depth.get(team, 0), rounds.index(ROUND_OF_STAGE[row[0]]) + 1)
    final = [row for row in knockout if ROUND_OF_STAGE[row[0]] == "final"]
    if final:
        stage, _, home, away, home_goals, away_goals, winner, _ = final[-1]
        winner = winner or (home if home_goals > away_goals else away)
        depth[winner] = len(rounds) + 1
    distance = {team: len(rounds) + 1 - value for team, value in depth.items()}
    first_round = [row for row in knockout if ROUND_OF_STAGE[row[0]] == rounds[0]]
    second = {team for row in knockout if ROUND_OF_STAGE[row[0]] != rounds[0] for team in row[2:4]}
    instants = find_edition_instants([(row[1], row[2], row[3]) for row in group],
                                     [row[1] for row in first_round])  # fmt: skip
    alive = {"G0": frozenset(teams), "G1": frozenset(teams),
             "G2": frozenset(team for row in first_round for team in row[2:4]),
             "K1": frozenset(second)}  # fmt: skip
    return Edition(competition, season, teams, groups, place, distance, alive, instants)


def find_edition_instants(
    group: list[tuple[str, int, int]], first_round: list[str]
) -> dict[str, str]:
    """The checkpoints' instants from the group matches (kick-off, home, away) and the first
    knockout round's kick-offs."""
    first: dict[int, str] = {}
    for kickoff, home, away in group:
        for team in (home, away):
            first.setdefault(team, kickoff[:10])
    days = sorted(kickoff[:10] for kickoff, _, _ in group)
    return {"G0": f"{days[0]}T00:00:00Z", "G1": _add_one_day(max(first.values())),
            "G2": _add_one_day(days[-1]), "K1": _add_one_day(max(day[:10] for day in first_round))}


def _add_one_day(day: str) -> str:
    return f"{(date.fromisoformat(day) + timedelta(days=1)).isoformat()}T00:00:00Z"


# ---- the reference -------------------------------------------------------------------------


def split_thirds_by_rating(
    teams: list[int], rating: dict[int, float], first: float
) -> dict[int, int]:
    """0 the top third by Elo, 1 the middle, 2 the bottom."""
    ranked = sorted(teams, key=lambda team: -rating.get(team, first))
    size = len(ranked)
    return {team: min(2, 3 * index // size) for index, team in enumerate(ranked)}


@dataclass(frozen=True)
class Seen:
    """A team of an earlier edition at a checkpoint: its Elo third among the teams still in
    (-1 when out), its Elo rank in its group, and what it reached."""

    competition: str
    ended: str  # the edition's last checkpoint: read only by checkpoints after it
    checkpoint: str
    third: int
    group_rank: int
    distance: int
    place: int


def list_seen_teams(editions: list[Edition], nations, parameters: NationsParameters) -> list[Seen]:
    """Every team of every edition at each checkpoint, with the Elo as it stood then."""
    found = []
    for edition in editions:
        for checkpoint in CHECKPOINTS:
            instant = edition.instants[checkpoint]
            rating, _ = find_current_ratings([match for match in nations
                                              if match.local_date < instant[:10]], parameters)
            first = parameters.first_rating
            thirds = split_thirds_by_rating(sorted(edition.alive[checkpoint]), rating, first)
            ranks = {team: index for clubs in edition.groups.values()
                     for index, team in enumerate(sorted(
                         clubs, key=lambda team_id: -rating.get(team_id, first)))}
            for team in edition.teams:
                found.append(Seen(edition.competition, max(edition.instants.values()),
                                  checkpoint, thirds.get(team, -1), ranks[team],
                                  edition.distance[team], edition.place[team]))  # fmt: skip
    return found


@dataclass
class History:
    """Per (competition, checkpoint, cell) the outcomes of the earlier editions: the cell an Elo
    third for the round reached, an Elo rank in the group for the place."""

    cells: dict[tuple[str, str, int], Counter]


def build_history(seen: list[Seen], before: str, what: str) -> History:
    """The editions that ended before `before`: `what` is "round" (the distance from the title,
    by third, the teams still in) or "place" (the final group place, by Elo rank in the group)."""
    cells: dict[tuple[str, str, int], Counter] = {}
    for item in seen:
        if item.ended >= before:
            continue
        if what == "round":
            if item.third < 0:
                continue
            cells.setdefault((item.competition, item.checkpoint, item.third), Counter())[
                item.distance] += 1
        else:
            cells.setdefault((item.competition, item.checkpoint, item.group_rank), Counter())[
                item.place] += 1
    return History(cells)


def place_shares(past: History, competition: str, checkpoint: str, rank: int, size: int,
                 pooled: list) -> list[float]:  # fmt: skip
    """The final group place's shares of the teams of the same Elo rank in their group at the
    same checkpoint, pooled across the competitions when thin; half a case added to each."""
    counts = past.cells.get((competition, checkpoint, rank), Counter())
    if sum(counts.values()) < THIN:
        own = sum(counts.values())
        counts = Counter()
        for (_, cell_checkpoint, cell_rank), cell in past.cells.items():
            if cell_checkpoint == checkpoint and cell_rank == rank:
                counts.update(cell)
        pooled.append((f"{competition} place", checkpoint, rank, own, sum(counts.values())))
    shares = np.array([counts.get(place, 0) for place in range(size)], dtype=float) + 0.5
    return (shares / shares.sum()).tolist()


def pool_reference_shares(past: History, competition: str, checkpoint: str, third: int,
                          outcomes: int, lowest: int, pooled: list) -> list[float]:  # fmt: skip
    """The shares of the format's outcomes (index 0 the group stage, outcomes-1 the winner) from
    the history's cell, pooled across the competitions when thin; the outcomes below `lowest`
    are impossible and dropped."""
    own = past.cells.get((competition, checkpoint, third), Counter())
    counts = own
    if sum(own.values()) < THIN:
        counts = Counter()
        for (_, cell_checkpoint, cell_third), cell in past.cells.items():
            if cell_checkpoint == checkpoint and cell_third == third:
                counts.update(cell)
        pooled.append((competition, checkpoint, third, sum(own.values()), sum(counts.values())))
    shares = np.zeros(outcomes)
    for distance, count in counts.items():
        shares[max(outcomes - 1 - distance, 0)] += count  # deeper than the format: its group stage
    shares[:lowest] = 0.0
    if shares.sum() == 0:
        shares[lowest:] = 1.0  # uniform over what can still happen
    return (shares / shares.sum()).tolist()


# ---- one run -------------------------------------------------------------------------------


def run(world_path: Path, stage: str, draws: int, say=print) -> dict:
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    research = tomllib.loads((ROOT / "config/vesta/research.toml").read_text(encoding="utf-8"))
    seed = int(research["verdict"]["seed"])
    nations_parameters = NationsParameters.from_config(spine)
    ties = TieParameters.from_config(spine)
    rows: list[Row] = []
    refused, notes, pooled = [], [], []
    calibration_lines: list[tuple[float, bool]] = []
    with open_world(world_path) as world:
        nations = read_nation_matches(world)
        editions = [edition for competition, first in HISTORY_FROM.items()
                    for (season,) in world.execute(
                        "SELECT s.season FROM seasons s JOIN competitions c USING (competition_id)"
                        " WHERE c.code = ? AND s.season >= ? ORDER BY s.season",
                        (competition, first))
                    if (edition := read_edition(world, competition, season)) is not None]
        cups = {}
        for competition, season in TOURNAMENTS[stage]:
            try:
                cups[(competition, season)] = read_cup_season(world, competition, season)
            except ValueError as error:
                refused.append(f"{competition} {season}: {error} - not judged")
        corrections = {}
        for key in list(cups):
            cups[key], corrections[key] = set_grounds_by_venue(world, cups[key])
        hosts = {key: _read_host(world, *key) for key in cups}
        multi_host = {key: _read_host_home_knockouts(world, *key) for key in cups}
        notes += [check_host(world, *key) + f"; {corrections[key]} group matches set by it"
                  for key in sorted(cups)]
    seen = list_seen_teams(editions, nations, nations_parameters)
    say(f"{len(editions)} editions read for the reference")
    for index, ((competition, season), cup) in enumerate(sorted(cups.items())):
        if not cup.table_rule.verified:
            refused.append(f"{competition} {season}: its tie-break rule is not verified")
            continue
        order = rank_final_phase(cup, seed)
        happened = find_outcomes_reached(cup, order)
        final_place = {team: place for teams in order.values()
                       for place, team in enumerate(teams)}  # fmt: skip
        edition = read_edition_from(cup)
        mean = _compute_mean_score(nations, cup)
        reference_strength = build_equal_strength(mean)
        for checkpoint in CHECKPOINTS:
            instant = edition[checkpoint]
            started = time.monotonic()
            strength = build_nations_strength(nations, nations_parameters, instant[:10],
                                              hosts[(competition, season)])  # fmt: skip
            run_seed = seed + 1000 * season + 10 * index + CHECKPOINTS.index(checkpoint)
            simulated = {method: simulate_cup(cup, None, instant, chosen, ties, draws, run_seed)
                         for method, chosen in (("vesta", strength),
                                                ("equal", reference_strength))}  # fmt: skip
            past = build_history(seen, instant, "round")
            past_places = build_history(seen, instant, "place")
            tables = build_phase_tables(cup, instant, reference_strength)
            rating, _ = find_current_ratings(
                [match for match in nations if match.local_date < instant[:10]], nations_parameters
            )
            alive = _find_alive_teams(cup, checkpoint, tables, happened)
            outcomes = len(cup.form.outcomes)
            thirds = split_thirds_by_rating(sorted(alive), rating, nations_parameters.first_rating)
            lowest = {"G0": 0, "G1": 0, "G2": 1, "K1": 2}[checkpoint]
            for team in sorted(alive):
                distributions = {method: _count_shares(result.reached[team], outcomes)
                                 for method, result in simulated.items()}  # fmt: skip
                distributions["reference"] = pool_reference_shares(
                    past, competition, checkpoint, thirds[team], outcomes, lowest, pooled)
                rows.append(Row(ROUND, competition, season, checkpoint, team, happened[team],
                                distributions))  # fmt: skip
            if checkpoint in PLACE_CHECKPOINTS:
                rows += _group_places(competition, season, checkpoint, tables, simulated,
                                      final_place, past_places, rating,
                                      nations_parameters.first_rating, pooled)  # fmt: skip
            calibration_lines += _collect_match_calibration(
                cup, instant, edition, checkpoint, strength
            )
            say(f"{competition} {season} {checkpoint} {instant}: {len(alive)} teams in,"
                f" {time.monotonic() - started:.0f} s")  # fmt: skip
        notes += multi_host[(competition, season)]
    counts = Counter()
    for key, cell in sorted(build_history(seen, "9999", "round").cells.items()):
        counts[key] = sum(cell.values())
    return {"stage": stage, "seasons": sorted({season for _, season in cups}), "rows": rows,
            "notes": notes, "refused": refused, "research": research, "draws": draws,
            "pooled": pooled, "history_counts": counts,
            "nations_calibration": (measure_calibration_error(calibration_lines)
                                    if calibration_lines else None, len(calibration_lines) // 3)}


def read_edition_from(cup) -> dict[str, str]:
    group = [(match.kickoff, match.home, match.away) for match in cup.phase]
    first_round = [match.kickoff for match in cup.knockout
                   if match.round_name == cup.form.rounds[0][0]]  # fmt: skip
    return find_edition_instants(group, first_round)


def _find_alive_teams(cup, checkpoint, tables, happened) -> set[int]:
    """The teams whose outcome the facts at the checkpoint leave open."""
    teams = {team for table in tables.values() for team in table.team_ids}
    if checkpoint in ("G0", "G1"):
        going_on = max(high for _, high in cup.form.advance.values())
        alive = set()
        for table in tables.values():
            best, _ = place_bounds(table)
            alive |= {team for column, team in enumerate(table.team_ids)
                      if best[column] <= going_on}  # fmt: skip
        return alive
    if checkpoint == "G2":
        return {team for team in teams if happened[team] >= 1}
    return {team for team in teams if happened[team] >= 2}


def _group_places(competition, season, checkpoint, tables, simulated, final_place, past, rating,
                  first, pooled) -> list[Row]:  # fmt: skip
    rows = []
    for table in tables.values():
        best, worst = place_bounds(table)
        size = len(table.team_ids)
        ranks = {team: index for index, team in enumerate(
            sorted(table.team_ids, key=lambda team_id: -rating.get(team_id, first)))}  # fmt: skip
        for column, team in enumerate(table.team_ids):
            if best[column] == worst[column]:
                continue
            distributions = {method: _count_shares(result.places[team] - 1, size)
                             for method, result in simulated.items()}  # fmt: skip
            distributions["reference"] = place_shares(past, competition, checkpoint,
                                                      ranks[team], size, pooled)  # fmt: skip
            rows.append(Row(PLACE, competition, season, checkpoint, team, final_place[team],
                            distributions))  # fmt: skip
    return rows


def _count_shares(values: np.ndarray, count: int) -> list[float]:
    return (np.bincount(values, minlength=count)[:count] / len(values)).tolist()


def _read_host(world: sqlite3.Connection, competition: str, season: int) -> frozenset[int]:
    """The host where every match of the tournament is in its country: at home on neutral
    ground in a drawn knockout match."""
    countries = {country for (country,) in world.execute(
        "SELECT DISTINCT v.country FROM matches m JOIN seasons s USING (season_id)"
        " JOIN competitions c USING (competition_id) LEFT JOIN venues v USING (venue_id)"
        " WHERE c.code = ? AND s.season = ?", (competition, season))}  # fmt: skip
    if len(countries) != 1 or None in countries:
        return frozenset()
    rows = world.execute("SELECT team_id FROM teams WHERE kind = 'nation' AND name = ?",
                         (_normalise_country(next(iter(countries))),)).fetchall()  # fmt: skip
    return frozenset(row[0] for row in rows) if len(rows) == 1 else frozenset()


def _normalise_country(name: str | None) -> str | None:
    return COUNTRY_OF_VENUE.get(name, name)


def set_grounds_by_venue(world: sqlite3.Connection, cup) -> tuple[object, int]:
    """The tournament's group matches with their ground set by the venue's country: at home for a
    team in its own country (the sides turned when it is the second-named), neutral otherwise;
    how many matches this changed."""
    venue = {
        match_id: (home, away, _normalise_country(country))
        for match_id, home, away, country in world.execute(
            "SELECT m.match_id, home.name, away.name, v.country FROM matches m"
            " JOIN teams home ON home.team_id = m.home_team_id"
            " JOIN teams away ON away.team_id = m.away_team_id"
            " LEFT JOIN venues v USING (venue_id)"
            f" WHERE m.match_id IN ({', '.join(str(m.match_id) for m in cup.phase)})"
        )
    }
    changed = 0
    phase = []
    for match in cup.phase:
        home, away, country = venue[match.match_id]
        if country is not None and away == country and home != country:
            match = replace(match, home=match.away, away=match.home, neutral=False,
                            home_goals=match.away_goals, away_goals=match.home_goals)
            changed += 1
        elif country is not None and (home == country) == match.neutral:
            match = replace(match, neutral=home != country)
            changed += 1
        phase.append(match)
    return replace(cup, phase=tuple(phase)), changed


def check_host(world: sqlite3.Connection, competition: str, season: int) -> str:
    """The plan's test of the host by the venue's country: a match is off neutral ground exactly
    when its home side plays in its own country (the world's `neutral_venue` against
    `venues.country`)."""
    rows = world.execute(
        "SELECT home.name, v.country, m.neutral_venue FROM matches m"
        " JOIN seasons s USING (season_id) JOIN competitions c USING (competition_id)"
        " JOIN teams home ON home.team_id = m.home_team_id LEFT JOIN venues v USING (venue_id)"
        " WHERE c.code = ? AND s.season = ?", (competition, season)).fetchall()  # fmt: skip
    wrong = [(home, country, neutral) for home, country, neutral in rows
             if bool(neutral) == (home == _normalise_country(country))]  # fmt: skip
    missing = sum(1 for _, country, _ in rows if country is None)
    return (f"{competition} {season}: the host test - {len(rows)} matches, {len(wrong)} whose"
            f" neutral flag disagrees with the venue's country, {missing} without a venue"
            + (": " + ", ".join(f"{home} in {country} (neutral {neutral})"
                                for home, country, neutral in wrong[:5]) if wrong else ""))


def _read_host_home_knockouts(world, competition, season) -> list[str]:
    """For a tournament over several countries: its knockout matches a host played at home."""
    rows = world.execute(
        "SELECT LOWER(m.stage), home.name, away.name, v.country FROM matches m"
        " JOIN seasons s USING (season_id) JOIN competitions c USING (competition_id)"
        " JOIN teams home ON home.team_id = m.home_team_id"
        " JOIN teams away ON away.team_id = m.away_team_id"
        " LEFT JOIN venues v USING (venue_id) WHERE c.code = ? AND s.season = ?",
        (competition, season)).fetchall()  # fmt: skip
    countries = {country for *_, country in rows}
    if len(countries) <= 1:
        return []
    knockout = [row for row in rows if row[0] in ROUND_OF_STAGE]
    at_home = [row for row in knockout if _normalise_country(row[3]) in (row[1], row[2])]
    named = ", ".join(f"{home} v {away} in {country}" for _, home, away, country in at_home)
    return [f"{competition} {season}: played over {len(countries)} countries; {len(at_home)} of"
            f" its {len(knockout)} knockout matches had a host at home ({named}),"
            " simulated on neutral ground when drawn"]


def _compute_mean_score(nations, cup) -> MeanScore:
    """The mean score of the six competitions' final tournaments before this one."""
    first = min(match.kickoff for match in cup.phase)[:10]
    chosen = [match for match in nations if match.competition in HISTORY_FROM
              and match.competition_kind == "nations tournament" and match.local_date < first]
    counts = np.zeros((SIZE, SIZE))
    for match in chosen:
        counts[min(match.home_goals, SIZE - 1), min(match.away_goals, SIZE - 1)] += 1
    return MeanScore(counts / counts.sum(), float(np.mean([m.home_goals for m in chosen])),
                     float(np.mean([m.away_goals for m in chosen])), len(chosen))  # fmt: skip


def _collect_match_calibration(
    cup, instant, instants, checkpoint, strength
) -> list[tuple[float, bool]]:
    """The nations model's own result probabilities for the tournament's matches until the next
    checkpoint (after K1, to the end), as information: the weak part T6 rests on."""
    following = CHECKPOINTS.index(checkpoint) + 1
    until = instants[CHECKPOINTS[following]] if following < len(CHECKPOINTS) else "9999"
    lines = []
    for match in (*cup.phase, *cup.knockout):
        if not (instant <= match.kickoff < until) or not match.played:
            continue
        matrix = strength(match.home, match.away, match.neutral)[0]
        probabilities = sum_result_probabilities(np.array([matrix]))[0]
        result = (0 if match.home_goals > match.away_goals
                  else 1 if match.home_goals == match.away_goals else 2)  # fmt: skip
        lines += [(float(probability), index == result)
                  for index, probability in enumerate(probabilities)]  # fmt: skip
    return lines


# ---- the report ----------------------------------------------------------------------------


def report(results: dict, judged: dict, world_path: Path, world_sha: str, commit: str,
           seconds: float) -> str:  # fmt: skip
    stage = results["stage"]
    calibration, matches = results["nations_calibration"]
    lines = [f"# The nations' tournaments: the {stage} verdict (T6)", "",
             f"World: `{world_path}` (sha256 {world_sha}). Code: `{commit}`. Tournaments:"
             f" {len({(row.code, row.season) for row in results['rows']})}; {results['draws']}"
             " simulations per tournament, checkpoint and method. Written by `python -m"
             " vesta.season.nations_verdict`; the plan"
             " declared before the run (eba056d2, amendments"
             " e365884e and 76b181b1), the alarm `[alarm.nations_progression]`."
             f" {seconds:.0f} seconds.", "",
             "The strength is the spine's nations model, which has not passed its own blind"
             " verdict (calibration 0.0112 against 0.0039, 28 October): a pass here is released"
             " only beside a released nations model, and T6 is run again if that model changes."
             + (f" Its calibration on these tournaments' matches (expected calibration error of"
                f" the result's three outcomes, {matches} matches, information): {calibration:.4f}."
                if calibration is not None else ""), ""]  # fmt: skip
    for key, result in judged.items():
        mean, low, high = result["difference"]
        lines += [f"## {key}: {'PASSED' if result['passed'] else 'NOT PASSED'}", "",
                  f"- {result['n']} (team, checkpoint) scored, in {result['blocks']} blocks (the"
                  " tournaments): the interval is as coarse as that.",
                  f"- Mean log score: Vesta {result['scores']['vesta']:.4f}, reference"
                  f" {result['scores']['reference']:.4f}, equal strength (information)"
                  f" {result['scores']['frozen']:.4f}.",
                  f"- Vesta minus the reference: {mean:+.4f} [{low:+.4f}; {high:+.4f}].",
                  f"- Years improved: {result['improved']} of {result['seasons']} (needed"
                  f" {result['needed']}).",
                  f"- Calibration of the lines: Vesta {result['calibration']['vesta']:.4f},"
                  f" reference {result['calibration']['reference']:.4f}.",
                  "- Vesta minus equal strength (information): {:+.4f} [{:+.4f}; {:+.4f}].".format(
                      *result["frozen_difference"])]  # fmt: skip
        lines += [f"- ALARM: {stop}" for stop in result["stops"]]
        lines += [f"- Warning: {warning}" for warning in result["warnings"]]
        lines += ["", "| checkpoint | n | Vesta | reference | equal | entropy | gap (SE) |",
                  "|---|---|---|---|---|---|---|"]
        for checkpoint, row in result["per_checkpoint"].items():
            lines.append(f"| {checkpoint} | {row['n']} | {row['vesta']:.4f} |"
                         f" {row['reference']:.4f} | {row['frozen']:.4f} | {row['entropy']:.4f} |"
                         f" {row['standard_errors']:+.2f} |")
        lines.append("")
    lines += ["## The reference's history", "",
              "Earlier team-tournaments per competition, checkpoint and third (all editions"
              " read); a cell under 30 for a tournament pools the six competitions:", "",
              "| competition | checkpoint | top | middle | bottom |", "|---|---|---|---|---|"]
    counts = results["history_counts"]
    for competition in HISTORY_FROM:
        for checkpoint in CHECKPOINTS:
            cells = [counts.get((competition, checkpoint, third), 0) for third in range(3)]
            lines.append(f"| {competition} | {checkpoint} | "
                         + " | ".join(f"{count}{' (thin)' if count < THIN else ''}"
                                      for count in cells) + " |")
    used = sorted({(competition, checkpoint, third, own, pool)
                   for competition, checkpoint, third, own, pool in results["pooled"]})  # fmt: skip
    lines += ["", f"Cells that used the pool ({len(used)}): " + "; ".join(
        f"{competition} {checkpoint} third {third + 1}: own {own}, pooled {pool}"
        for competition, checkpoint, third, own, pool in used), ""]  # fmt: skip
    judged_tournaments = sorted({(row.code, row.season) for row in results["rows"]})
    lines += ["## The tournaments", "",
              f"Judged ({len(judged_tournaments)} of the {len(TOURNAMENTS[stage])} the plan"
              f" lists for this stage): "
              + ", ".join(f"{code} {season}" for code, season in judged_tournaments) + ".", ""]
    if ("ASIAN", 2023) in judged_tournaments:
        lines += ["The Asian Cup 2023's order of the best thirds rests on the AFC Manual of 2026"
                  " (Appendix 2), the 2023 Manual not read.", ""]
    if results["refused"]:
        lines += ["## Not judged", ""] + [f"- {reason}" for reason in results["refused"]] + [""]
    lines += ["## Notes", ""] + [f"- {note}" for note in results["notes"]]
    lines += ["- The group place's reference: the final group places of the earlier editions'"
              " teams of the same Elo rank in their group at the same checkpoint (half a case"
              " added to each place); those editions' places ranked by points, goal difference"
              " and goals, their own tie-breaks not read."]
    return "\n".join(lines)


def main(arguments: list[str]) -> int:
    from vesta.research.repository import run_git

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("world", type=Path)
    parser.add_argument("--stage", choices=("development", "blind"), default="development")
    parser.add_argument("--draws", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--key", action="append", choices=(ROUND, PLACE), default=None,
                        help="judge only these keys (blind: only the keys development passed)")
    options = parser.parse_args(arguments)
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    draws = options.draws or int(spine["season"]["simulations"])
    started = time.monotonic()
    commit = run_git(ROOT, "rev-parse", "HEAD").strip()
    results = run(options.world, options.stage, draws)
    keys = tuple(options.key or (ROUND, PLACE))
    judged = judge(results, keys=keys, alarm_name="nations_progression",
                   information="equal", groups=tuple(HISTORY_FROM))  # fmt: skip
    digest = hashlib.sha256()
    with options.world.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 24), b""):
            digest.update(chunk)
    text = report(results, judged, options.world, digest.hexdigest(), commit,
                  time.monotonic() - started)  # fmt: skip
    out = options.out or (ROOT / "docs/research"
                          / f"{date.today().isoformat()}-season-family-nations-{options.stage}.md")
    out.write_text(text + "\n", encoding="utf-8")
    for key, result in judged.items():
        print(key, "PASSED" if result["passed"] else "NOT PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
