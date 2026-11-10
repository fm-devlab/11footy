"""What lineups publish before a match: each player's chance to start (`L1.starts`), his minutes
(`L6.minutes`), his position if he starts (`L9.position`), the team's formation
(`L2.formation.team`), and the expected eleven (`expected_elevens`), from the model that passed its
verdicts (development and blind).

The model's parts of a season (the weights of the chance to start, the groups' priors, the
leagues' formations, each formation's slot positions) come from its five seasons before and do
not change during it, so they are built once a season and kept in the cache folder
(`lineups-priors-<season>.json`, with the release's identity: a new release builds them again).
Built season by season, one season's rows at a time, so the morning job stays under its memory
cap. The outcomes of the formation and the position are the quantities' lines
(`config/vesta/quantities.csv`), an outcome outside them counted as `other`.
"""

import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from vesta.forecast_log.writing import ExpectedSlot, PublishedDistribution
from vesta.lineups.model import (
    FADE,
    HISTORY,
    MINUTE_CELLS,
    OTHER,
    TeamMatch,
    build_histogram,
    build_player_rows,
    collect_targets,
    fit_logistic,
    read_world,
    scale_to_eleven,
    shrink_to_prior,
)
from vesta.quantities import load_quantities
from vesta.spine.dixon_coles import parse_instant_to_seconds
from vesta.spine.timing import TimeRules

TRAINING_SEASONS = 5
SMOOTHING = 0.5
STRENGTH = {"came_on": 3.0, "minutes": 3.0, "position": 1.0, "formation": 2.0}
EPSILON = 1e-12


def list_outcomes(key: str) -> list[str]:
    """The outcomes of a categorical lineups key, from its lines (`outcome=<name>`)."""
    return [line.split("=", 1)[1] for line in load_quantities()[key].quantity.lines]


@dataclass
class ModelPriors:
    """The model's parts of a season (the verdict's `Priors` without the references)."""

    weights: list[float]
    came_on: dict[str, float]
    start_minutes: dict[str, list[float]]
    bench_minutes: dict[str, list[float]]
    positions: dict[str, dict[str, float]]
    formations: dict[str, dict[str, float]]
    slot_positions: dict[str, list[str]]  # formation -> the position code of slots 1 to 11


def build_priors(world: sqlite3.Connection, leagues: list[str], season: int,
                 rules: TimeRules) -> ModelPriors:  # fmt: skip
    """The parts of `season`, from its five seasons before, one season's rows at a time."""
    first = season - TRAINING_SEASONS
    loaded = read_world(world, first - 1, season - 1, rules)
    designs, labels = [], []
    groups = defaultdict(lambda: [0.0, 0.0])
    starts = defaultdict(lambda: np.zeros(MINUTE_CELLS))
    benches = defaultdict(lambda: np.zeros(MINUTE_CELLS))
    positions = defaultdict(lambda: defaultdict(float))
    formations = defaultdict(lambda: defaultdict(float))
    position_outcomes, formation_outcomes = list_outcomes("L9.position"), list_outcomes(
        "L2.formation.team")
    for year in range(first, season):
        rows, teams = collect_targets(loaded, set(leagues), {year})
        designs.append(np.array([row.features for row in rows]))
        labels.append(np.array([row.started for row in rows], float))
        for row in rows:
            if not row.started:
                groups[str(row.group)][0] += row.minutes > 0
                groups[str(row.group)][1] += 1
                if row.minutes > 0:
                    benches[str(row.group)][min(row.minutes, MINUTE_CELLS - 1)] += 1
            else:
                starts[str(row.group)][min(row.minutes, MINUTE_CELLS - 1)] += 1
                if row.position:
                    code = row.position if row.position in position_outcomes else OTHER
                    positions[str(row.group)][code] += 1
        for team in teams:
            if team.formation:
                code = team.formation if team.formation in formation_outcomes else OTHER
                formations[team.competition][code] += 1
        del rows, teams
    weights = fit_logistic(np.vstack(designs), np.concatenate(labels))
    return ModelPriors(
        weights.tolist(),
        {group: (on + 1.0) / (count + 2.0) for group, (on, count) in groups.items()},
        {
            group: ((cells + 0.01) / (cells + 0.01).sum()).tolist()
            for group, cells in starts.items()
        },
        {
            group: ((cells + 0.01) / (cells + 0.01).sum()).tolist()
            for group, cells in benches.items()
        },
        {
            group: _smooth_frequencies(found, position_outcomes)
            for group, found in positions.items()
        },
        {
            league: _smooth_frequencies(found, formation_outcomes)
            for league, found in formations.items()
        },
        read_slot_positions(world, leagues, first, season - 1),
    )


def _smooth_frequencies(found: dict[str, float], outcomes: list[str]) -> dict[str, float]:
    values = np.array([found.get(item, 0.0) + SMOOTHING for item in outcomes])
    return dict(zip(outcomes, (values / values.sum()).tolist(), strict=True))


def read_slot_positions(world: sqlite3.Connection, leagues: list[str], first: int,
                        last: int) -> dict[str, list[str]]:  # fmt: skip
    """Each formation's commonest position code at each of its eleven slots in the seasons."""
    marks = ", ".join("?" for _ in leagues)
    counted: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for formation, slot, code, count in world.execute(
            f"SELECT t.formation, p.formation_slot, p.position_code, COUNT(*) FROM match_players p"
            f" JOIN match_team_sheets t ON t.match_id = p.match_id AND t.team_id = p.team_id"
            f" JOIN matches m ON m.match_id = p.match_id JOIN seasons s ON s.season_id ="
            f" m.season_id JOIN competitions c ON c.competition_id = s.competition_id"
            f" WHERE c.code IN ({marks}) AND s.season BETWEEN ? AND ? AND p.started = 1"
            f" AND t.formation IS NOT NULL AND p.formation_slot BETWEEN 1 AND 11"
            f" AND p.position_code IS NOT NULL GROUP BY 1, 2, 3",
            (*leagues, first, last)):  # fmt: skip
        counted[(formation, slot)][code] += count
    found: dict[str, list[str]] = {}
    for formation in {formation for formation, _ in counted}:
        slots = [counted.get((formation, slot)) for slot in range(1, 12)]
        if all(slots):
            found[formation] = [max(codes, key=lambda code: (codes[code], code))
                                for codes in slots]  # fmt: skip
    return found


def load_or_build_priors(
    world: sqlite3.Connection,
    leagues: list[str],
    season: int,
    rules: TimeRules,
    folder: Path,
    identity: str,
) -> ModelPriors:
    """The season's parts from `folder`, built and kept there when missing or of another
    release."""
    path = folder / f"lineups-priors-{season}.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("identity") == identity:
            return ModelPriors(**stored["priors"])
    priors = build_priors(world, leagues, season, rules)
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"identity": identity, "priors": priors.__dict__}),
                    encoding="utf-8")  # fmt: skip
    return priors


@dataclass(frozen=True)
class PlayerParts:
    """What the morning read of a player of the pool, kept for the lineup-time publication
    (`vesta.jobs.lineup`): his chance to come on when he does not start, his minutes as a starter
    and as a substitute, his position if he starts."""

    group: str | None
    comes_on: float
    as_starter: tuple[float, ...]
    as_substitute: tuple[float, ...]
    position: dict[str, float]


def mix_minutes(start: float, parts: PlayerParts) -> tuple[np.ndarray, float, float]:
    """A player's minutes (cells 0 to MINUTE_CELLS - 1), his chance to play and his expected
    minutes if he plays, given his chance to start: the morning's and the lineup's alike."""
    minutes = (start * np.array(parts.as_starter)
               + (1 - start) * parts.comes_on * np.array(parts.as_substitute))  # fmt: skip
    minutes[0] += (1 - start) * (1 - parts.comes_on)
    minutes = minutes / minutes.sum()
    on_pitch = minutes[1:]
    expected = float(np.arange(1, MINUTE_CELLS) @ on_pitch / max(on_pitch.sum(), EPSILON))
    return minutes, start + (1 - start) * parts.comes_on, expected


@dataclass(frozen=True)
class TeamLineup:
    team_id: int
    distributions: list[PublishedDistribution]  # version empty: the gate sets it
    formation: str
    probability_formation: float
    slots: list[ExpectedSlot]
    # the pool, for the scorers: (player, position group, chance to start, chance to play,
    # expected minutes if he plays)
    pool: tuple[tuple[int, str | None, float, float, float], ...] = ()
    parts: dict = field(default_factory=dict)  # player -> PlayerParts, for the lineup time


class LineupsOfTheDay:
    """The lineups of the morning's league fixtures."""

    def __init__(self, world: sqlite3.Connection, leagues: list[str], season: int, now: str,
                 rules: TimeRules, folder: Path, identity: str):  # fmt: skip
        self.priors = load_or_build_priors(world, leagues, season, rules, folder, identity)
        self.world = read_world(world, season - 1, season, rules)
        self.now = now
        self.formation_outcomes = list_outcomes("L2.formation.team")
        self.position_outcomes = list_outcomes("L9.position")

    def build_fixture_lineups(
        self,
        match_id: int,
        competition: str,
        season: int,
        kickoff_utc: str,
        home_team_id: int,
        away_team_id: int,
    ) -> tuple[TeamLineup | None, ...]:
        """Both sides' lineups. A player in both pools (named on the bench of the two clubs, never
        on the pitch since he moved, so no appearance tells his team) stays with the club whose
        sheet named him last; the other side is built again without him."""
        teams = (home_team_id, away_team_id)
        sides = [
            self.build_team_lineup(match_id, competition, season, kickoff_utc, team)
            for team in teams
        ]
        if sides[0] is None or sides[1] is None:
            return tuple(sides)
        shared = ({item[0] for item in sides[0].pool} & {item[0] for item in sides[1].pool})
        if not shared:
            return tuple(sides)
        last = self.find_last_named(shared, teams)
        for index, team in enumerate(teams):
            leaving = frozenset(player for player in shared if last[player] != team)
            if leaving:
                sides[index] = self.build_team_lineup(
                    match_id, competition, season, kickoff_utc, team, leaving
                )
        return tuple(sides)

    def find_last_named(self, players: set[int], teams: tuple[int, ...]) -> dict[int, int]:
        """player -> the team (of `teams`) whose sheet, bench included, named him last before
        now."""
        found: dict[int, tuple[float, int, int]] = {}
        for team in teams:
            for item in self.world.team_matches.get(team, []):
                if item.known_at > self.now:
                    continue
                for player in players & self.world.sheets.get((item.match_id, team), {}).keys():
                    found[player] = max(found.get(player, (-1.0, 0, 0)),
                                        (item.seconds, item.match_id, team))  # fmt: skip
        return {player: item[2] for player, item in found.items()}

    def build_team_lineup(
        self,
        match_id: int,
        competition: str,
        season: int,
        kickoff_utc: str,
        team_id: int,
        exclude: frozenset[int] = frozenset(),
    ) -> TeamLineup | None:
        """A team's lineup distributions and expected eleven for its fixture, without the
        players of `exclude`; None without a history in the world."""
        played = [item for item in self.world.team_matches.get(team_id, [])
                  if item.known_at <= self.now]  # fmt: skip
        before = [item for item in played[-120:] if (item.match_id, team_id) in self.world.sheets]
        if not before:
            return None
        fixture = TeamMatch(match_id, kickoff_utc, competition, True, season, self.now, self.now,
                            parse_instant_to_seconds(kickoff_utc))  # fmt: skip
        recent = before[::-1][:HISTORY]
        weights = [FADE ** age for age in range(len(recent))]
        rows = [
            row
            for row in build_player_rows(self.world, team_id, fixture, before, recent, weights, {})
            if row.player_id not in exclude
        ]
        if len(rows) < 11:
            return None
        priors = self.priors
        design = np.column_stack([np.ones(len(rows)), [row.features for row in rows]])
        starts = scale_to_eleven(design @ np.array(priors.weights), np.array([row.goalkeeper
                                                                              for row in rows]))
        distributions: list[PublishedDistribution] = []
        position_of: dict[int, dict[str, float]] = {}
        pool, parts = [], {}
        flat = [1.0 / MINUTE_CELLS] * MINUTE_CELLS
        for row, start in zip(rows, starts, strict=True):
            start = float(start)
            group = str(row.group)
            on = row.came_on_rate
            comes_on = ((on[0] + STRENGTH["came_on"] * priors.came_on.get(group, 0.3))
                        / (on[1] + STRENGTH["came_on"]))  # fmt: skip
            as_starter = shrink_to_prior(build_histogram(row.start_minutes, MINUTE_CELLS),
                                         np.array(priors.start_minutes.get(group, flat)),
                                         STRENGTH["minutes"])  # fmt: skip
            as_substitute = shrink_to_prior(build_histogram(row.bench_minutes, MINUTE_CELLS),
                                            np.array(priors.bench_minutes.get(group, flat)),
                                            STRENGTH["minutes"])  # fmt: skip
            position = _shrink_categorical(
                dict(row.positions),
                priors.positions.get(group),
                STRENGTH["position"],
                self.position_outcomes,
            )
            position_of[row.player_id] = position
            parts[row.player_id] = PlayerParts(row.group, float(comes_on),
                                               tuple(float(x) for x in as_starter),
                                               tuple(float(x) for x in as_substitute),
                                               position)  # fmt: skip
            minutes, plays, expected = mix_minutes(start, parts[row.player_id])
            pool.append((row.player_id, row.group, start, plays, expected))
            distributions += [
                PublishedDistribution(
                    "L1.starts", row.player_id, "", {"yes": _clip_probability(start)}
                ),
                PublishedDistribution(
                    "L6.plays", row.player_id, "", {"yes": _clip_probability(plays)}
                ),
                PublishedDistribution(
                    "L6.minutes",
                    row.player_id,
                    "",
                    {"probabilities": minutes.tolist(), "tail_from": MINUTE_CELLS, "tail": 0.0},
                ),
                PublishedDistribution("L9.position", row.player_id, "", position),
            ]
        own: dict[str, float] = defaultdict(float)
        for item, weight in zip(recent, weights, strict=False):
            formation = self.world.formations.get((item.match_id, team_id))
            if formation:
                own[formation] += weight
        formations = _shrink_categorical(
            dict(own),
            priors.formations.get(competition),
            STRENGTH["formation"],
            self.formation_outcomes,
        )
        distributions.append(PublishedDistribution("L2.formation.team", team_id, "", formations))
        formation = max((item for item in formations if item in priors.slot_positions),
                        key=lambda item: formations[item], default=None)  # fmt: skip
        if formation is None:
            return None
        slots = choose_expected_eleven([row.player_id for row in rows], starts, position_of,
                                       priors.slot_positions[formation])  # fmt: skip
        return TeamLineup(team_id, distributions, formation, formations[formation], slots,
                          tuple(pool), parts)


def choose_expected_eleven(players: list[int], starts, position_of: dict[int, dict[str, float]],
                           codes: list[str]) -> list[ExpectedSlot]:  # fmt: skip
    """One player per slot, the assignment that makes the eleven likeliest: the product of each
    chosen player's chance to start and his chance of the slot's position."""
    cost = np.array([[-math.log(max(float(start), EPSILON))
                      - math.log(max(position_of[player].get(code, 0.0), EPSILON))
                      for code in codes] for player, start in zip(players, starts, strict=True)])
    chosen, slot_index = linear_sum_assignment(cost)
    return sorted((ExpectedSlot(int(slot) + 1, players[player], codes[slot],
                                float(starts[player]),
                                float(position_of[players[player]].get(codes[slot], 0.0)))
                   for player, slot in zip(chosen, slot_index, strict=True)),
                  key=lambda slot: slot.formation_slot)  # fmt: skip


def _shrink_categorical(own: dict[str, float], prior: dict[str, float] | None, strength: float,
                        outcomes: list[str]) -> dict[str, float]:  # fmt: skip
    values = np.array([sum(weight for item, weight in own.items()
                           if (item if item in outcomes else OTHER) == outcome)
                       + strength * (prior or {}).get(outcome, 1.0 / len(outcomes))
                       for outcome in outcomes])  # fmt: skip
    return dict(zip(outcomes, (values / values.sum()).tolist(), strict=True))


def _clip_probability(probability: float) -> float:
    return min(max(probability, 0.0), 1.0)
