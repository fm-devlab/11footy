"""A UEFA cup or a national tournament simulated to its final (T4, T6): the round each team
reaches.

The phase before the knockout (a league phase, or groups) is drawn and ranked as a league season
(`vesta.season.league`, the competition's tie-break rules of that season); the places that go on
build the knockout by the format's bracket rule (`config/vesta/tournament-formats.toml`,
`vesta.season.brackets`), and the knockout is played out tie by tie (`vesta.season.knockout`).
Once the knockout has begun, its tree as it stands is the start instead. Every simulation uses
one seeded generator.

The outcomes of `T4.round_reached` and `T6.round_reached` are, in order, the phase, each round of
the knockout, and the winner. A line is decided by exact reasoning: a team out of the tournament
has its outcome; a team still in has reached every round it has played or been drawn into.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vesta.forecast_log.writing import PublishedDistribution
from vesta.forecast_reading import read_line
from vesta.quantities import QuantityDefinition
from vesta.season.brackets import build
from vesta.season.knockout import Strength, Tie, TieSampler, play
from vesta.season.league import LeagueSeason, simulate_league
from vesta.season.standings import TieBreakRule
from vesta.spine.ties import TieParameters, TieRule

FORMATS_FILE = Path(__file__).resolve().parents[3] / "config" / "vesta" / "tournament-formats.toml"


@dataclass(frozen=True)
class DropIn:
    """The clubs of `place` in each group of the cup `source` that go on into this cup's
    knockout; the best `seeded` of them, ranked across the groups by `ranking`, are seeded."""

    source: str
    place: int
    seeded: int = 0
    ranking: tuple[str, ...] = ()


@dataclass(frozen=True)
class TournamentFormat:
    phase: str
    advance: dict[str, tuple[int, int]]
    bracket: str
    rounds: tuple[tuple[str, int], ...]
    best_thirds: int = 0  # the thirds that go on, compared across the groups
    thirds_ranking: tuple[str, ...] = ()
    hosts: tuple[str, ...] = ()  # the host teams' names, at home on the tournament's grounds
    neutral_rounds: tuple[str, ...] = ()  # single matches on neutral ground before the final
    drop_in: "DropIn | None" = None  # the clubs a higher cup's phase sends into the knockout
    fixed_from: str | None = None  # the round from which one draw sets the bracket to the final
    fixed: dict | None = None  # a `fixed` bracket: first_round, later, thirds_table

    @property
    def outcomes(self) -> tuple[str, ...]:
        return (self.phase, *(name for name, _ in self.rounds), "winner")


def read_tournament_format(
    competition: str, season: int, path: Path = FORMATS_FILE
) -> TournamentFormat:
    """The format of `competition` in `season`; refused when the file has none."""
    for row in tomllib.loads(path.read_text(encoding="utf-8"))["format"]:
        seasons = range(row["first_season"], row["last_season"] + 1)
        if competition in row["competitions"] and season in seasons:
            return TournamentFormat(
                phase=row["phase"],
                advance={name: tuple(places) for name, places in row["advance"].items()},
                bracket=row["bracket"],
                rounds=tuple((name, int(legs)) for name, legs in row["rounds"]),
                best_thirds=int(row.get("best_thirds", 0)),
                thirds_ranking=tuple(row.get("thirds_ranking", "").split("|"))
                if row.get("thirds_ranking") else (),
                hosts=tuple(row.get("hosts", ())),
                neutral_rounds=tuple(row.get("neutral_rounds", ())),
                drop_in=DropIn(row["drop_in"]["from"], int(row["drop_in"]["place"]),
                               int(row["drop_in"].get("seeded", 0)),
                               tuple(row["drop_in"].get("ranking", "").split("|"))
                               if row["drop_in"].get("ranking") else ())
                if "drop_in" in row else None,
                fixed_from=row.get("fixed_from"),
                fixed={name: row[name] for name in ("first_round", "later", "thirds_table")
                       if name in row} if row["bracket"] == "fixed" else None,
            )
    raise ValueError(f"no format for {competition} in {season}: write it in {path.name}")


@dataclass(frozen=True)
class TournamentDraws:
    """(n,) per team the index of its outcome in `TournamentFormat.outcomes`."""

    outcomes: tuple[str, ...]
    reached: dict[int, np.ndarray]


def simulate_tournament(
    form: TournamentFormat,
    tables: dict[str, LeagueSeason],
    table_rule: TieBreakRule,
    strength: Strength,
    tie_rule: TieRule,
    parameters: TieParameters,
    draws: int,
    seed: int,
) -> TournamentDraws:
    """From the phase as it stands: `tables` is one table ("") for a league phase, one per group
    (its letter) for groups."""
    generator = np.random.default_rng(seed)
    drawn = {group: simulate_league(table, table_rule, draws, int(generator.integers(2**31)))
             for group, table in sorted(tables.items())}  # fmt: skip
    places = {group: result.places for group, result in drawn.items()}
    thirds = _choose_best_thirds(form, drawn, generator) if form.best_thirds else None
    sampler = TieSampler(strength, tie_rule, parameters, generator)
    round_index = {name: index + 1 for index, (name, _) in enumerate(form.rounds)}
    reached = {team: np.zeros(draws, dtype=np.int64)
               for table in tables.values() for team in table.team_ids}  # fmt: skip
    for simulation in range(draws):
        order = {group: [tables[group].team_ids[column]
                         for column in np.argsort(places[group][simulation])]
                 for group in tables}  # fmt: skip
        chosen = tuple(thirds[simulation]) if thirds is not None else ()
        tree = build(form.bracket, order, form.rounds, generator, chosen, fixed=form.fixed)
        furthest: dict[int, int] = {}
        play(tree, sampler, furthest, round_index)
        for team, index in furthest.items():
            reached[team][simulation] = index
    return TournamentDraws(form.outcomes, reached)


def _choose_best_thirds(form, drawn, generator) -> list[list[str]]:
    """Per simulation, the groups whose third goes on, by the format's ranking of thirds."""
    groups = sorted(drawn)
    criteria = {"points": "points", "goal_difference": "goal_difference",
                "goals_for": "goals_for"}  # fmt: skip
    keys = []
    for criterion in form.thirds_ranking:
        if criterion == "lots":
            keys.append(generator.random((len(next(iter(drawn.values())).places), len(groups))))
            continue
        if criterion == "wins":
            columns = []
            for group in groups:
                result = drawn[group]
                third = np.argmax(result.places == 2, axis=1)
                columns.append(_count_wins(result)[np.arange(len(third)), third])
            keys.append(-np.stack(columns, axis=1).astype(float))
            continue
        if criterion not in criteria:
            raise ValueError(f"unknown criterion of the thirds: {criterion!r}")
        columns = []
        for group in groups:
            result = drawn[group]
            third = np.argmax(result.places == 2, axis=1)
            rows = np.arange(len(third))
            columns.append(getattr(result, criteria[criterion])[rows, third])
        keys.append(-np.stack(columns, axis=1).astype(float))
    count = keys[0].shape[0]
    rows = np.repeat(np.arange(count), len(groups))
    order = np.lexsort([key.ravel() for key in reversed(keys)] + [rows]).reshape(count, -1)
    best = order % len(groups)
    return [[groups[index] for index in row[: form.best_thirds]] for row in best]


def _count_wins(result) -> np.ndarray:
    """(n, teams) every team's wins in each simulated group, from its matches."""
    matches = result.matches
    wins = np.zeros(result.points.shape, dtype=np.int64)
    for index, (home, away) in enumerate(zip(matches.home, matches.away, strict=True)):
        wins[:, home] += matches.home_goals[:, index] > matches.away_goals[:, index]
        wins[:, away] += matches.away_goals[:, index] > matches.home_goals[:, index]
    return wins


def simulate_knockout(
    form: TournamentFormat,
    tree: Tie,
    out: dict[int, str],
    strength: Strength,
    tie_rule: TieRule,
    parameters: TieParameters,
    draws: int,
    seed: int,
) -> TournamentDraws:
    """From a knockout under way: its tree as it stands, and the teams already out with the
    outcome they reached."""
    generator = np.random.default_rng(seed)
    sampler = TieSampler(strength, tie_rule, parameters, generator)
    round_index = {name: index + 1 for index, (name, _) in enumerate(form.rounds)}
    reached = {team: np.full(draws, form.outcomes.index(outcome), dtype=np.int64)
               for team, outcome in out.items()}  # fmt: skip
    for simulation in range(draws):
        furthest: dict[int, int] = {}
        play(tree, sampler, furthest, round_index)
        for team, index in furthest.items():
            reached.setdefault(team, np.zeros(draws, dtype=np.int64))[simulation] = index
    return TournamentDraws(form.outcomes, reached)


def round_reached_distributions(
    key: str,
    draws: TournamentDraws,
    known: dict[int, tuple[int, int]],
    model_version: str,
    definitions: dict[str, QuantityDefinition],
) -> list[PublishedDistribution]:
    """`key` (T4 or T6 `.round_reached`) of every team; `known` gives, per team, the lowest and
    highest outcome still possible (indexes into the outcomes), from `still_possible`."""
    definition = definitions[key]
    count = len(draws.outcomes)
    published = []
    for team, reached in sorted(draws.reached.items()):
        shares = np.bincount(reached, minlength=count) / len(reached)
        values = [[name, float(share)] for name, share in zip(draws.outcomes, shares, strict=True)]
        low, high = known.get(team, (0, count - 1))
        decided = tuple(
            line for line in definition.quantity.lines
            if len({round(read_line("ordered", _make_certain_outcome(draws.outcomes, index), line))
                    for index in range(low, high + 1)}) == 1
        )  # fmt: skip
        published.append(PublishedDistribution(key, team, model_version, values,
                                               decided_lines=decided))  # fmt: skip
    return published


def find_possible_outcomes(form: TournamentFormat, tree: Tie | None, out: dict[int, str],
                           teams: list[int]) -> dict[int, tuple[int, int]]:  # fmt: skip
    """Per team, the lowest and highest outcome still possible: fixed for a team out (listed in
    `out`, or the loser of a decided tie of the tree); for a team in the knockout, from the
    deepest round it is known to play, to the winner."""
    outcomes = form.outcomes
    possible = {team: (0, len(outcomes) - 1) for team in teams}
    if tree is not None:
        possible = {}
        known, lost = _find_known_rounds(tree, {name: index + 1 for index, (name, _)
                                                in enumerate(form.rounds)})  # fmt: skip
        for team, deepest in known.items():
            possible[team] = (deepest, deepest) if team in lost else (deepest, len(outcomes) - 1)
    for team, outcome in out.items():
        index = outcomes.index(outcome)
        possible[team] = (index, index)
    return possible


def _find_known_rounds(tree: Tie | int, round_index: dict[str, int]) -> tuple[dict[int, int], set]:
    """The deepest round each team is known to play (its ties decided up to there), and the
    teams that lost a decided tie."""
    known: dict[int, int] = {}
    lost: set[int] = set()

    def collect_known_rounds(node: Tie | int) -> int | None:
        if isinstance(node, int):
            return node
        first, second = collect_known_rounds(node.first), collect_known_rounds(node.second)
        for team in (first, second):
            if team is not None:
                known[team] = max(known.get(team, 0), round_index[node.round_name])
        if node.winner is not None:
            lost.update(team for team in (first, second) if team not in (None, node.winner))
            if node.round_name == "final":
                known[node.winner] = round_index[node.round_name] + 1
                lost.add(node.winner)  # the winner's outcome is settled too
        return node.winner

    collect_known_rounds(tree)
    return known, lost


def _make_certain_outcome(outcomes: tuple[str, ...], index: int) -> list:
    return [[name, 1.0 if position == index else 0.0] for position, name in enumerate(outcomes)]
