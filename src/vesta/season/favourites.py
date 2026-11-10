"""The favourites of a national tournament (T6.round_reached): the season job.

    python -m vesta.season.favourites <football-world.db> <forecast-log.db> <competition> <season>
                                       <as-of date> <model version>

Reads the tournament from the world (its groups from the stage of each match, `group a` to
`group f`; the matches played with their official result, the matches to play), gives every match
to play and every tie not yet drawn the nations model's strength as it stands at the date
(`vesta.season.strength`), simulates the tournament by its format (`config/vesta/
tournament-formats.toml`) and publishes `T6.round_reached` of every team at the stage
`season_day`. A reader takes the favourite from the line `outcome=winner` (it sorts; it computes
nothing).

Once the knockout is drawn (its matches in the world), the job starts from it instead: the final
group places are ranked from the results by the competition's rules, the bracket rule builds the
tree, every tie of the first round is checked against the world's fixtures (a pairing the rules do
not give, after a criterion the world does not hold such as the card score, is refused, never
guessed), and the ties played carry their winner (the world's `winner_team_id` where the score is
level after extra time or penalties), a two-legged tie half played its first leg. The teams out
in the groups have reached the group stage; every line a decided tie settles is decided.

It refuses what it cannot do honestly: a format or a rule not written, a tie-break rule not
verified (nothing that depends on one is published), a knockout drawn before the groups are over,
a pairing the rules do not give. Nothing is published before the model's verdict and the
maintainers' approval.
"""

import sqlite3
import sys
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from vesta.forecast_log.store import open_log
from vesta.forecast_log.writing import Publication, PublishedDistribution, publish
from vesta.quantities import QuantityDefinition, load_quantities
from vesta.research.repository import run_git
from vesta.season.brackets import build
from vesta.season.knockout import Strength, Tie, list_leaves
from vesta.season.league import (
    LeagueSeason,
    _count_points_now_and_most,
    place_bounds,
    simulate_league,
)
from vesta.season.standings import read_tie_break_rule
from vesta.season.strength import build_nations_strength
from vesta.season.tournament import (
    _choose_best_thirds,
    find_possible_outcomes,
    read_tournament_format,
    round_reached_distributions,
    simulate_knockout,
    simulate_tournament,
)
from vesta.spine.nations import NationsParameters
from vesta.spine.score_matrix import SIZE
from vesta.spine.ties import TieParameters, TieRule, read_tie_rule
from vesta.spine.world import RESULT_STATUSES, open_world, read_nation_matches

ROOT = Path(__file__).resolve().parents[3]
KEY = "T6.round_reached"
TO_PLAY = ("scheduled", "postponed")


@dataclass(frozen=True)
class KnockoutMatch:
    """A match of the knockout in the world, in the order of kick-off."""

    stage: str
    home: int
    away: int
    status: str
    home_goals: int | None
    away_goals: int | None
    winner: int | None


@dataclass(frozen=True)
class TournamentState:
    season_id: int
    groups: dict[str, LeagueSeason]
    knockout: tuple[KnockoutMatch, ...] = ()

    @property
    def knockout_started(self) -> bool:
        return bool(self.knockout)


def read_tournament(world: sqlite3.Connection, competition: str, season: int,
                    strength: Strength) -> TournamentState:  # fmt: skip
    """The tournament's groups as they stand, the matches to play given the strength's matrices."""
    rows = world.execute(
        """
        SELECT s.season_id, LOWER(COALESCE(m.stage, '')), m.status, m.home_team_id,
               m.away_team_id, m.home_goals, m.away_goals, m.neutral_venue, m.winner_team_id
        FROM matches m
        JOIN seasons s ON s.season_id = m.season_id
        JOIN competitions c ON c.competition_id = s.competition_id
        WHERE c.code = ? AND s.season = ?
        ORDER BY m.kickoff_utc, m.match_id
        """,
        (competition, season),
    ).fetchall()
    if not rows:
        raise ValueError(f"no match of {competition} {season} in the world")
    teams: dict[str, set[int]] = {}
    played: dict[str, list[tuple[int, int, int, int]]] = {}
    fixtures: dict[str, list[tuple[int, int, bool]]] = {}
    knockout = []
    for _, stage, status, home, away, home_goals, away_goals, neutral, winner in rows:
        if not stage.startswith("group "):
            if status in RESULT_STATUSES + TO_PLAY:
                knockout.append(KnockoutMatch(stage, home, away, status, home_goals, away_goals,
                                              winner))  # fmt: skip
            continue
        group = stage.split()[1].upper()
        teams.setdefault(group, set()).update((home, away))
        if status in RESULT_STATUSES and home_goals is not None:
            played.setdefault(group, []).append((home, away, home_goals, away_goals))
        elif status in TO_PLAY:
            fixtures.setdefault(group, []).append((home, away, bool(neutral)))
    groups = {}
    for group, members in sorted(teams.items()):
        to_play = fixtures.get(group, [])
        matrices = (np.array([strength(home, away, neutral)[0] for home, away, neutral in to_play])
                    if to_play else np.zeros((0, SIZE, SIZE)))  # fmt: skip
        groups[group] = LeagueSeason(tuple(sorted(members)), tuple(played.get(group, [])), {},
                                     tuple((home, away) for home, away, _ in to_play),
                                     matrices)  # fmt: skip
    return TournamentState(rows[0][0], groups, tuple(knockout))


def forecast_round_reached(
    state: TournamentState,
    competition: str,
    season: int,
    strength: Strength,
    parameters: TieParameters,
    draws: int,
    seed: int,
    model_version: str,
    definitions: dict[str, QuantityDefinition],
) -> list[PublishedDistribution]:
    """`T6.round_reached` of every team of the tournament."""
    form = read_tournament_format(competition, season)
    table_rule = read_tie_break_rule(competition, season)
    two_legs = any(legs == 2 for _, legs in form.rounds)
    rule = read_tie_rule(competition, season) if two_legs else TieRule(False, False)
    teams = [team for group in state.groups.values() for team in group.team_ids]
    if not state.knockout_started:
        simulated = simulate_tournament(form, state.groups, table_rule, strength, rule,
                                        parameters, draws, seed)  # fmt: skip
        known = group_stage_bounds(state, form)
    else:
        tree = build_knockout_tree(state, form, table_rule, seed)
        out = {team: form.phase for team in teams if team not in list_leaves(tree)}
        simulated = simulate_knockout(form, tree, out, strength, rule, parameters, draws, seed)
        known = find_possible_outcomes(form, tree, out, teams)
    return round_reached_distributions(KEY, simulated, known, model_version, definitions)


def group_stage_bounds(state: TournamentState, form) -> dict[int, tuple[int, int]]:
    """Per team, the lowest and highest outcome still possible during the group stage, by exact
    reasoning on the points: out (the group stage, decided) when no place it can still reach goes
    on, or when its best is a third place and at least as many other groups' thirds as go on are
    surely above it (a group's third ends with at least its third-highest points now); the round
    of 16 at least when it is surely in a place that goes on directly."""
    last = len(form.outcomes) - 1
    directly = max((places[1] for places in form.advance.values()), default=0) - (
        1 if form.best_thirds else 0)  # fmt: skip
    third_floor = {}
    for name, group in state.groups.items():
        now, _ = _count_points_now_and_most(group)
        third_floor[name] = sorted(now, reverse=True)[2] if len(now) >= 3 else -1
    bounds = {}
    for name, group in state.groups.items():
        best, worst = place_bounds(group)
        _, most = _count_points_now_and_most(group)
        for column, team in enumerate(group.team_ids):
            out = best[column] > directly + (1 if form.best_thirds else 0)
            if form.best_thirds and best[column] == directly + 1:
                above = sum(third_floor[other] > most[column]
                            for other in state.groups if other != name)  # fmt: skip
                out = above >= form.best_thirds
            if out:
                bounds[team] = (0, 0)
            elif worst[column] <= directly:
                bounds[team] = (1, last)
            else:
                bounds[team] = (0, last)
    return bounds


def build_knockout_tree(state: TournamentState, form, table_rule, seed: int) -> Tie:
    """The knockout as it stands: the bracket of the final group places, checked against the
    world's first-round fixtures, its ties played carrying their winner."""
    unfinished = [name for name, group in state.groups.items() if group.fixtures]
    if unfinished:
        raise ValueError(f"the knockout is drawn but groups {unfinished} are not over")
    generator = np.random.default_rng(seed)
    drawn = {name: simulate_league(group, table_rule, 1, seed)
             for name, group in sorted(state.groups.items())}  # fmt: skip
    order = {name: [state.groups[name].team_ids[column]
                    for column in np.argsort(drawn[name].places[0])]
             for name in drawn}  # fmt: skip
    thirds = tuple(_choose_best_thirds(form, drawn, generator)[0]) if form.best_thirds else ()
    tree = build(form.bracket, order, form.rounds, generator, thirds, fixed=form.fixed)
    first_round = form.rounds[0][0]
    return _apply_world_results(tree, state.knockout, first_round)


def _apply_world_results(node, matches, first_round):
    """The tree with the winner of every tie played and the first leg of every tie half played;
    a first-round tie the world does not hold is refused."""
    if isinstance(node, int):
        return node
    first = _apply_world_results(node.first, matches, first_round)
    second = _apply_world_results(node.second, matches, first_round)
    sides = (_resolve_side(first), _resolve_side(second))
    if None in sides:
        return Tie(node.round_name, node.legs, first, second, node.neutral)
    between = [match for match in matches if {match.home, match.away} == set(sides)]
    if not between:
        if node.round_name == first_round:
            raise ValueError(f"the rules give {sides[0]} v {sides[1]} in the {first_round}, the"
                             " world does not: a criterion it does not hold may have decided a"
                             " place")  # fmt: skip
        return Tie(node.round_name, node.legs, first, second, node.neutral)
    played = [match for match in between if match.status in RESULT_STATUSES]
    if node.legs == 2 and len(played) == 1:  # half played: the first leg, `first` at home
        leg = played[0]
        goals = ((leg.home_goals, leg.away_goals) if leg.home == sides[0]
                 else (leg.away_goals, leg.home_goals))  # fmt: skip
        if leg.home != sides[0]:  # the world's first leg was at the other side's ground
            first, second = second, first
            goals = goals[::-1]
        return Tie(node.round_name, 2, first, second, node.neutral, first_leg=goals)
    if len(played) == node.legs:
        winner = _find_decided_winner(played, sides)
        if winner is None:
            raise ValueError(f"the tie {sides[0]} v {sides[1]} is played but the world has no"
                             " winner")  # fmt: skip
        return Tie(node.round_name, node.legs, first, second, node.neutral, winner=winner)
    return Tie(node.round_name, node.legs, first, second, node.neutral)


def _resolve_side(node) -> int | None:
    return node if isinstance(node, int) else node.winner


def _find_decided_winner(played, sides) -> int | None:
    last = played[-1]
    if last.winner is not None:
        return last.winner
    goals = dict.fromkeys(sides, 0)
    for match in played:
        goals[match.home] += match.home_goals
        goals[match.away] += match.away_goals
    first, second = sides
    if goals[first] == goals[second]:
        return None
    return first if goals[first] > goals[second] else second


def read_host_id(world: sqlite3.Connection, name: str) -> int:
    """The world's id of a host named in the format; refused when it is not exactly one team."""
    rows = world.execute("SELECT team_id FROM teams WHERE name = ? AND kind = 'nation'",
                         (name,)).fetchall()  # fmt: skip
    if len(rows) != 1:
        raise ValueError(f"the host {name!r} is {len(rows)} national teams in the world")
    return rows[0][0]


def main(arguments: list[str]) -> int:
    if len(arguments) != 6:
        print(__doc__)
        return 2
    world_path, log_path, competition, season, as_of, model_version = arguments
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    research = tomllib.loads((ROOT / "config/vesta/research.toml").read_text(encoding="utf-8"))
    if not read_tie_break_rule(competition, int(season)).verified:
        print(f"{competition} {season}: its tie-break rule is not verified; nothing is published")
        return 1
    hosts = read_tournament_format(competition, int(season)).hosts
    with open_world(Path(world_path)) as world:
        host_ids = frozenset(read_host_id(world, name) for name in hosts)
        strength = build_nations_strength(read_nation_matches(world),
                                          NationsParameters.from_config(spine), as_of,
                                          host_ids)  # fmt: skip
        state = read_tournament(world, competition, int(season), strength)
    distributions = forecast_round_reached(
        state, competition, int(season), strength, TieParameters.from_config(spine),
        int(spine["season"]["simulations"]), int(research["verdict"]["seed"]), model_version,
        load_quantities(),
    )
    log = open_log(Path(log_path))
    publication = Publication(
        stage="season_day", trigger="morning",
        published_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        code_commit=run_git(ROOT, "rev-parse", "HEAD").strip(), season_id=state.season_id,
    )  # fmt: skip
    publication_id = publish(log, publication, distributions)
    log.close()
    print(f"publication {publication_id}: {len(distributions)} teams of {competition} {season}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
