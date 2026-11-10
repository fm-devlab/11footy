"""The UEFA cups' progression (T4): a season of the three cups simulated together, at an instant.

A cup's season is read from the world (`read_cup_season`): its phase (a league phase, or groups),
the matches of its knockout grouped into ties. At an instant, only what was known then is used:

- **the phase:** its matches played before the instant with their official result, the others to
  play (their pairings were drawn before the phase began);
- **the knockout:** a round's pairings are known once one of its matches has kicked off (one
  draw sets a round), the results of the matches kicked off before the instant; where the format
  sets the bracket from a round to the final by one draw (`fixed_from`, from 2024-25), the world's
  later pairings are known once that round has begun. Every other pairing is drawn by the format's
  bracket rule (`vesta.season.brackets`). No draw is read before its round has begun.

The three cups of a season are simulated together: until 2023-24 the thirds of a cup's groups
drop into the next cup's knockout (`TournamentFormat.drop_in`), so the cup they come from has its
phase drawn in the same simulation. A club still in at the instant gets every outcome of the
format (`TournamentFormat.outcomes`: the phase, each round, the winner) from the draws; the
phase's place of every club of the phase too.

What happened (`truth`): the deepest round of the knockout a club played, or the round its place
in the final phase table sends it to when a tie was awarded without a match (Spartak Moscow v RB
Leipzig, 2021-22); the winner of the final; its place in the final table ranked by the season's
rules.
"""

import sqlite3
from dataclasses import dataclass, field

import numpy as np

from vesta.season.brackets import (
    SEEDED_PAIRS_2024,
    build,
    build_league_phase_from_play_offs,
    build_winners_v_play_offs,
    mark_neutral_rounds,
    open_draws,
)
from vesta.season.knockout import Strength, Tie, TieSampler, play
from vesta.season.league import LeagueSeason, SeasonDraws, place_bounds, simulate_league
from vesta.season.standings import TieBreakRule, read_tie_break_rule
from vesta.season.tournament import TournamentFormat, _choose_best_thirds, read_tournament_format
from vesta.spine.score_matrix import SIZE
from vesta.spine.ties import TieParameters, TieRule, read_tie_rule
from vesta.spine.world import RESULT_STATUSES

# the world's stage of a knockout round -> the format's round
ROUND_OF_STAGE = {
    "knockout round playoffs": "knockout_play_offs",
    "round of 32": "round_of_32",
    "round of 16": "round_of_16",
    "quarterfinals": "quarter_finals",
    "quarter-finals": "quarter_finals",
    "semifinals": "semi_finals",
    "semi-finals": "semi_finals",
    "final": "final",
}
PHASE_STAGES = ("league phase",)  # and every stage that opens with "group"
# brackets set by the places alone, no draw: the knockout follows the final tables (T6)
SET_BY_PLACES = ("fixed", "world_cup_32", "afc_asian_cup_2027")


@dataclass(frozen=True)
class CupMatch:
    match_id: int
    round_name: str  # "phase", or a round of the format
    group: str  # the group's letter; "" for a league phase or the knockout
    kickoff: str
    home: int
    away: int
    neutral: bool
    played: bool
    home_goals: int | None
    away_goals: int | None
    winner: int | None  # the match's winner, where the world names one (extra time, penalties)
    tie_id: int | None


@dataclass(frozen=True)
class CupSeason:
    code: str
    season: int
    form: TournamentFormat
    table_rule: TieBreakRule
    tie_rule: TieRule
    phase: tuple[CupMatch, ...]
    knockout: tuple[CupMatch, ...]
    tie_winners: dict[int, int] = field(default_factory=dict)

    @property
    def groups(self) -> dict[str, tuple[int, ...]]:
        members: dict[str, set[int]] = {}
        for match in self.phase:
            members.setdefault(match.group, set()).update((match.home, match.away))
        return {group: tuple(sorted(teams)) for group, teams in sorted(members.items())}

    @property
    def round_index(self) -> dict[str, int]:
        return {name: index + 1 for index, (name, _) in enumerate(self.form.rounds)}


def read_cup_season(world: sqlite3.Connection, code: str, season: int) -> CupSeason:
    """The phase and the knockout of `code` in `season` (qualifying rounds left out)."""
    rows = world.execute(
        """
        SELECT m.match_id, LOWER(COALESCE(m.stage, '')), COALESCE(m.group_name, ''), m.kickoff_utc,
               m.home_team_id, m.away_team_id, m.neutral_venue, m.status, m.home_goals,
               m.away_goals, m.winner_team_id, m.tie_id
        FROM matches m
        JOIN seasons s ON s.season_id = m.season_id
        JOIN competitions c ON c.competition_id = s.competition_id
        WHERE c.code = ? AND s.season = ?
        ORDER BY m.kickoff_utc, m.match_id
        """,
        (code, season),
    ).fetchall()
    if not rows:
        raise ValueError(f"no match of {code} {season} in the world")
    phase, knockout = [], []
    for (match_id, stage, group, kickoff, home, away, neutral, status, home_goals, away_goals,
         winner, tie_id) in rows:  # fmt: skip
        played = status in RESULT_STATUSES and home_goals is not None
        if stage.startswith("group") or stage in PHASE_STAGES:
            letter = group.split()[-1].upper() if group else ""
            if letter.isdigit():  # the world names some groups 1-6 for A-F (World Cup 2014-2022)
                letter = chr(64 + int(letter))
            phase.append(CupMatch(match_id, "phase", letter, kickoff, home, away, bool(neutral),
                                  played, home_goals, away_goals, winner, None))  # fmt: skip
        elif stage in ROUND_OF_STAGE:
            knockout.append(CupMatch(match_id, ROUND_OF_STAGE[stage], "", kickoff, home, away,
                                     bool(neutral), played, home_goals, away_goals, winner,
                                     tie_id))  # fmt: skip
    tie_winners = dict(world.execute(
        """
        SELECT t.tie_id, t.winner_team_id FROM match_ties t
        JOIN seasons s ON s.season_id = t.season_id
        JOIN competitions c ON c.competition_id = s.competition_id
        WHERE c.code = ? AND s.season = ? AND t.winner_team_id IS NOT NULL
        """,
        (code, season),
    ).fetchall())  # fmt: skip
    form = read_tournament_format(code, season)
    two_legs = any(legs == 2 for _, legs in form.rounds)
    return CupSeason(code, season, form, read_tie_break_rule(code, season),
                     read_tie_rule(code, season) if two_legs else TieRule(False, False),
                     tuple(phase), tuple(knockout), tie_winners)  # fmt: skip


# ---- the state at an instant ---------------------------------------------------------------


def build_phase_tables(cup: CupSeason, instant: str, strength: Strength) -> dict[str, LeagueSeason]:
    """Each group (or the league phase, "") as it stands at `instant`, the matches to play given
    `strength`'s matrices."""
    tables = {}
    for group, teams in cup.groups.items():
        matches = [match for match in cup.phase if match.group == group]
        played = tuple((match.home, match.away, match.home_goals, match.away_goals)
                       for match in matches if is_result_known(match, instant))  # fmt: skip
        to_play = [match for match in matches if not is_result_known(match, instant)]
        matrices = (np.array([strength(match.home, match.away, match.neutral)[0]
                              for match in to_play])
                    if to_play else np.zeros((0, SIZE, SIZE)))  # fmt: skip
        tables[group] = LeagueSeason(teams, played, {},
                                     tuple((match.home, match.away) for match in to_play),
                                     matrices)  # fmt: skip
    return tables


def is_result_known(match: CupMatch, instant: str) -> bool:
    """A result known at `instant`: kicked off before it, and played."""
    return match.kickoff < instant and match.played


def is_phase_over(tables: dict[str, LeagueSeason]) -> bool:
    return all(not table.fixtures for table in tables.values())


@dataclass(frozen=True)
class KnownKnockout:
    """The knockout's draws known at an instant: the last known round's ties (or the whole tree
    when a fixed bracket reaches the final), linked to the ties before them."""

    last_round: str | None
    nodes: tuple[Tie | int, ...]
    awarded: tuple[tuple[int, int], ...] = ()  # (winner, loser) of ties awarded without a match


def find_known_knockout(cup: CupSeason, instant: str,
                        final_order: dict[str, list[int]] | None) -> KnownKnockout:
    """The pairings known at `instant` with the results known then; `final_order`, the phase's
    final places (when the phase is over), names the clubs of a tie awarded without a match."""
    names = [name for name, _ in cup.form.rounds]
    legs = dict(cup.form.rounds)
    begun = [name for name in names
             if any(match.round_name == name and match.kickoff < instant
                    for match in cup.knockout)]  # fmt: skip
    if not begun:
        return KnownKnockout(None, ())
    last = begun[-1]
    if names[: len(begun)] != begun:
        raise ValueError(f"{cup.code} {cup.season}: round {last} begun before an earlier one")
    if cup.form.fixed_from in begun:
        last = names[-1]  # the bracket from `fixed_from` to the final was set by one draw
    node_of: dict[int, Tie | int] = {}
    awarded = []
    nodes: list[Tie | int] = []
    for name in names[: names.index(last) + 1]:
        ties = _collect_world_ties(cup, name)
        nodes = []
        for matches in ties:
            first, second = matches[0].home, matches[0].away
            node = _make_world_tie(cup, name, legs[name], matches, node_of.get(first, first),
                                   node_of.get(second, second), instant)  # fmt: skip
            node_of[first] = node_of[second] = node
            nodes.append(node)
        expected = count_ties_in_round(cup, name)
        if len(nodes) < expected and name in begun:
            pairs = _find_awarded_ties(
                cup, name, node_of, nodes, final_order, expected - len(nodes)
            )
            for winner, loser in pairs:
                node = Tie(name, 1, node_of.get(winner, winner), node_of.get(loser, loser),
                           winner=winner)  # fmt: skip
                node_of[winner] = node_of[loser] = node
                nodes.append(node)
                awarded.append((winner, loser))
        if len(nodes) != expected:
            raise ValueError(f"{cup.code} {cup.season} {name}: {len(nodes)} ties, the format"
                             f" has {expected}")  # fmt: skip
    return KnownKnockout(last, tuple(nodes), tuple(awarded))


def count_ties_in_round(cup: CupSeason, name: str) -> int:
    """The ties of round `name` by the format: one final, and before each round as many ties as
    it needs winners beside the clubs the phase sends straight into it."""
    names = [round_name for round_name, _ in cup.form.rounds]
    groups = len(cup.groups)
    ties = 1
    for later in reversed(names[names.index(name) + 1:]):
        low, high = cup.form.advance.get(later, (1, 0))
        ties = 2 * ties - (high - low + 1) * groups
    return ties


def _collect_world_ties(cup: CupSeason, name: str) -> list[list[CupMatch]]:
    """The ties of round `name` in the world, each its matches in kick-off order; a match with no
    tie is a single match."""
    ties: dict[object, list[CupMatch]] = {}
    for match in cup.knockout:
        if match.round_name == name:
            key = match.tie_id if match.tie_id is not None else ("single", match.match_id)
            ties.setdefault(key, []).append(match)
    return [sorted(matches, key=lambda match: match.kickoff) for _, matches in sorted(
        ties.items(), key=lambda item: item[1][0].kickoff)]  # fmt: skip


def _make_world_tie(cup, name, legs, matches, first, second, instant) -> Tie:
    """A world tie as a node: `first` at home in its first match; the first leg's score once
    played; the winner once every match is played."""
    single = len(matches) == 1 and matches[0].tie_id is None
    legs = 1 if single else legs
    neutral = matches[0].neutral if legs == 1 else False
    sides = (matches[0].home, matches[0].away)
    played = [match for match in matches if is_result_known(match, instant)]
    if len(played) == legs and len(matches) == legs:
        return Tie(name, legs, first, second, neutral, winner=_find_tie_winner(cup, matches, sides))
    if legs == 2 and len(played) == 1:
        leg = played[0]
        return Tie(name, 2, first, second, neutral, first_leg=(leg.home_goals, leg.away_goals))
    return Tie(name, legs, first, second, neutral)


def _find_tie_winner(cup: CupSeason, matches: list[CupMatch], sides: tuple[int, int]) -> int:
    """The tie's winner from its goals (extra time included), then the season's away goals; a
    tie still level went to penalties: the world's winner of the tie, or of its last match."""
    goals = dict.fromkeys(sides, 0)
    for match in matches:
        goals[match.home] += match.home_goals
        goals[match.away] += match.away_goals
    if goals[sides[0]] != goals[sides[1]]:
        return max(sides, key=goals.get)
    if len(matches) == 2 and cup.tie_rule.away_goals:
        away = {match.away: match.away_goals for match in matches}
        if away[sides[0]] != away[sides[1]]:
            return max(sides, key=away.get)
    tie_id = matches[0].tie_id
    if tie_id is not None and tie_id in cup.tie_winners:
        return cup.tie_winners[tie_id]
    if matches[-1].winner is not None:
        return matches[-1].winner
    raise ValueError(f"{cup.code} {cup.season}: a tie level with no winner in the world")


def find_tie_winner_disagreements(cup: CupSeason) -> list[tuple[str, int, int, int]]:
    """The ties whose winner in the world's `match_ties` is not the one their goals give:
    (round, tie, the world's winner, the goals' winner)."""
    found = []
    for name, _ in cup.form.rounds:
        for matches in _collect_world_ties(cup, name):
            tie_id = matches[0].tie_id
            if tie_id is None or tie_id not in cup.tie_winners or not all(
                    match.played for match in matches):  # fmt: skip
                continue
            ours = _find_tie_winner(cup, matches, (matches[0].home, matches[0].away))
            if ours != cup.tie_winners[tie_id]:
                found.append((name, tie_id, cup.tie_winners[tie_id], ours))
    return found


def _find_awarded_ties(cup, name, node_of, nodes, final_order, missing) -> list[tuple[int, int]]:
    """The ties of round `name` awarded without a match: the clubs that should have played it
    and did not, the one that played on the winner."""
    if final_order is None:
        raise ValueError(f"{cup.code} {cup.season} {name}: a tie is missing before the phase ends")
    should = set(find_direct_entrants(cup.form, final_order).get(name, ()))
    names = [round_name for round_name, _ in cup.form.rounds]
    if names.index(name) > 0:
        previous = names[names.index(name) - 1]
        should |= {node.winner for node in set(node_of.values())
                   if isinstance(node, Tie) and node.round_name == previous}  # fmt: skip
    playing = {
        team for node in nodes for team in (_resolve_side(node.first), _resolve_side(node.second))
    }
    absent = sorted(should - playing - {None})
    later = {team for match in cup.knockout
             if names.index(match.round_name) > names.index(name)
             for team in (match.home, match.away)}  # fmt: skip
    winners = [team for team in absent if team in later]
    losers = [team for team in absent if team not in later]
    if len(winners) != missing or len(losers) != missing:
        raise ValueError(f"{cup.code} {cup.season} {name}: {missing} ties missing, the clubs"
                         f" {absent} do not make them")  # fmt: skip
    return list(zip(winners, losers, strict=True))


def _resolve_side(node: Tie | int) -> int | None:
    return node if isinstance(node, int) else node.winner


def find_direct_entrants(
    form: TournamentFormat, order: dict[str, list[int]]
) -> dict[str, list[int]]:
    """The clubs each round takes from the phase's final places."""
    entrants: dict[str, list[int]] = {}
    for name, (low, high) in form.advance.items():
        for group in sorted(order):
            entrants.setdefault(name, []).extend(order[group][low - 1: high])
    return entrants


# ---- the joint simulation ------------------------------------------------------------------


@dataclass(frozen=True)
class CupDraws:
    """Per club of the cup still in or in its phase, the index of its outcome in
    `outcomes` in every simulation; per club of the phase its place (1 the first)."""

    outcomes: tuple[str, ...]
    reached: dict[int, np.ndarray]
    places: dict[int, np.ndarray]


def simulate_cup(
    cup: CupSeason,
    source: CupSeason | None,
    instant: str,
    strength: Strength,
    parameters: TieParameters,
    draws: int,
    seed: int,
) -> CupDraws:
    """`cup` at `instant`, with `source`, the cup its drop-ins come from (its phase drawn in the
    same simulations), under `strength`."""
    generator = np.random.default_rng(seed)
    form = cup.form
    tables = build_phase_tables(cup, instant, strength)
    drawn = {group: simulate_league(table, cup.table_rule, draws,
                                    int(generator.integers(2**31)))
             for group, table in tables.items()}  # fmt: skip
    source_tables = source_drawn = None
    if form.drop_in is not None:
        if source is None or source.code != form.drop_in.source:
            raise ValueError(f"{cup.code} {cup.season} takes clubs from {form.drop_in.source}")
        source_tables = build_phase_tables(source, instant, strength)
        source_drawn = {group: simulate_league(table, source.table_rule, draws,
                                               int(generator.integers(2**31)))
                        for group, table in source_tables.items()}  # fmt: skip
    final_order = None
    if is_phase_over(tables):
        final_order = {group: _order(tables[group], result.places[0])
                       for group, result in drawn.items()}  # fmt: skip
    by_places = form.bracket in SET_BY_PLACES
    thirds = (_choose_best_thirds(form, drawn, generator)
              if form.best_thirds else [()] * draws)  # fmt: skip
    if by_places:
        state = KnownKnockout(None, ())
        known = [match for match in cup.knockout if is_known_match(match, instant)]
        if known:  # the knockout has begun: its first round fixes the places lots decided
            drawn, thirds = _keep_consistent_simulations(
                cup, tables, drawn, thirds, known, generator
            )
    else:
        state = find_known_knockout(cup, instant, final_order)
    sampler = TieSampler(strength, cup.tie_rule, parameters, generator)
    round_index = cup.round_index
    phase_teams = [team for table in tables.values() for team in table.team_ids]
    reached: dict[int, np.ndarray] = {team: np.zeros(draws, dtype=np.int64)
                                      for team in phase_teams}  # fmt: skip
    places = {team: np.zeros(draws, dtype=np.int64) for team in phase_teams}
    for group, result in drawn.items():
        for column, team in enumerate(tables[group].team_ids):
            places[team] = result.places[:, column] + 1
    drop_ins_known = (form.drop_in is not None and source_tables is not None
                      and is_phase_over(source_tables))  # fmt: skip
    for simulation in range(draws):
        order = {group: _order(tables[group], result.places[simulation])
                 for group, result in drawn.items()}  # fmt: skip
        drop_ins: tuple[int, ...] = ()
        if form.drop_in is not None:
            drop_ins = _drop_ins(form, source_tables, source_drawn, simulation, generator)
        if by_places:
            tree = build(form.bracket, order, form.rounds, generator, tuple(thirds[simulation]),
                         neutral_rounds=form.neutral_rounds, fixed=form.fixed)  # fmt: skip
            tree = apply_known_results(tree, cup, instant)
        else:
            tree = _build_knockout_tree(cup, state, order, drop_ins, generator)
        furthest: dict[int, int] = {}
        play(tree, sampler, furthest, round_index)
        for team, index in furthest.items():
            if team not in reached:
                if not drop_ins_known:
                    continue  # a club of the higher cup, not yet this cup's
                reached[team] = np.zeros(draws, dtype=np.int64)
            reached[team][simulation] = index
    return CupDraws(form.outcomes, reached, places)


def is_known_match(match: CupMatch, instant: str) -> bool:
    return is_result_known(match, instant)


def apply_known_results(node: Tie | int, cup: CupSeason, instant: str) -> Tie | int:
    """A bracket set by the places, with the winner of every match the world holds played
    before `instant` between the two teams the bracket gives."""
    if isinstance(node, int):
        return node
    first = apply_known_results(node.first, cup, instant)
    second = apply_known_results(node.second, cup, instant)
    sides = (_resolve_side(first), _resolve_side(second))
    winner = None
    if None not in sides:
        for match in cup.knockout:
            if (match.round_name == node.round_name and {match.home, match.away} == set(sides)
                    and is_result_known(match, instant)):  # fmt: skip
                winner = _find_tie_winner(cup, [match], (match.home, match.away))
    return Tie(node.round_name, node.legs, first, second, node.neutral, winner=winner)


def _keep_consistent_simulations(cup, tables, drawn, thirds, known_matches, generator):
    """Once the knockout has begun the group places are settled: keep the simulations whose
    places give the world's first-round pairings (lots and criteria the world does not hold, the
    card score, may have decided a place); refused when none does."""
    first_round = cup.form.rounds[0][0]
    pairs = {frozenset((match.home, match.away)) for match in known_matches
             if match.round_name == first_round}  # fmt: skip
    count = len(next(iter(drawn.values())).places)
    good = []
    for simulation in range(count):
        order = {group: _order(tables[group], result.places[simulation])
                 for group, result in drawn.items()}  # fmt: skip
        tree = build(cup.form.bracket, order, cup.form.rounds, generator,
                     tuple(thirds[simulation]), fixed=cup.form.fixed)  # fmt: skip
        given = {frozenset((_resolve_side(node.first), _resolve_side(node.second)))
                 for node in _round(tree, first_round)}  # fmt: skip
        if pairs <= given:
            good.append(simulation)
    if not good:
        raise ValueError(f"{cup.code} {cup.season}: no ranking of the groups gives the world's"
                         f" {first_round} pairings")  # fmt: skip
    chosen = np.array([good[index % len(good)] for index in range(count)])
    return ({group: _take(result, chosen) for group, result in drawn.items()},
            [thirds[index] for index in chosen])  # fmt: skip


def _take(result: SeasonDraws, rows: np.ndarray) -> SeasonDraws:
    return SeasonDraws(result.points[rows], result.places[rows], result.goal_difference[rows],
                       result.goals_for[rows])  # fmt: skip


def _round(node, name) -> list[Tie]:
    if isinstance(node, int):
        return []
    own = [node] if node.round_name == name else []
    return own + _round(node.first, name) + _round(node.second, name)


def _order(table: LeagueSeason, places: np.ndarray) -> list[int]:
    return [table.team_ids[column] for column in np.argsort(places, kind="stable")]


def _drop_ins(form, tables, drawn, simulation, generator) -> tuple[int, ...]:
    """The clubs of the drop-in's place in each group of the higher cup, its best `seeded`
    first by the format's ranking (the criteria the draws hold, then lots)."""
    rule = form.drop_in
    clubs, keys = [], []
    for group in sorted(tables):
        result: SeasonDraws = drawn[group]
        column = int(np.flatnonzero(result.places[simulation] == rule.place - 1)[0])
        clubs.append(tables[group].team_ids[column])
        key = []
        for criterion in rule.ranking:
            if criterion == "lots":
                key.append(float(generator.random()))
            else:
                key.append(float(getattr(result, criterion)[simulation, column]))
        keys.append(tuple(key))
    if not rule.seeded:
        return tuple(clubs)
    ranked = sorted(range(len(clubs)), key=lambda index: keys[index], reverse=True)
    return tuple(clubs[index] for index in ranked)


def _build_knockout_tree(
    cup: CupSeason,
    state: KnownKnockout,
    order: dict[str, list[int]],
    drop_ins: tuple[int, ...],
    generator: np.random.Generator,
) -> Tie:
    """The knockout of one simulation: the known draws, the rest drawn by the format's rule."""
    form = cup.form
    names = [name for name, _ in form.rounds]
    if state.last_round is None:
        return build(form.bracket, order, form.rounds, generator, drop_ins=drop_ins,
                     neutral_rounds=form.neutral_rounds)  # fmt: skip
    if state.last_round == names[-1]:
        return state.nodes[0]
    after = form.rounds[names.index(state.last_round) + 1:]
    if state.last_round == "knockout_play_offs" and form.bracket == "uefa_league_phase_2024":
        play_offs = _play_offs_by_seeds(order[""], state.nodes)
        tree = build_league_phase_from_play_offs(order[""], form.rounds, generator, play_offs)
    elif state.last_round == "knockout_play_offs" and form.bracket == "uefa_knockout_play_offs":
        winners = [order[group][0] for group in sorted(order)]
        tree = build_winners_v_play_offs(winners, list(state.nodes), after, generator)
    else:
        tree = open_draws(list(state.nodes), after, generator)
    return mark_neutral_rounds(tree, form.neutral_rounds) if form.neutral_rounds else tree


def _play_offs_by_seeds(order: list[int], nodes) -> dict[tuple[int, int], list[Tie]]:
    place = {team: index + 1 for index, team in enumerate(order)}
    play_offs: dict[tuple[int, int], list[Tie]] = {}
    for node in nodes:
        seeded = min(place[node.first], place[node.second])
        for seeds, _ in SEEDED_PAIRS_2024:
            if seeded in seeds:
                play_offs.setdefault(seeds, []).append(node)
    if sorted(len(ties) for ties in play_offs.values()) != [2, 2, 2, 2]:
        raise ValueError("the knockout play-offs do not follow the places 9-24")
    return play_offs


# ---- what happened, and what is settled ----------------------------------------------------


def rank_final_phase(cup: CupSeason, seed: int, tries: int = 200) -> dict[str, list[int]]:
    """The final table of each group (or the league phase) from the official results, ranked by
    the season's rules (lots drawn with `seed`). Where the bracket is set by the places, the lots
    stand for criteria the world does not hold (the card score): the order kept is the first, over
    `tries` draws of the lots, that gives the world's first-round pairings."""
    tables = build_phase_tables(cup, "9999", lambda *_: (np.zeros((SIZE, SIZE)), 0.0, 0.0))
    if not is_phase_over(tables):
        raise ValueError(f"{cup.code} {cup.season}: the phase is not over in the world")
    first_round = cup.form.rounds[0][0]
    pairs = {frozenset((match.home, match.away)) for match in cup.knockout
             if match.round_name == first_round}  # fmt: skip
    if cup.form.bracket not in SET_BY_PLACES or not pairs:  # drawn knockouts: lots drawn once
        return {group: _order(table, simulate_league(table, cup.table_rule, 1, seed).places[0])
                for group, table in tables.items()}  # fmt: skip
    drawn = {group: simulate_league(table, cup.table_rule, tries, seed)
             for group, table in tables.items()}  # fmt: skip
    thirds = (_choose_best_thirds(cup.form, drawn, np.random.default_rng(seed))
              if cup.form.best_thirds else [()] * tries)  # fmt: skip
    for attempt in range(tries if cup.form.bracket in SET_BY_PLACES and pairs else 1):
        order = {group: _order(tables[group], result.places[attempt])
                 for group, result in drawn.items()}  # fmt: skip
        if cup.form.bracket not in SET_BY_PLACES or not pairs:
            return order
        tree = build(cup.form.bracket, order, cup.form.rounds, np.random.default_rng(seed),
                     tuple(thirds[attempt]), fixed=cup.form.fixed)  # fmt: skip
        if pairs == {frozenset((_resolve_side(node.first), _resolve_side(node.second)))
                     for node in _round(tree, first_round)}:  # fmt: skip
            return order
    raise ValueError(f"{cup.code} {cup.season}: no ranking of the groups gives the world's"
                     f" {first_round} pairings")


def find_outcomes_reached(cup: CupSeason, order: dict[str, list[int]],
                          drop_ins: tuple[int, ...] = ()) -> dict[int, int]:  # fmt: skip
    """Per club of the phase and every drop-in, the index of the outcome it reached."""
    outcomes = cup.form.outcomes
    round_index = cup.round_index
    reached = {team: 0 for teams in order.values() for team in teams}
    for team in drop_ins:
        reached[team] = 1
    if cup.form.bracket not in SET_BY_PLACES:  # a tie awarded without a match (the cups)
        for name, teams in find_direct_entrants(cup.form, order).items():
            for team in teams:
                reached[team] = max(reached[team], round_index[name])
    for match in cup.knockout:
        for team in (match.home, match.away):
            reached[team] = max(reached.get(team, 0), round_index[match.round_name])
    final = [match for match in cup.knockout if match.round_name == "final"]
    if final:
        sides = (final[0].home, final[0].away)
        reached[_find_tie_winner(cup, final, sides)] = outcomes.index("winner")
    return reached


def find_clubs_still_in(cup: CupSeason, tables: dict[str, LeagueSeason],
                        state: KnownKnockout) -> set[int]:  # fmt: skip
    """The clubs whose outcome in the cup is not yet settled at `instant`, by the facts: a club of
    the phase that can still reach a place that goes on, a club in the knockout that has not lost
    a known tie, the final's winner excluded."""
    going_on = max(high for _, high in cup.form.advance.values())
    alive: set[int] = set()
    if state.last_round is None:  # in the phase: a place that goes on still within reach
        for table in tables.values():
            best, _ = place_bounds(table)
            alive |= {team for column, team in enumerate(table.team_ids)
                      if best[column] <= going_on}  # fmt: skip
        return alive

    def collect_alive(node) -> None:
        if isinstance(node, int):
            alive.add(node)
            return
        collect_alive(node.first)
        collect_alive(node.second)
        if node.winner is not None:
            loser = {_resolve_side_team(node.first), _resolve_side_team(node.second)} - {
                node.winner
            }
            alive.difference_update(loser)
            if node.round_name == "final":
                alive.discard(node.winner)

    for node in state.nodes:
        collect_alive(node)
    return alive


def _resolve_side_team(node) -> int | None:
    """The club that came out of `node`: itself, or the winner of its tie."""
    return node if isinstance(node, int) else node.winner


