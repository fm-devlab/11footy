"""A league table ranked by its competition's rules of that season, in every simulated season.

The rules are read from `config/vesta/tie-break-rules.csv`, one row per competition and range of
seasons, never assumed: a competition and season without a row is refused. The criteria, applied
in order to the teams still level:
- `points`, `goal_difference`, `goals_for`, `away_goals_for`, `wins`, `away_wins`: over the whole
  season;
- `head_to_head_points`, `head_to_head_goal_difference`, `head_to_head_goals_for`,
  `head_to_head_away_goals_for`: over the matches between the teams level when the first of a run
  of head-to-head criteria is reached; the table among them is computed once and serves the whole
  run (Serie A's "classifica avulsa", FIFA's "the second step does not restart");
- `restart_head_to_head`, after a run of head-to-head criteria: teams still level, fewer than those
  the run began with, go through the run again among themselves alone (UEFA's reapplication);
- `restart_on_separation`, anywhere in the list: whenever a criterion separates some of the level
  teams, those still level start the list again among themselves (LaLiga: the rules "according to
  their number");
- `opponents_points`, `opponents_goal_difference`, `opponents_goals_for`: the sums of the season
  criteria of every opponent met (a league phase, where the teams meet different opponents);
- `play_off=<places>`: the teams still level are ordered by the criteria after it, then for each
  listed place p held by one of them, with p + 1 held by another, those two play for it, at even
  chances (no forecast of a play-off is made);
- `deciding_match`: a match on neutral ground between two teams still level, at even chances (for
  more than two, the regulations say nothing: a random order);
- `penalties_if_last_match`: two teams still level that met in the last match of both go to
  penalties, at even chances (the AFC); otherwise the next criterion;
- `play_off_unless_uefa_finalist`: no play-off for a team the run names a UEFA finalist
  (`play_off_exempt`; Serie A from 2026-27): the order of the criteria after it stands; until
  the finals are known the run names none;
- `shared`: the teams still level share their place (the Premier League's "deemed to occupy the
  same position"): each holds the first of their places;
- `lots`: whatever the world does not hold (fair play, discipline, coefficients, a drawing of
  lots), drawn at random.
A row may give two teams level their own list (`criteria_two`, CAF), used when exactly two are
level on the season criteria the lists open with.
A row is `verified` (`yes`) once the rules are confirmed against the competition's regulations;
until then (`no`) the simulation runs on it for research, and the publication refuses it. A row
`carried_forward` holds last season's verified rule for a season whose regulations cannot be read
yet; the season job publishes on it only for the seasons a decision names (`CARRIED_FORWARD`),
and the page says so.
"""

import csv
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np

TIE_BREAK_RULES_FILE = (
    Path(__file__).resolve().parents[3] / "config" / "vesta" / "tie-break-rules.csv"
)
SEASON_CRITERIA = ("points", "goal_difference", "goals_for", "away_goals_for", "wins",
                   "away_wins")  # fmt: skip
HEAD_TO_HEAD = ("head_to_head_points", "head_to_head_goal_difference", "head_to_head_goals_for",
                "head_to_head_away_goals_for")  # fmt: skip
OPPONENTS = ("opponents_points", "opponents_goal_difference", "opponents_goals_for")
# The seasons whose tie-break row may carry last season's rule forward, and the decision that
# allows it: their regulations were not readable as primary texts.
CARRIED_FORWARD = {
    ("PL", 2026): "docs/decisions/2026-10-28-season-release-and-tie-breaks.md",
    ("L1", 2026): "docs/decisions/2026-10-28-season-release-and-tie-breaks.md",
}
MARKERS = ("lots", "shared", "deciding_match", "restart_head_to_head",
           "play_off_unless_uefa_finalist",
           "restart_on_separation", "penalties_if_last_match")
OFFSET = 500  # a criterion's value plus this is positive, and below 1,000, in the sort key


@dataclass(frozen=True)
class TieBreakRule:
    """`criteria_two`, when the regulations give two teams level their own list (CAF), replaces
    `criteria` for a group of exactly two level on the leading season criteria."""

    criteria: tuple[str, ...]
    verified: bool
    criteria_two: tuple[str, ...] = ()
    # the teams no play-off applies to (Serie A from 2026-27: a UEFA finalist, the order of the
    # criteria after the play-off decides), given by the run; columns of the table in `rank`,
    # team ids in `vesta.season.league.simulate_league`
    play_off_exempt: frozenset[int] = frozenset()
    carried_forward: bool = False  # last season's rule, by a decision (`CARRIED_FORWARD`)

    @property
    def publishable(self) -> bool:
        """Verified, or carried forward where a decision allows it."""
        return self.verified or self.carried_forward


def read_tie_break_rule(
    competition: str, season: int, path: Path = TIE_BREAK_RULES_FILE
) -> TieBreakRule:
    """The order of a league table of `competition` in `season`; refused when the table has none."""
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if (row["competition"] == competition
                    and int(row["first_season"]) <= season <= int(row["last_season"])):  # fmt: skip
                if None in row or row["verified"] not in ("yes", "no", "carried_forward"):
                    raise ValueError(f"{path.name}: a malformed row for {competition} (a comma"
                                     " outside quotes?)")  # fmt: skip
                criteria = tuple(row["criteria"].split("|"))
                two = tuple(row["criteria_two"].split("|")) if row.get("criteria_two") else ()
                for criterion in criteria + two:
                    _check_criterion(criterion)
                if two and _take_leading_criteria(two) != _take_leading_criteria(criteria):
                    raise ValueError(f"{competition} {season}: the list for two teams must open"
                                     " with the same season criteria")  # fmt: skip
                carried = row["verified"] == "carried_forward"
                if carried and (competition, season) not in CARRIED_FORWARD:
                    raise ValueError(f"{competition} {season}: carried forward without a decision"
                                     " that allows it (CARRIED_FORWARD)")  # fmt: skip
                return TieBreakRule(criteria, row["verified"] == "yes", two,
                                    carried_forward=carried)
    raise ValueError(f"no tie-break rules for {competition} in {season}: write them in {path.name}")


@dataclass(frozen=True)
class SeasonMatches:
    """Every match of the season, played or drawn: the teams' columns (m,) and the goals (n, m)
    of each simulated season."""

    home: np.ndarray
    away: np.ndarray
    home_goals: np.ndarray
    away_goals: np.ndarray


def count_season_totals(matches: SeasonMatches, teams: int, deductions: np.ndarray) -> dict:
    """The season criteria of every team in every simulated season, (n, teams) each."""
    count = matches.home_goals.shape[0]
    totals = {name: np.zeros((count, teams), dtype=np.int64) for name in SEASON_CRITERIA}
    home_won = matches.home_goals > matches.away_goals
    away_won = matches.away_goals > matches.home_goals
    drawn = ~home_won & ~away_won
    for side, own, other, won in ((matches.home, matches.home_goals, matches.away_goals, home_won),
                                  (matches.away, matches.away_goals, matches.home_goals,
                                   away_won)):  # fmt: skip
        for column in range(teams):
            mine = side == column
            if not mine.any():
                continue
            totals["points"][:, column] += (3 * won[:, mine] + drawn[:, mine]).sum(axis=1)
            totals["goal_difference"][:, column] += (own[:, mine] - other[:, mine]).sum(axis=1)
            totals["goals_for"][:, column] += own[:, mine].sum(axis=1)
            totals["wins"][:, column] += won[:, mine].sum(axis=1)
            if side is matches.away:
                totals["away_goals_for"][:, column] += own[:, mine].sum(axis=1)
                totals["away_wins"][:, column] += won[:, mine].sum(axis=1)
    totals["points"] -= np.asarray(deductions, dtype=np.int64)[None, :]
    return totals


def rank(totals: dict, matches: SeasonMatches, rule: TieBreakRule,
         generator: np.random.Generator) -> np.ndarray:  # fmt: skip
    """(n, teams) every team's final place, 0 the first; teams sharing a place hold the same."""
    count, teams = totals["points"].shape
    leading = len(_take_leading_criteria(rule.criteria))
    key = np.zeros((count, teams), dtype=np.int64)
    for criterion in rule.criteria[:leading]:
        key = key * 1000 + totals[criterion] + OFFSET
    order = np.argsort(-key, axis=1, kind="stable")
    ordered_keys = np.take_along_axis(key, order, axis=1)
    places = np.empty_like(order)
    places[np.arange(count)[:, None], order] = np.arange(teams)[None, :]
    level = (ordered_keys[:, 1:] == ordered_keys[:, :-1]).any(axis=1)
    pairs = _index_pairs(matches)
    context = _Context(rule.criteria, totals, matches, pairs, generator, rule.play_off_exempt)
    context_two = _Context(rule.criteria_two or rule.criteria, totals, matches, pairs, generator,
                           rule.play_off_exempt)  # fmt: skip
    for simulation in np.flatnonzero(level):
        row = order[simulation]
        start = 0
        while start < teams:
            end = start + 1
            while end < teams and ordered_keys[simulation, end] == ordered_keys[simulation, start]:
                end += 1
            if end - start > 1:
                chosen = context_two if end - start == 2 else context
                blocks = _order(list(row[start:end]), start, leading, None, chosen, simulation)
                position = start
                for block in blocks:
                    for team in block:
                        places[simulation, team] = position
                    position += len(block)
            start = end
    return places


@dataclass(frozen=True)
class _Context:
    criteria: tuple[str, ...]
    totals: dict
    matches: SeasonMatches
    pairs: dict
    generator: np.random.Generator
    exempt: frozenset[int] = frozenset()

    @property
    def restart_on_separation(self) -> bool:
        return "restart_on_separation" in self.criteria

    @cached_property
    def last_match(self) -> dict[int, int]:
        """Each team's last match (the matches run in the order of the season)."""
        last: dict[int, int] = {}
        for (home, away), indexes in self.pairs.items():
            for team in (home, away):
                last[team] = max(last.get(team, -1), max(indexes))
        return last


def _order(group, start, index, head_to_head_among, context, simulation) -> list[list[int]]:
    """The teams of `group`, level from place `start` (0 the first), as blocks in their final
    order (a block of more than one shares its place), from the criterion at `index` on;
    `head_to_head_among` the teams of the run of head-to-head criteria under way."""
    criteria = context.criteria
    if len(group) == 1:
        return [group]
    if index >= len(criteria):
        return [[team] for team in group]
    criterion = criteria[index]
    generator = context.generator
    if criterion == "play_off_unless_uefa_finalist":  # the exemption is the rule's; no order
        return _order(group, start, index + 1, head_to_head_among, context, simulation)
    if criterion == "restart_on_separation":
        return _order(group, start, index + 1, head_to_head_among, context, simulation)
    if criterion == "restart_head_to_head":
        if head_to_head_among is not None and len(group) < len(head_to_head_among):
            first = index
            while first > 0 and criteria[first - 1] in HEAD_TO_HEAD:
                first -= 1
            return _order(group, start, first, None, context, simulation)
        return _order(group, start, index + 1, None, context, simulation)
    if criterion in ("lots", "deciding_match"):
        return [[group[position]] for position in generator.permutation(len(group))]
    if criterion == "shared":
        return [group]
    if criterion == "penalties_if_last_match":
        if len(group) == 2:
            first, second = group
            met = context.pairs.get((first, second), []) + context.pairs.get((second, first), [])
            last = context.last_match
            if last.get(first) == last.get(second) and last.get(first) in met:
                return [[group[position]] for position in generator.permutation(2)]
        return _order(group, start, index + 1, head_to_head_among, context, simulation)
    if criterion.startswith("play_off="):
        blocks = _order(group, start, index + 1, None, context, simulation)
        return _play_off_places(blocks, start, _place_list(criterion), generator, context.exempt)
    among = None
    if criterion in HEAD_TO_HEAD:
        among = head_to_head_among if head_to_head_among is not None else group
        values = _count_head_to_head(among, criterion, context.matches, context.pairs, simulation)
    elif criterion in OPPONENTS:
        base = context.totals[criterion.removeprefix("opponents_")][simulation]
        values = dict.fromkeys(group, 0)
        for (home, away), indexes in context.pairs.items():
            if home in values:
                values[home] += int(base[away]) * len(indexes)
            if away in values:
                values[away] += int(base[home]) * len(indexes)
    else:
        values = {team: int(context.totals[criterion][simulation, team]) for team in group}
    ordered = sorted(group, key=lambda team: -values[team])
    result, position = [], 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[position]]:
            end += 1
        level = ordered[position:end]
        if context.restart_on_separation and 1 < len(level) < len(group):
            result += _order(level, start + position, 0, None, context, simulation)
        else:
            result += _order(level, start + position, index + 1, among, context, simulation)
        position = end
    return result


def _play_off_places(blocks, start, places, generator, exempt=frozenset()) -> list[list[int]]:
    """The teams at a listed place p and at p + 1 (1 the first) play for it at even chances,
    unless one of them is exempt (the order already given stands)."""
    teams = [team for block in blocks for team in block]
    block_of = [number for number, block in enumerate(blocks) for _ in block]
    separate = len(blocks)
    for place in sorted(places):
        here = place - 1 - start
        if here >= 0 and here + 1 < len(teams) and not {teams[here], teams[here + 1]} & exempt:
            if generator.random() < 0.5:
                teams[here], teams[here + 1] = teams[here + 1], teams[here]
            block_of[here], block_of[here + 1] = separate, separate + 1  # no longer sharing
            separate += 2
    result: list[list[int]] = []
    for position, team in enumerate(teams):
        if position and block_of[position] == block_of[position - 1]:
            result[-1].append(team)
        else:
            result.append([team])
    return result


def _take_leading_criteria(criteria: tuple[str, ...]) -> tuple[str, ...]:
    """The season criteria the list opens with (ranked at once for every simulation)."""
    count = 0
    while count < len(criteria) and criteria[count] in SEASON_CRITERIA:
        count += 1
    return criteria[:count]


def _place_list(criterion: str) -> list[int]:
    return [int(place) for place in criterion.split("=")[1].split(",")]


def _count_head_to_head(group, criterion, matches, pairs, simulation) -> dict[int, int]:
    values = dict.fromkeys(group, 0)
    for home in group:
        for away in group:
            for match in pairs.get((home, away), ()):
                home_goals = int(matches.home_goals[simulation, match])
                away_goals = int(matches.away_goals[simulation, match])
                if criterion == "head_to_head_points":
                    values[home] += 3 if home_goals > away_goals else int(home_goals == away_goals)
                    values[away] += 3 if away_goals > home_goals else int(home_goals == away_goals)
                elif criterion == "head_to_head_goal_difference":
                    values[home] += home_goals - away_goals
                    values[away] += away_goals - home_goals
                elif criterion == "head_to_head_goals_for":
                    values[home] += home_goals
                    values[away] += away_goals
                else:  # head_to_head_away_goals_for
                    values[away] += away_goals
    return values


def _index_pairs(matches: SeasonMatches) -> dict[tuple[int, int], list[int]]:
    pairs: dict[tuple[int, int], list[int]] = {}
    for index, (home, away) in enumerate(zip(matches.home.tolist(), matches.away.tolist(),
                                             strict=True)):  # fmt: skip
        pairs.setdefault((home, away), []).append(index)
    return pairs


def _check_criterion(criterion: str) -> None:
    if criterion in SEASON_CRITERIA + HEAD_TO_HEAD + OPPONENTS + MARKERS:
        return
    if criterion.startswith("play_off=") and all(
        place.isdigit() for place in criterion.split("=")[1].split(",")
    ):
        return
    raise ValueError(f"unknown tie-break criterion {criterion!r}")
