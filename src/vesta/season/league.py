"""A league season simulated to its end (T1): final positions and points of every team.

Every match still to play is drawn from the spine's score matrix of that match, independently,
`draws` times from one seeded generator, and added to the table as it stands (the matches played,
the points deducted); each simulated table is ranked by the competition's rules of that season
(`vesta.season.standings`). Strengths stay those of the matrices: no injuries, transfers or form to
come.

A line is decided by exact reasoning on the points, never by its share of the simulations: a team
is surely above another when its points now exceed the other's most points still possible, surely
below when its most points still possible fall short of the other's points now (level on points,
the rules could go either way); a position or points line that holds for every place and points
total still possible, or for none, is decided. The simulated shares agree, as no simulation can
break those bounds.
"""

from dataclasses import dataclass, replace
from functools import cache

import numpy as np

from vesta.forecast_log.writing import PublishedDistribution
from vesta.forecast_reading import read_line
from vesta.quantities import QuantityDefinition
from vesta.season.standings import SeasonMatches, TieBreakRule, count_season_totals, rank
from vesta.spine.score_matrix import SIZE

POSITION, POINTS = "T1.final_position", "T1.points"


@dataclass(frozen=True)
class LeagueSeason:
    """A season as it stands: its teams, the matches played (home and away team ids, goals), the
    points deducted by team, and the matches still to play with their matrices (m, 11, 11)."""

    team_ids: tuple[int, ...]
    played: tuple[tuple[int, int, int, int], ...]
    deductions: dict[int, int]
    fixtures: tuple[tuple[int, int], ...]
    matrices: np.ndarray


@dataclass(frozen=True)
class SeasonDraws:
    """(n, teams) the final points, places (0 the first), goal difference and goals for of every
    simulated season, the teams in the order of `LeagueSeason.team_ids`."""

    points: np.ndarray
    places: np.ndarray
    goal_difference: np.ndarray
    goals_for: np.ndarray
    matches: SeasonMatches | None = None  # every match of every simulation, as ranked


def simulate_league(season: LeagueSeason, rule: TieBreakRule, draws: int,
                    seed: int) -> SeasonDraws:  # fmt: skip
    generator = np.random.default_rng(seed)
    column = {team: index for index, team in enumerate(season.team_ids)}
    played = np.array(season.played, dtype=np.int64).reshape(-1, 4)
    fixtures = np.array(season.fixtures, dtype=np.int64).reshape(-1, 2)
    drawn_home = np.zeros((draws, len(fixtures)), dtype=np.int64)
    drawn_away = np.zeros((draws, len(fixtures)), dtype=np.int64)
    matrices = np.asarray(season.matrices, dtype=float).reshape(-1, SIZE, SIZE)
    for index, matrix in enumerate(matrices):
        cumulative = np.cumsum(matrix.ravel() / matrix.sum())
        cell = np.minimum(np.searchsorted(cumulative, generator.random(draws), side="right"),
                          SIZE * SIZE - 1)  # fmt: skip
        drawn_home[:, index], drawn_away[:, index] = cell // SIZE, cell % SIZE
    matches = SeasonMatches(
        home=np.array([column[team] for team in np.r_[played[:, 0], fixtures[:, 0]].tolist()]),
        away=np.array([column[team] for team in np.r_[played[:, 1], fixtures[:, 1]].tolist()]),
        home_goals=np.hstack([np.broadcast_to(played[:, 2], (draws, len(played))), drawn_home]),
        away_goals=np.hstack([np.broadcast_to(played[:, 3], (draws, len(played))), drawn_away]),
    )
    deductions = np.array([season.deductions.get(team, 0) for team in season.team_ids])
    totals = count_season_totals(matches, len(season.team_ids), deductions)
    if rule.play_off_exempt:  # the run names team ids; the table ranks columns
        rule = replace(rule, play_off_exempt=frozenset(
            column[team] for team in rule.play_off_exempt if team in column))  # fmt: skip
    places = rank(totals, matches, rule, generator)
    return SeasonDraws(totals["points"], places, totals["goal_difference"], totals["goals_for"],
                       matches)


def build_league_distributions(
    season: LeagueSeason,
    draws: SeasonDraws,
    model_version: str,
    definitions: dict[str, QuantityDefinition],
) -> list[PublishedDistribution]:
    """`T1.final_position` and `T1.points` of every team, each with its decided lines."""
    teams = len(season.team_ids)
    now, most = _count_points_now_and_most(season)
    best, worst = place_bounds(season)
    published = []
    for column, team in enumerate(season.team_ids):
        places = np.bincount(draws.places[:, column], minlength=teams) / len(draws.places)
        # a count holds no negative total: a deduction beyond every point won reads as 0
        points = np.bincount(np.maximum(draws.points[:, column], 0)) / len(draws.points)
        reachable = range(int(best[column]), int(worst[column]) + 1)
        published.append(PublishedDistribution(
            POSITION, team, model_version, places.tolist(),
            decided_lines=_find_decided_lines(definitions[POSITION], "rank",
                                              [_place(place, teams) for place in reachable])))
        published.append(
            PublishedDistribution(
                POINTS,
                team,
                model_version,
                {"probabilities": points.tolist(), "tail_from": len(points), "tail": 0.0},
                decided_lines=_find_decided_lines(
                    definitions[POINTS],
                    "count",
                    [
                        _make_certain_total(max(total, 0))
                        for total in range(now[column], most[column] + 1)
                    ],
                ),
            )
        )
    return published  # fmt: skip


def place_bounds(season: LeagueSeason) -> tuple[np.ndarray, np.ndarray]:
    """Per team (in `team_ids` order), the best and worst place still possible (1 the first),
    from the points now and the most points still possible."""
    now, most = _count_points_now_and_most(season)
    teams = len(season.team_ids)
    best = np.array([1 + int((now > most[column]).sum()) for column in range(teams)])
    worst = np.array([teams - int((most < now[column]).sum()) for column in range(teams)])
    return best, worst


def _count_points_now_and_most(season: LeagueSeason) -> tuple[np.ndarray, np.ndarray]:
    column = {team: index for index, team in enumerate(season.team_ids)}
    now = np.array([-season.deductions.get(team, 0) for team in season.team_ids])
    for home, away, home_goals, away_goals in season.played:
        now[column[home]] += 3 if home_goals > away_goals else int(home_goals == away_goals)
        now[column[away]] += 3 if away_goals > home_goals else int(home_goals == away_goals)
    most = now.copy()
    for home, away in season.fixtures:
        most[column[home]] += 3
        most[column[away]] += 3
    return now, most


def _find_decided_lines(
    definition: QuantityDefinition, kind: str, reachable: list
) -> tuple[str, ...]:
    """The lines whose answer is the same for every outcome still reachable."""
    decided = []
    for line in definition.quantity.lines:
        answers = {round(read_line(kind, values, line)) for values in reachable}
        if len(answers) == 1:
            decided.append(line)
    return tuple(decided)


@cache
def _place(place: int, teams: int) -> list[float]:
    return [1.0 if index == place - 1 else 0.0 for index in range(teams)]


def _make_certain_total(total: int) -> dict:
    probabilities = [0.0] * total + [1.0]
    return {"probabilities": probabilities, "tail_from": total + 1, "tail": 0.0}
