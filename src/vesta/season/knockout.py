"""Knockout rounds simulated to the final (T4, T6): the bracket as a tree, every tie drawn.

A knockout is a tree: a leaf is a team, a node a tie of a round between the winners of its two
branches. The bracket rules of a competition and season (`vesta.season.brackets`) build the tree
from the places of the phase before it; a round whose pairs are drawn afresh is the same, in law,
as a tree whose branches are arranged at random, so every format is a tree. A tie already played
carries its winner, a tie half played its first leg.

Each tie follows the model of the ties step (`vesta.spine.ties`), drawn instead of summed: the
legs' scores from their matrices, the aggregate, the away goals where the competition's rules of
that season (`config/vesta/tie-rules.csv`) say so, extra time as two Poissons of the last match's
intensities over `extra_time_minutes` at `extra_time_pace`, then a shoot-out the home side of the
last match wins with `shootout_home_share` (at even chances on neutral ground). The matrices of
pairs nobody has forecast yet come from the strength of the competition's model, frozen today.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from vesta.spine.score_matrix import SIZE
from vesta.spine.ties import REGULAR_MINUTES, TieParameters, TieRule

# (home team, away team, neutral ground) -> (matrix, home intensity, away intensity)
Strength = Callable[[int, int, bool], tuple[np.ndarray, float, float]]


@dataclass(frozen=True)
class Tie:
    """A tie of `round_name` between the winners of `first` and `second`; with two legs, `first`
    is at home in the first leg. `winner` once decided, `first_leg` (goals of `first`, of
    `second`) once the first leg is played."""

    round_name: str
    legs: int
    first: "Tie | int"
    second: "Tie | int"
    neutral: bool = False
    winner: int | None = None
    first_leg: tuple[int, int] | None = None




class TieSampler:
    """Draws ties, with the matrices of each pair kept once computed."""

    def __init__(self, strength: Strength, rule: TieRule, parameters: TieParameters,
                 generator: np.random.Generator) -> None:  # fmt: skip
        self.strength = strength
        self.rule = rule
        self.parameters = parameters
        self.generator = generator
        self._matches: dict[tuple[int, int, bool], tuple[np.ndarray, float, float]] = {}

    def draw_winner(self, tie: Tie, first: int, second: int) -> int:
        if tie.winner is not None:
            return tie.winner
        if tie.legs == 1:
            home_goals, away_goals = self._score(first, second, tie.neutral)
            return self._settle(first, second, home_goals - away_goals, tie.neutral, False)
        if tie.first_leg is not None:
            first_goals, second_away = tie.first_leg
        else:
            first_goals, second_away = self._score(first, second, False)
        second_goals, first_away = self._score(second, first, False)
        aggregate = (second_away + second_goals) - (first_goals + first_away)  # for `second`
        away_margin = second_away - first_away  # away goals, for `second`
        if aggregate == 0 and self.rule.away_goals and away_margin != 0:
            return second if away_margin > 0 else first
        return self._settle(second, first, aggregate, False, True)

    def _settle(self, home, away, margin, neutral, second_leg) -> int:
        """`margin` for `home` after 90 minutes (the aggregate, for a second leg): extra time,
        then penalties."""
        if margin != 0:
            return home if margin > 0 else away
        _, home_intensity, away_intensity = self._prepare_match(home, away, neutral)
        share = (self.parameters.extra_time_minutes / REGULAR_MINUTES
                 * self.parameters.extra_time_pace)  # fmt: skip
        extra_home = self.generator.poisson(home_intensity * share)
        extra_away = self.generator.poisson(away_intensity * share)
        if extra_home != extra_away:
            return home if extra_home > extra_away else away
        away_goals_count = self.rule.away_goals and self.rule.away_goals_in_extra_time
        if second_leg and away_goals_count and extra_away > 0:
            return away  # level in extra time of a second leg: the visitors scored away
        home_share = 0.5 if neutral else self.parameters.shootout_home_share
        return home if self.generator.random() < home_share else away

    def _score(self, home: int, away: int, neutral: bool) -> tuple[int, int]:
        cumulative = self._look_up_cumulative(home, away, neutral)
        cell = min(int(np.searchsorted(cumulative, self.generator.random(), side="right")),
                   SIZE * SIZE - 1)  # fmt: skip
        return cell // SIZE, cell % SIZE

    def _prepare_match(self, home, away, neutral):
        key = (home, away, neutral)
        if key not in self._matches:
            matrix, home_intensity, away_intensity = self.strength(home, away, neutral)
            matrix = np.asarray(matrix, dtype=float)
            self._matches[key] = (np.cumsum(matrix.ravel() / matrix.sum()), home_intensity,
                                  away_intensity)  # fmt: skip
        return self._matches[key]

    def _look_up_cumulative(self, home, away, neutral) -> np.ndarray:
        return self._prepare_match(home, away, neutral)[0]


def play(tree: Tie | int, sampler: TieSampler, furthest: dict[int, int],
         round_index: dict[str, int]) -> int:  # fmt: skip
    """The winner of `tree`; `furthest` gets the round each team reaches (the winner of the
    final one more than the final)."""
    if isinstance(tree, int):
        return tree
    first = play(tree.first, sampler, furthest, round_index)
    second = play(tree.second, sampler, furthest, round_index)
    here = round_index[tree.round_name]
    for team in (first, second):
        furthest[team] = max(furthest.get(team, -1), here)
    winner = sampler.draw_winner(tree, first, second)
    if tree.round_name == "final":
        furthest[winner] = here + 1
    return winner


def list_leaves(tree: Tie | int) -> list[int]:
    if isinstance(tree, int):
        return [tree]
    return list_leaves(tree.first) + list_leaves(tree.second)
