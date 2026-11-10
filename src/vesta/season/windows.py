"""Streaks, the next k matches and windows (T3, T12, T13), read from drawn scores.

Every match a team still has to play is drawn from its matrix (or, for the reference, from the
team's own form); these functions read a team's draws. A team's matches are seen from its side:
its goals for and against, (draws, matches) in kick-off order, the matches already played as one
row broadcast to every draw.

- the next k matches: points, wins, losses, goals and clean sheets;
- the first of the next matches the team loses, or keeps a clean sheet in (an ordered outcome: the
  first, the second, ... or none of them);
- the season's streaks: the run a team is on at the checkpoint carries into the drawn matches (a
  winning run of three that the next match extends is four), and a run that has ended is not
  revived.
"""

import numpy as np

OUTCOME_NONE = "none"


def award_points(goals_for: np.ndarray, goals_against: np.ndarray) -> np.ndarray:
    """3 for a win, 1 for a draw, 0 for a defeat."""
    return np.where(goals_for > goals_against, 3, np.where(goals_for == goals_against, 1, 0))


def count_window(goals_for: np.ndarray, goals_against: np.ndarray) -> dict[str, np.ndarray]:
    """(draws,) the points, wins, losses, goals and clean sheets of the matches given."""
    points = award_points(goals_for, goals_against)
    return {
        "points": points.sum(axis=1),
        "wins": (points == 3).sum(axis=1),
        "losses": (points == 0).sum(axis=1),
        "goals": goals_for.sum(axis=1),
        "clean_sheets": (goals_against == 0).sum(axis=1),
    }


def find_first(happens: np.ndarray) -> np.ndarray:
    """(draws,) the index of the first match where `happens`, or the number of matches when it
    never does (the outcome `none`)."""
    count = happens.shape[1]
    first = np.argmax(happens, axis=1)
    return np.where(happens.any(axis=1), first, count)




def find_longest_run(played: np.ndarray, drawn: np.ndarray) -> np.ndarray:
    """(draws,) the longest run of True over the season: the matches played (n0,) then the drawn
    ones (draws, n1); a run under way at the checkpoint goes on into the drawn matches."""
    draws = drawn.shape[0]
    current = np.zeros(draws, dtype=np.int64)
    best = np.zeros(draws, dtype=np.int64)
    for value in played:
        current = (current + 1) * bool(value)
        best = np.maximum(best, current)
    for column in range(drawn.shape[1]):
        current = (current + 1) * drawn[:, column]
        best = np.maximum(best, current)
    return best


def run_at_end(played: np.ndarray) -> int:
    """The length of the run of True the played matches end on."""
    length = 0
    for value in played[::-1]:
        if not value:
            break
        length += 1
    return length


def count_season_streaks(
    played_for: np.ndarray,
    played_against: np.ndarray,
    played_home: np.ndarray,
    drawn_for: np.ndarray,
    drawn_against: np.ndarray,
    drawn_home: np.ndarray,
) -> dict[str, np.ndarray]:
    """(draws,) the season's T3 quantities from the matches played and the drawn ones."""
    played = award_points(played_for, played_against)
    drawn = award_points(drawn_for, drawn_against)
    home_losses = ((played == 0) & played_home).any()
    return {
        "longest_winning_streak": find_longest_run(played == 3, drawn == 3),
        "longest_unbeaten_run": find_longest_run(played > 0, drawn > 0),
        "unbeaten_at_home": ~(home_losses | ((drawn == 0) & drawn_home).any(axis=1)),
        "loses_three_in_a_row": find_longest_run(played == 0, drawn == 0) >= 3,
    }


def count_shares(values: np.ndarray, size: int) -> list[float]:
    """The shares of 0 .. size-1 in `values` (draws,), the larger values lumped in the last."""
    clipped = np.minimum(values, size - 1)
    return (np.bincount(clipped, minlength=size)[:size] / len(values)).tolist()
