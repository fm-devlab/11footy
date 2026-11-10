"""Streaks, the next k matches and windows (T3, T12, T13): the counts agree, the firsts agree with
the counts, and a run under way at the checkpoint is carried, never broken or revived."""

import numpy as np

from vesta.season.windows import (
    award_points,
    count_season_streaks,
    count_shares,
    count_window,
    find_first,
    find_longest_run,
    run_at_end,
)


def test_the_counts_of_a_window_agree():
    generator = np.random.default_rng(1)
    goals_for = generator.poisson(1.4, (500, 5))
    goals_against = generator.poisson(1.1, (500, 5))
    counts = count_window(goals_for, goals_against)
    draws = ((goals_for == goals_against).sum(axis=1))
    assert (counts["points"] == 3 * counts["wins"] + draws).all()
    assert (counts["wins"] + draws + counts["losses"] == 5).all()
    assert (counts["goals"] == goals_for.sum(axis=1)).all()




def test_a_run_under_way_is_carried_and_an_ended_run_is_not_revived():
    drawn = np.array([[True, True, False], [False, True, True], [True, False, False]])
    # the played matches end on a run of three: the first draw extends it to five
    on_a_run = np.array([False, True, True, True])
    assert run_at_end(on_a_run) == 3
    assert find_longest_run(on_a_run, drawn).tolist() == [5, 3, 4]
    # the run of three ended with the last match played: it is not revived by the draws
    ended = np.array([True, True, True, False])
    assert run_at_end(ended) == 0
    assert find_longest_run(ended, drawn).tolist() == [3, 3, 3]
    # a draw's run never falls below the run the team is on, nor below the longest one played
    generator = np.random.default_rng(3)
    many = generator.random((400, 10)) < 0.5
    best = find_longest_run(on_a_run, many)
    assert (best >= 3).all()
    assert (best >= run_at_end(on_a_run) + find_first(~many)).all()


def test_the_season_streaks():
    played_for = np.array([2, 0, 1])
    played_against = np.array([0, 1, 0])
    played_home = np.array([True, True, False])  # a home defeat already
    drawn_for = np.array([[1, 1], [0, 0]])
    drawn_against = np.array([[0, 0], [2, 2]])
    drawn_home = np.array([[True, False], [True, False]])
    streaks = count_season_streaks(
        played_for, played_against, played_home, drawn_for, drawn_against, drawn_home
    )
    assert streaks["longest_winning_streak"].tolist() == [3, 1]
    assert not streaks["unbeaten_at_home"].any()
    assert streaks["loses_three_in_a_row"].tolist() == [False, False]
    assert (streaks["longest_unbeaten_run"] >= streaks["longest_winning_streak"]).all()


def test_the_first_loss_agrees_with_the_losses():
    generator = np.random.default_rng(2)
    goals_for = generator.poisson(1.2, (800, 5))
    goals_against = generator.poisson(1.2, (800, 5))
    lost = award_points(goals_for, goals_against) == 0
    first = find_first(lost)
    assert ((first == 5) == (count_window(goals_for, goals_against)["losses"] == 0)).all()
    assert ((first == 0) == lost[:, 0]).all()  # the next match lost
    assert sum(count_shares(first, 6)) == 1.0
