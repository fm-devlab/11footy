"""The UEFA cups' qualifiers (T4.qualifies.team): the rounds' names, the draws known at an
instant, the tree of winners and losers and its simulation."""

import numpy as np

from vesta.season.cup_qualifiers import (
    QualifyingSeason,
    QualifyingTie,
    find_checkpoints,
    find_known_ties,
    find_qualifying_round,
    find_subject_clubs,
    link_ties,
    simulate_qualifiers,
)
from vesta.spine.score_matrix import Calibration, score_matrices
from vesta.spine.ties import TieParameters, TieRule

PARAMETERS = TieParameters(extra_time_minutes=30.0, extra_time_pace=1.0, shootout_home_share=0.5)
RULE = TieRule(False, False)


def test_the_rounds_are_named_in_both_eras():
    assert find_qualifying_round("CL", "qualifying third round") == "Q3"
    assert find_qualifying_round("EL", "third qualifying round") == "Q3"
    assert find_qualifying_round("ECLQ", "second round") == "Q2"
    assert find_qualifying_round("CLQ", "playoff round") == "PO"
    assert find_qualifying_round("CL", "playoffs") == "PO"
    assert find_qualifying_round("CL", "round of 16") is None
    assert find_qualifying_round("EL", "second round") is None  # the old Europa League's knockout


def _tie(key, cup, name, day, home, away, winner=None, played=False):
    matches = ((f"2022-{day}T19:00:00Z", home, away, 1 if played else None,
                0 if played else None, played),
               (f"2022-{day[:3]}{int(day[3:]) + 7:02d}T19:00:00Z", away, home,
                0 if played else None, 0 if played else None, played))  # fmt: skip
    return QualifyingTie(key, cup, name, matches[0][0], home, away, 2, matches, winner)


def _season():
    # CL Q3: 1 v 2 (1 wins); EL PO: 1's... the loser of the CL Q3 (2) meets 5, the CL PO: 1 v 3
    ties = (
        _tie("a", "CL", "Q3", "08-02", 1, 2, winner=1),
        _tie("b", "EL", "Q3", "08-04", 4, 6, winner=4),
        _tie("c", "CL", "PO", "08-16", 1, 3, winner=1),
        _tie("d", "EL", "PO", "08-18", 2, 4, winner=4),
    )
    return QualifyingSeason(2022, ties, {"CL": frozenset({1}), "EL": frozenset({4})})


DATES = {("CL", 2022, "Q3"): "2022-07-18", ("CL", 2022, "PO"): "2022-08-01",
         ("EL", 2022, "Q3"): "2022-07-18", ("EL", 2022, "PO"): "2022-08-01"}  # fmt: skip


def test_a_checkpoint_needs_its_path_drawn_before_it():
    season = _season()
    assert find_checkpoints(season, "CL", DATES) == {"Q3": "2022-08-02T00:00:00Z",
                                                     "PO": "2022-08-16T00:00:00Z"}  # fmt: skip
    late = DATES | {("CL", 2022, "PO"): "2022-08-02"}  # drawn on the day Q3 begins
    assert find_checkpoints(season, "CL", late) == {"PO": "2022-08-16T00:00:00Z"}
    known = {tie.tie_key for tie in find_known_ties(season, "2022-08-02T00:00:00Z", late)}
    assert known == {"a", "b", "d"}  # the CL play-off, drawn that day, is not known yet


def test_the_tree_links_winners_and_losers_dropping_down():
    linked = link_ties(list(_season().ties))
    by_key = {tie.tie_key: (first, second) for tie, first, second in linked}
    first, second = by_key["c"]
    assert first.tie == 0 and first.won and second.club == 3  # the CL Q3's winner v 3
    first, second = by_key["d"]
    assert first.tie == 0 and not first.won  # the CL Q3's loser drops into the EL play-off
    assert second.tie == 1 and second.won
    assert find_subject_clubs(linked, "EL", "Q3") == {4, 6}
    assert find_subject_clubs(linked, "CL", "Q3") == {1, 2, 3}  # 3 enters the play-off directly


def test_the_simulation_carries_a_club_down_its_path():
    ratings = {1: 0.5, 2: 0.0, 3: 0.0, 4: 0.0, 6: 0.0}

    def strength(home, away, neutral):
        home_rate = float(np.exp(0.25 + ratings[home] - ratings[away]))
        away_rate = float(np.exp(ratings[away] - ratings[home]))
        return (score_matrices(np.array([home_rate]), np.array([away_rate]), Calibration())[0],
                home_rate, away_rate)  # fmt: skip

    linked = link_ties(list(_season().ties))
    through = simulate_qualifiers(linked, "2022-08-02T00:00:00Z", strength,
                                  {"CL": RULE, "EL": RULE}, PARAMETERS, 2000, 3)  # fmt: skip
    none = np.zeros(2000, dtype=bool)
    cl = [through.get((club, "CL"), none) for club in (1, 2, 3)]
    assert (np.sum(cl, axis=0) == 1).all()  # one club wins the CL play-off in every draw
    el = [through.get((club, "EL"), none) for club in (1, 2, 4, 6)]
    assert (np.sum(el, axis=0) == 1).all()
    assert cl[0].mean() > cl[2].mean()  # the strongest goes through most
    assert through[(2, "EL")].mean() > 0  # into the EL after losing the CL's third round
    assert not (through[(2, "EL")] & cl[1]).any()  # never both: a loser of the CL's Q3
