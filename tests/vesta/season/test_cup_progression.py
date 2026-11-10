"""The UEFA cups' progression (T4): the bracket rules with drop-ins, the joint simulation's
coherence, the knockout as known at an instant, the truth and the verdict's scores."""

import numpy as np
import pytest

from vesta.season.brackets import build
from vesta.season.cup_progression import (
    CupMatch,
    CupSeason,
    count_ties_in_round,
    find_known_knockout,
    find_outcomes_reached,
    simulate_cup,
)
from vesta.season.cups_verdict import compute_entropy, score_log_loss, score_ranked_probability
from vesta.season.knockout import list_leaves
from vesta.season.standings import read_tie_break_rule
from vesta.season.tournament import read_tournament_format
from vesta.spine.score_matrix import Calibration, score_matrices
from vesta.spine.ties import TieParameters, TieRule, resolve_second_leg

PARAMETERS = TieParameters(extra_time_minutes=30.0, extra_time_pace=1.0, shootout_home_share=0.5)
NO_AWAY_GOALS = TieRule(False, False)


def _strength(ratings: dict[int, float]):
    def strength(home: int, away: int, neutral: bool):
        advantage = 0.0 if neutral else 0.25
        home_rate = float(np.exp(0.2 + advantage + ratings[home] - ratings[away]))
        away_rate = float(np.exp(0.2 + ratings[away] - ratings[home]))
        matrix = score_matrices(np.array([home_rate]), np.array([away_rate]), Calibration())[0]
        return matrix, home_rate, away_rate

    return strength


def _ties(node, name):
    if isinstance(node, int):
        return []
    own = [node] if node.round_name == name else []
    return own + _ties(node.first, name) + _ties(node.second, name)


def _groups(count: int, base: int) -> dict[str, list[int]]:
    return {chr(65 + index): [base + 10 * index + place for place in (1, 2, 3, 4)]
            for index in range(count)}  # fmt: skip


def test_the_round_of_32_seeds_the_winners_and_the_best_thirds():
    places = _groups(12, 0)
    drop_ins = (901, 902, 903, 904, 905, 906, 907, 908)  # the four best first
    tree = build("uefa_round_of_32", places, read_tournament_format("EL", 2020).rounds,
                 np.random.default_rng(3), drop_ins=drop_ins)  # fmt: skip
    assert sorted(list_leaves(tree)) == sorted(
        [club for group in places.values() for club in group[:2]] + list(drop_ins)
    )
    for tie in _ties(tree, "round_of_32"):  # the seeded club at home in the second leg
        assert tie.second % 10 == 1 or tie.second in drop_ins[:4]
        assert tie.first % 10 == 2 or tie.first in drop_ins[4:]
        if tie.second < 900 and tie.first < 900:
            assert tie.second // 10 != tie.first // 10  # never the runner-up of its own group


def test_the_knockout_play_offs_meet_the_higher_cups_thirds():
    places = _groups(8, 0)
    drop_ins = tuple(range(901, 909))
    tree = build("uefa_knockout_play_offs", places, read_tournament_format("ECL", 2022).rounds,
                 np.random.default_rng(4), drop_ins=drop_ins)  # fmt: skip
    for tie in _ties(tree, "knockout_play_offs"):
        assert tie.first in drop_ins and tie.second % 10 == 2  # the runner-up at home second
    for tie in _ties(tree, "round_of_16"):
        assert tie.first.round_name == "knockout_play_offs" and tie.second % 10 == 1
    assert count_ties_in_round(_cup("ECL", 2022, places), "knockout_play_offs") == 8
    assert count_ties_in_round(_cup("ECL", 2022, places), "round_of_16") == 8
    assert count_ties_in_round(_cup("ECL", 2022, places), "quarter_finals") == 4


def _cup(code: str, season: int, places: dict[str, list[int]], played: bool = False,
         knockout: tuple[CupMatch, ...] = ()) -> CupSeason:  # fmt: skip
    """A group stage whose clubs end in the order given (played: each club beats those below)."""
    phase = []
    match_id = 0
    for group, clubs in places.items():
        for home in clubs:
            for away in clubs:
                if home == away:
                    continue
                match_id += 1
                better = clubs.index(home) < clubs.index(away)
                phase.append(CupMatch(match_id, "phase", group, f"2021-09-{1 + match_id % 28:02d}",
                                      home, away, False, played, 1 if better else 0,
                                      0 if better else 1, None, None))  # fmt: skip
    form = read_tournament_format(code, season)
    return CupSeason(code, season, form, read_tie_break_rule(code, season), NO_AWAY_GOALS,
                     tuple(phase), knockout)  # fmt: skip


def test_the_joint_simulation_adds_up_and_takes_the_higher_cups_thirds():
    upper = _groups(8, 0)
    lower = _groups(8, 500)
    ratings = {club: 0.3 - 0.1 * (club % 10) for group in (*upper.values(), *lower.values())
               for club in group}  # fmt: skip
    source = _cup("CL", 2022, upper)
    cup = _cup("EL", 2022, lower)
    draws = simulate_cup(cup, source, "2021-08-01", _strength(ratings), PARAMETERS, 300, 5)
    outcomes = draws.outcomes
    assert outcomes == ("group_stage", "knockout_play_offs", "round_of_16", "quarter_finals",
                        "semi_finals", "final", "winner")  # fmt: skip
    # the higher cup's clubs are not this cup's before its phase is over: only the 32 of the phase
    assert set(draws.reached) == {club for group in lower.values() for club in group}
    for place in range(1, 5):  # one club in each place of each group in every simulation
        assert all(sum((draws.places[club] == place).sum() for club in group) == 300
                   for group in lower.values())  # fmt: skip
    # with the higher cup's phase over its thirds are this cup's: every round adds up
    played = _cup("CL", 2022, upper, played=True)
    draws = simulate_cup(cup, played, "2021-12-31", _strength(ratings), PARAMETERS, 300, 5)
    thirds = {group[2] for group in upper.values()}
    assert thirds <= set(draws.reached)
    reached = np.array(list(draws.reached.values()))
    for index, clubs in ((1, 24), (2, 16), (3, 8), (4, 4), (5, 2), (6, 1)):
        assert ((reached >= index).sum(axis=0) == clubs).all()  # in every simulation
    assert all((draws.reached[club] >= 1).all() for club in thirds)  # a third never goes out early


def test_a_tie_under_way_agrees_with_the_ties_step():
    strength = _strength({1: 0.3, 2: -0.1})
    # the first leg at club 2's ground ended 1-0 to 2; club 1 at home in the second leg
    leg = CupMatch(1000, "round_of_16", "", "2022-02-15", 2, 1, False, True, 1, 0, None, 77)
    from vesta.season.cup_progression import _make_world_tie

    tie = _make_world_tie(_cup("CL", 2022, {}), "round_of_16", 2, [leg], 2, 1, "2022-03-01")
    assert (tie.first, tie.second, tie.first_leg, tie.winner) == (2, 1, (1, 0), None)
    matrix, home, away = strength(1, 2, False)
    analytic = sum(resolve_second_leg(matrix, home, away, (1, 0), NO_AWAY_GOALS,
                                      PARAMETERS).through["home"].values())  # fmt: skip
    from vesta.season.knockout import TieSampler

    sampler = TieSampler(strength, NO_AWAY_GOALS, PARAMETERS, np.random.default_rng(8))
    draws = 4000
    simulated = np.mean([sampler.draw_winner(tie, 2, 1) == 1 for _ in range(draws)])
    assert abs(simulated - analytic) < 3 * np.sqrt(analytic * (1 - analytic) / draws)


def test_ties_are_settled_by_their_goals_and_a_tie_awarded_is_found():
    places = {"A": [1, 3, 5, 7], "B": [2, 4, 6, 8], "C": [9, 11, 13, 15], "D": [10, 12, 14, 16],
              "E": [17, 19, 21, 23], "F": [18, 20, 22, 24], "G": [25, 27, 29, 31],
              "H": [26, 28, 30, 32]}  # fmt: skip
    # the world's tie winner (77 -> 2) is wrong: 1 won 4-0 and lost 2-3, so 6-3 on aggregate
    first = CupMatch(1, "round_of_16", "", "2022-02-15", 1, 2, False, True, 4, 0, 1, 77)
    second = CupMatch(2, "round_of_16", "", "2022-03-08", 2, 1, False, True, 3, 2, 2, 77)
    quarter = CupMatch(3, "quarter_finals", "", "2022-04-05", 1, 9, False, False, None, None,
                       None, 78)  # fmt: skip
    cup = _cup("CL", 2022, places, played=True, knockout=(first, second, quarter))
    cup = CupSeason(cup.code, cup.season, cup.form, cup.table_rule, cup.tie_rule, cup.phase,
                    cup.knockout, {77: 2})  # fmt: skip
    order = {group: clubs for group, clubs in places.items()}
    happened = find_outcomes_reached(cup, order)
    outcomes = cup.form.outcomes
    assert happened[1] == outcomes.index("quarter_finals")
    assert happened[2] == outcomes.index("round_of_16")
    assert happened[3] == outcomes.index("round_of_16")  # a runner-up: sent there by its place
    assert happened[5] == 0
    with pytest.raises(ValueError, match="ties missing"):  # 1 tie of 8: not an award, a gap
        find_known_knockout(cup, "2022-03-10", order)


def test_the_scores():
    assert score_log_loss([0.5, 0.5], 0) == pytest.approx(np.log(2))
    assert score_log_loss([1.0, 0.0], 1) == pytest.approx(-np.log(1e-4))
    assert compute_entropy([0.5, 0.5]) == pytest.approx(np.log(2))
    assert score_ranked_probability([0.0, 1.0, 0.0], 1) == 0.0
    assert score_ranked_probability([1.0, 0.0, 0.0], 2) == pytest.approx(1.0)


def test_candidate_2_tempers_walk_forward():
    from vesta.season.cups_verdict import Row, apply_temperature, fit_tau, temper_shares

    assert temper_shares([0.5, 0.5], 2.0) == pytest.approx([0.5, 0.5])
    assert (
        temper_shares([0.8, 0.2, 0.0], 2.0)[0] > 0.8
        and temper_shares([0.8, 0.2, 0.0], 2.0)[2] == 0.0
    )
    assert temper_shares([0.8, 0.2], 0.5)[0] < 0.8
    # forecasts too flat: the favourite (0.6) always wins -> a sharpening tau
    flat = [Row("T4.round_reached", "CL", 2018, "C0", team, 0, {"vesta": [0.6, 0.4]})
            for team in range(50)]
    assert fit_tau(flat) > 1.0
    later = [Row("T4.round_reached", "CL", 2019, "C0", 1, 1, {"vesta": [0.6, 0.4]}),
             Row("T4.round_reached", "CL", 2019, "C3", 2, 1, {"vesta": [0.6, 0.4]})]
    rows, taus = apply_temperature(later, flat)
    assert taus[(2019, "T4.round_reached", "phase")] > 1.0
    assert taus[(2019, "T4.round_reached", "knockout")] == 1.0  # no earlier knockout rows
    assert rows[0].distributions["candidate_1"] == [0.6, 0.4]
    assert rows[0].distributions["vesta"][0] > 0.6
    # the season judged never fits its own tau: 2018's rows alone, not 2019's
    _, own = apply_temperature(flat, [])
    assert own[(2018, "T4.round_reached", "phase")] == 1.0
