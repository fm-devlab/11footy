"""The nations' tournaments verdict (T6): the fixed brackets of the regulations, the reference's
history by Elo thirds and its pool, and the checkpoints."""

from collections import Counter

import numpy as np
import pytest

from vesta.season.brackets import build
from vesta.season.knockout import list_leaves
from vesta.season.nations_verdict import (
    THIN,
    History,
    find_edition_instants,
    place_shares,
    pool_reference_shares,
    split_thirds_by_rating,
)
from vesta.season.tournament import read_tournament_format


def _first_round(tree, name):
    if isinstance(tree, int):
        return []
    own = [(tree.first, tree.second)] if tree.round_name == name else []
    return own + _first_round(tree.first, name) + _first_round(tree.second, name)


def _places(groups: str) -> dict[str, list[int]]:
    return {group: [10 * index + place for place in (1, 2, 3, 4)]
            for index, group in enumerate(groups)}  # fmt: skip


def test_the_euro_bracket_follows_its_table_of_thirds():
    places = _places("ABCDEF")  # 1A is 1, 2A 2, 3A 3; 1B 11 and so on
    tree = build(
        "fixed",
        places,
        read_tournament_format("EURO", 2024).rounds,
        np.random.default_rng(1),
        ("A", "B", "C", "D"),
        fixed=read_tournament_format("EURO", 2024).fixed,
    )
    pairs = set(_first_round(tree, "round_of_16"))
    # art. 21.05: with the thirds of A, B, C and D, WB v 3A, WC v 3D, WE v 3B, WF v 3C
    assert {(11, 3), (21, 33), (41, 13), (51, 23)} <= pairs
    assert (1, 22) in pairs and (32, 42) in pairs  # WA v RC, RD v RE
    assert len(list_leaves(tree)) == 16


def test_the_afcon_and_the_copa_brackets():
    form = read_tournament_format("AFCON", 2023)
    tree = build("fixed", _places("ABCDEF"), form.rounds, np.random.default_rng(1),
                 ("A", "B", "E", "F"), fixed=form.fixed)  # fmt: skip
    pairs = set(_first_round(tree, "round_of_16"))
    # art. 75.6, ABEF: 1A v 3E, 1B v 3A, 1C v 3B, 1D v 3F
    assert {(1, 43), (11, 3), (21, 13), (31, 53)} <= pairs
    copa = read_tournament_format("COPA", 2019)
    tree = build("fixed", _places("ABC"), copa.rounds, np.random.default_rng(1), ("B", "C"),
                 fixed=copa.fixed)  # fmt: skip
    pairs = set(_first_round(tree, "quarter_finals"))
    assert (1, 23) in pairs and (21, 13) in pairs  # 1A v 3C, 1C v 3B: the only way both meet one


def test_the_thirds_by_elo():
    rating = {team: float(team) for team in range(1, 10)}
    thirds = split_thirds_by_rating(list(range(1, 10)), rating, 0.0)
    assert [thirds[team] for team in (9, 8, 7)] == [0, 0, 0]
    assert [thirds[team] for team in (3, 2, 1)] == [2, 2, 2]


def test_the_reference_reads_its_cell_pools_when_thin_and_keeps_what_can_happen():
    many = Counter({0: 10, 1: 10, 4: 20})  # 40 earlier teams: winners, finalists, out in groups
    few = Counter({0: 5})
    past = History({("EURO", "G0", 0): many, ("COPA", "G0", 0): few, ("COPA", "G2", 0): few})
    pooled: list = []
    shares = pool_reference_shares(past, "EURO", "G0", 0, 6, 0, pooled)  # EURO: six outcomes
    assert shares[5] == pytest.approx(0.25) and shares[4] == pytest.approx(0.25)  # winner, final
    assert shares[1] == pytest.approx(0.5) and pooled == []  # distance 4 is the round of 16
    shares = pool_reference_shares(past, "COPA", "G0", 0, 5, 0, pooled)  # thin: the pool of six
    assert pooled == [("COPA", "G0", 0, 5, 45)]
    assert sum(shares) == pytest.approx(1.0)
    # a team through the groups cannot end in them: the group stage dropped, the rest renormalised
    shares = pool_reference_shares(past, "EURO", "G0", 0, 6, 1, [])
    assert shares[0] == 0.0 and sum(shares) == pytest.approx(1.0)
    # nothing in the history: uniform over what can still happen
    assert pool_reference_shares(History({}), "GOLD", "K1", 2, 5, 2, []) == [
        0,
        0,
        1 / 3,
        1 / 3,
        1 / 3,
    ]
    assert THIN == 30


def test_the_place_reference_by_elo_rank_in_the_group():
    past = History({("EURO", "G0", 0): Counter({0: 30, 1: 10})})
    shares = place_shares(past, "EURO", "G0", 0, 4, [])
    assert shares[0] > shares[1] > shares[2] == shares[3] > 0


def test_the_checkpoints_of_an_edition():
    group = [("2024-06-14T19:00:00Z", 1, 2), ("2024-06-15T13:00:00Z", 3, 4),
             ("2024-06-19T16:00:00Z", 1, 3), ("2024-06-26T19:00:00Z", 2, 4)]  # fmt: skip
    instants = find_edition_instants(group, ["2024-06-29T16:00:00Z", "2024-07-02T19:00:00Z"])
    assert instants == {"G0": "2024-06-14T00:00:00Z", "G1": "2024-06-16T00:00:00Z",
                        "G2": "2024-06-27T00:00:00Z", "K1": "2024-07-03T00:00:00Z"}  # fmt: skip


def test_a_format_without_a_round_of_16_skips_the_line_that_names_it():
    from vesta.quantities import load_quantities
    from vesta.season.cups_verdict import Row, _collect_line_pairs

    shares = [0.5, 0.2, 0.15, 0.1, 0.05]  # the Gold Cup 2023: group, quarter-finals, ..., winner
    row = Row("T6.round_reached", "GOLD", 2023, "G0", 1, 2, {"vesta": shares})
    pairs = _collect_line_pairs(row, "vesta", load_quantities())
    assert len(pairs) == 4  # the round-of-16 line repeats the quarter-finals': not counted twice
    assert pairs[-1] == (pytest.approx(0.5), True)  # any=quarter_finals,...: reached, said 0.5
