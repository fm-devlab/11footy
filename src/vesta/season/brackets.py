"""The bracket rules: from the places of the phase before the knockout to its tree.

Each rule is named in `config/vesta/tournament-formats.toml` for the competitions and seasons it
holds; a format without its rule written here is refused, never approximated.
- `uefa_league_phase_2024`: the UEFA cups from 2024-25, by UEFA's bracket: knockout
  play-offs 9/10 v 23/24, 11/12 v 21/22, 13/14 v 19/20, 15/16 v 17/18 (which of each pair meets
  which, drawn); their winners against 7/8, 5/6, 3/4 and 1/2 in the round of 16, the two clubs
  of each seeded pair in opposite halves; then the quarter-finals 1/2 v 7/8 and 3/4 v 5/6 within
  each half, and the semi-finals by the bracket. The better placed club plays the second leg at
  home in the play-offs and the round of 16; later, which club is at home first is drawn.
- `group_winners_v_runners_up`: the UEFA cups' group stages: each runner-up drawn against the
  winner of another group, the winner at home in the second leg; the later rounds drawn afresh,
  with no seeding. The protection of clubs of the same country in the round of 16 is not held.
- `world_cup_32`: the FIFA World Cup of 32 teams (2014 to 2022), single matches on neutral ground:
  1A v 2B, 1C v 2D, 1E v 2F, 1G v 2H in one half, 1B v 2A, 1D v 2C, 1F v 2E, 1H v 2G in the other,
  the quarter-finals the neighbouring pairs.
- `afc_asian_cup_2027`: the AFC Asian Cup 2027 (Competition Regulations art. 9.8-9.11), single
  matches at the central venue (held as neutral ground): the round of 16 2A v 2C, 1D v 3B/E/F,
  1B v 3A/C/D, 1F v 2E, 1C v 3A/B/F, 1E v 2D, 1A v 3C/D/E, 2B v 2F, the four best thirds placed by
  the regulation's table of the 15 combinations of their groups (art. 9.9); the quarter-finals
  the neighbouring pairs, the semi-finals the neighbouring quarter-finals.
"""

from dataclasses import replace

import numpy as np

from vesta.season.knockout import Tie


def build(rule: str, places: dict[str, list[int]], rounds: tuple[tuple[str, int], ...],
          generator: np.random.Generator, thirds: tuple[str, ...] = (),
          drop_ins: tuple[int, ...] = (), neutral_rounds: tuple[str, ...] = (),
          fixed: dict | None = None) -> Tie:  # fmt: skip
    """The tree of `rule` from `places` (per group, or "" for a single table: the teams in their
    final order) and the rounds (name, legs) of the format, the last the final; `thirds` the
    groups whose third-placed team goes on, where the format takes the best thirds; `drop_ins`
    the clubs a higher cup sends into the knockout, its seeded ones first; the ties of
    `neutral_rounds` on neutral ground; `fixed` the bracket a `fixed` rule reads from the format."""
    if rule == "fixed":
        if fixed is None:
            raise ValueError("a fixed bracket needs its first round and later pairings")
        return build_fixed_bracket(fixed, places, rounds, thirds)
    if rule not in RULES:
        raise ValueError(f"no bracket rule {rule!r} is written")
    if rule in WITH_THIRDS:
        tree = RULES[rule](places, rounds, generator, thirds)
    elif rule in WITH_DROP_INS:
        tree = RULES[rule](places, rounds, generator, drop_ins)
    else:
        tree = RULES[rule](places, rounds, generator)
    return mark_neutral_rounds(tree, neutral_rounds) if neutral_rounds else tree


def mark_neutral_rounds(tree: Tie | int, rounds: tuple[str, ...]) -> Tie | int:
    """`tree` with the ties of `rounds` on neutral ground."""
    if isinstance(tree, int):
        return tree
    return replace(tree, first=mark_neutral_rounds(tree.first, rounds),
                   second=mark_neutral_rounds(tree.second, rounds),
                   neutral=tree.neutral or tree.round_name in rounds)  # fmt: skip


# the knockout play-offs of the league phase: the seeded pair of places, the unseeded pair
SEEDED_PAIRS_2024 = (((9, 10), (23, 24)), ((11, 12), (21, 22)), ((13, 14), (19, 20)),
                     ((15, 16), (17, 18)))  # fmt: skip


def _build_uefa_league_phase_2024(places, rounds, generator) -> Tie:
    order = places[""]
    names = [name for name, _ in rounds]
    legs = dict(rounds)

    def shuffle_pair(first_place: int, second_place: int) -> list[int]:
        pair = [order[first_place - 1], order[second_place - 1]]
        return [pair[index] for index in generator.permutation(2)]

    play_offs = {}
    for seeds, unseeded in SEEDED_PAIRS_2024:
        better, worse = shuffle_pair(*seeds), shuffle_pair(*unseeded)
        play_offs[seeds] = [Tie(names[0], legs[names[0]], worse[index], better[index])
                            for index in range(2)]  # fmt: skip
    return build_league_phase_from_play_offs(order, rounds, generator, play_offs)


def build_league_phase_from_play_offs(order: list[int], rounds: tuple[tuple[str, int], ...],
                                      generator: np.random.Generator,
                                      play_offs: dict[tuple[int, int], list[Tie]]) -> Tie:
    """The bracket of `uefa_league_phase_2024` from the round of 16, given the knockout
    play-offs' ties keyed by the pair of places (9/10, 11/12, 13/14, 15/16) of their seeded
    clubs: drawn, or the world's once played."""
    names = [name for name, _ in rounds]
    legs = dict(rounds)

    def shuffle_pair(first_place: int, second_place: int) -> list[int]:
        pair = [order[first_place - 1], order[second_place - 1]]
        return [pair[index] for index in generator.permutation(2)]

    halves: list[dict] = [{}, {}]
    for seeded, play_off in (((1, 2), (15, 16)), ((3, 4), (13, 14)), ((5, 6), (11, 12)),
                             ((7, 8), (9, 10))):  # fmt: skip
        clubs = shuffle_pair(*seeded)
        ties = [play_offs[play_off][index] for index in generator.permutation(2)]
        for half in range(2):
            halves[half][seeded] = Tie(names[1], legs[names[1]], ties[half], clubs[half])
    semi_finals = []
    for half in halves:
        quarter_finals = [_draw_tie_order(names[2], legs[names[2]], half[(1, 2)], half[(7, 8)],
                                          generator),
                          _draw_tie_order(names[2], legs[names[2]], half[(3, 4)], half[(5, 6)],
                                          generator)]  # fmt: skip
        semi_finals.append(_draw_tie_order(names[3], legs[names[3]], *quarter_finals, generator))
    return Tie(names[4], legs[names[4]], semi_finals[0], semi_finals[1], neutral=True)


def _group_winners_v_runners_up(places, rounds, generator) -> Tie:
    groups = sorted(places)
    names = [name for name, _ in rounds]
    legs = dict(rounds)
    while True:  # a runner-up never meets the winner of its own group
        drawn = generator.permutation(len(groups))
        if all(drawn[index] != index for index in range(len(groups))):
            break
    ties = [Tie(names[0], legs[names[0]], places[groups[index]][1], places[groups[target]][0])
            for index, target in enumerate(drawn)]  # fmt: skip
    ties = [ties[index] for index in generator.permutation(len(ties))]
    for name in names[1:]:
        neutral = name == names[-1]
        ties = [_draw_tie_order(name, legs[name], ties[index], ties[index + 1], generator, neutral)
                for index in range(0, len(ties), 2)]  # fmt: skip
    return ties[0]


def _build_uefa_round_of_32(places, rounds, generator, drop_ins) -> Tie:
    """The Europa League to 2020-21: its group winners and the higher cup's best thirds (the
    first half of `drop_ins`) against its runners-up and the other thirds, a winner never against
    the runner-up of its own group, the seeded club at home in the second leg; the later rounds
    drawn afresh."""
    groups = sorted(places)
    half = len(drop_ins) // 2
    seeded = [(group, places[group][0]) for group in groups]
    seeded += [(None, club) for club in drop_ins[:half]]
    unseeded = [(group, places[group][1]) for group in groups]
    unseeded += [(None, club) for club in drop_ins[half:]]
    if len(seeded) != len(unseeded):
        raise ValueError("the round of 32 needs as many seeded clubs as unseeded")
    names = [name for name, _ in rounds]
    legs = dict(rounds)
    while True:
        drawn = generator.permutation(len(unseeded))
        if all(seeded[index][0] is None or seeded[index][0] != unseeded[target][0]
               for index, target in enumerate(drawn)):  # fmt: skip
            break
    ties = [Tie(names[0], legs[names[0]], unseeded[target][1], seeded[index][1])
            for index, target in enumerate(drawn)]  # fmt: skip
    return open_draws(ties, rounds[1:], generator)


def _build_uefa_knockout_play_offs(places, rounds, generator, drop_ins) -> Tie:
    """The group stages of 2021-22 to 2023-24: the runners-up against the higher cup's thirds in
    the knockout round play-offs, the runner-up at home in the second leg; the group winners
    against the play-offs' winners in the round of 16, the winner at home in the second leg; the
    later rounds drawn afresh."""
    groups = sorted(places)
    if len(drop_ins) != len(groups):
        raise ValueError(f"{len(groups)} runners-up need as many clubs from the higher cup,"
                         f" not {len(drop_ins)}")  # fmt: skip
    names = [name for name, _ in rounds]
    legs = dict(rounds)
    order = generator.permutation(len(drop_ins))
    play_offs = [Tie(names[0], legs[names[0]], drop_ins[order[index]], places[group][1])
                 for index, group in enumerate(groups)]  # fmt: skip
    return build_winners_v_play_offs([places[group][0] for group in groups], play_offs, rounds[1:],
                                     generator)  # fmt: skip


def build_winners_v_play_offs(
    winners: list[int],
    play_offs: list,
    rounds: tuple[tuple[str, int], ...],
    generator: np.random.Generator,
) -> Tie:
    """The round of 16 of the group stages of 2021-22 to 2023-24 from the knockout play-offs'
    ties (drawn, or the world's once played): each group winner drawn against a play-off's
    winner, at home in the second leg; the later rounds drawn afresh."""
    name, legs = rounds[0]
    order = generator.permutation(len(play_offs))
    last_16 = [Tie(name, legs, play_offs[order[index]], winner)
               for index, winner in enumerate(winners)]  # fmt: skip
    return open_draws(last_16, rounds[1:], generator)


def open_draws(ties: list, rounds: tuple[tuple[str, int], ...],
               generator: np.random.Generator) -> Tie:  # fmt: skip
    """The rounds after `ties` (ties or clubs), each drawn afresh with no seeding, which side is
    at home first drawn too; the last round, the final, on neutral ground."""
    if not rounds:
        if len(ties) != 1:
            raise ValueError(f"{len(ties)} ties and no round left")
        return ties[0]
    ties = [ties[index] for index in generator.permutation(len(ties))]
    for position, (name, legs) in enumerate(rounds):
        neutral = position == len(rounds) - 1
        ties = [_draw_tie_order(name, legs, ties[index], ties[index + 1], generator, neutral)
                for index in range(0, len(ties), 2)]  # fmt: skip
    if len(ties) != 1:
        raise ValueError(f"{len(ties)} ties left after the final")
    return ties[0]


def _build_world_cup_32(places, rounds, _generator) -> Tie:
    names = [name for name, _ in rounds]

    def make_tie(name, first, second) -> Tie:
        return Tie(name, 1, first, second, neutral=True)

    def pair_winner_with_runner_up(winner_group: str, runner_up_group: str) -> Tie:
        return make_tie(names[0], places[winner_group][0], places[runner_up_group][1])

    halves = []
    for pairs in ((("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")),
                  (("B", "A"), ("D", "C"), ("F", "E"), ("H", "G"))):  # fmt: skip
        last_16 = [pair_winner_with_runner_up(*pair) for pair in pairs]
        quarter_finals = [make_tie(names[1], last_16[0], last_16[1]),
                          make_tie(names[1], last_16[2], last_16[3])]  # fmt: skip
        halves.append(make_tie(names[2], *quarter_finals))
    return make_tie(names[3], *halves)


# art. 9.9: for the groups of the four thirds that go on, the third each of 1A, 1B, 1C, 1D meets
ASIAN_CUP_2027_THIRDS = {
    "ABCD": "CDAB", "ABCE": "CABE", "ABCF": "CABF", "ABDE": "DABE", "ABDF": "DABF",
    "ABEF": "EABF", "ACDE": "CDAE", "ACDF": "CDAF", "ACEF": "CAFE", "ADEF": "DAFE",
    "BCDE": "CDBE", "BCDF": "CDBF", "BCEF": "ECBF", "BDEF": "EDBF", "CDEF": "CDFE",
}  # fmt: skip


def _build_afc_asian_cup_2027(places, rounds, _generator, thirds) -> Tie:
    names = [name for name, _ in rounds]
    met = ASIAN_CUP_2027_THIRDS["".join(sorted(thirds))]
    third = dict(zip("ABCD", (places[group][2] for group in met), strict=True))

    def make_tie(round_index, first, second) -> Tie:
        return Tie(names[round_index], 1, first, second, neutral=True)

    def pick_club(group: str, place: int) -> int:
        return places[group][place - 1]

    last_16 = [
        make_tie(0, pick_club("A", 2), pick_club("C", 2)),
        make_tie(0, pick_club("D", 1), third["D"]),
        make_tie(0, pick_club("B", 1), third["B"]),
        make_tie(0, pick_club("F", 1), pick_club("E", 2)),
        make_tie(0, pick_club("C", 1), third["C"]),
        make_tie(0, pick_club("E", 1), pick_club("D", 2)),
        make_tie(0, pick_club("A", 1), third["A"]),
        make_tie(0, pick_club("B", 2), pick_club("F", 2)),
    ]
    quarter_finals = [make_tie(1, last_16[index], last_16[index + 1]) for index in range(0, 8, 2)]
    semi_finals = [make_tie(2, quarter_finals[0], quarter_finals[1]),
                   make_tie(2, quarter_finals[2], quarter_finals[3])]  # fmt: skip
    return make_tie(3, *semi_finals)


# the third-placed team each group winner meets, by the groups whose thirds go on
THIRDS_TABLES = {
    # Copa América 2019 (Art. 17): 1A plays 3B or 3C, 1C plays 3A or 3B
    "copa_2019": {"AB": {"1A": "B", "1C": "A"}, "AC": {"1A": "C", "1C": "A"},
                  "BC": {"1A": "C", "1C": "B"}},
    # six groups of four, the four best thirds (the AFC Asian Cup's art. 9.10, the AFCON's art.
    # 75.6): the opponents of 1A, 1B, 1C and 1D
    "six_groups": {groups: dict(zip(("1A", "1B", "1C", "1D"), opponents, strict=True))
                   for groups, opponents in ASIAN_CUP_2027_THIRDS.items()},
    # the EURO of 24 (art. 21.05 of the 2018-20 and 2022-24 regulations): the opponents of 1B, 1C,
    # 1E and 1F
    "euro_24": {groups: dict(zip(("1B", "1C", "1E", "1F"), opponents, strict=True))
                for groups, opponents in {
                    "ABCD": "ADBC", "ABCE": "AEBC", "ABCF": "AFBC", "ABDE": "DEAB",
                    "ABDF": "DFAB", "ABEF": "EFBA", "ACDE": "EDCA", "ACDF": "FDCA",
                    "ACEF": "EFCA", "ADEF": "EFDA", "BCDE": "EDBC", "BCDF": "FDCB",
                    "BCEF": "FECB", "BDEF": "FEDB", "CDEF": "FEDC"}.items()},
}


def build_fixed_bracket(
    spec: dict,
    places: dict[str, list[int]],
    rounds: tuple[tuple[str, int], ...],
    thirds: tuple[str, ...] = (),
) -> Tie:
    """A bracket written in the regulations, every match a single one on neutral ground (a host
    plays at home through the strength): `first_round` lists its ties as "1A-2B", a third as
    "3>1A" (the third the format's `thirds_table` gives 1A); `later` lists, per later round, the
    pairs of the previous round's ties (numbered from 1) whose winners meet."""
    table = THIRDS_TABLES[spec["thirds_table"]][''.join(sorted(thirds))] if thirds else {}
    names = [name for name, _ in rounds]

    def resolve_slot(slot: str) -> int:
        if slot.startswith("3>"):
            return places[table[slot[2:]]][2]
        return places[slot[1]][int(slot[0]) - 1]

    ties = [Tie(names[0], 1, resolve_slot(first), resolve_slot(second), neutral=True)
            for first, second in (pair.split("-") for pair in spec["first_round"])]  # fmt: skip
    for name, pairs in zip(names[1:], spec["later"], strict=True):
        ties = [Tie(name, 1, ties[first - 1], ties[second - 1], neutral=True)
                for first, second in pairs]  # fmt: skip
    if len(ties) != 1:
        raise ValueError(f"the fixed bracket ends with {len(ties)} ties, not a final")
    return ties[0]


def _draw_tie_order(name, legs, first, second, generator, neutral=False) -> Tie:
    if generator.random() < 0.5:
        first, second = second, first
    return Tie(name, legs, first, second, neutral=neutral)


RULES = {
    "uefa_league_phase_2024": _build_uefa_league_phase_2024,
    "group_winners_v_runners_up": _group_winners_v_runners_up,
    "world_cup_32": _build_world_cup_32,
    "afc_asian_cup_2027": _build_afc_asian_cup_2027,
    "uefa_round_of_32": _build_uefa_round_of_32,
    "uefa_knockout_play_offs": _build_uefa_knockout_play_offs,
}
WITH_THIRDS = {"afc_asian_cup_2027"}
WITH_DROP_INS = {"uefa_round_of_32", "uefa_knockout_play_offs"}
