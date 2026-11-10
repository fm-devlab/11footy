"""The UEFA cups' qualifiers (T4.qualifies.team): the chance a club reaches the main stage of the
cup whose qualifying round it plays.

UEFA draws a qualifying round before the round before it is played, with slots such as "the
winner of tie A" or "the loser of tie B" (the clubs a higher cup's qualifying drops down). At the
eve of round r of a cup, its ties of rounds r and r+1 are known where the calendar says round
r+1's draw came before round r's first matches (`config/vesta/season.toml` [qualifying_draws],
from UEFA's published dates); a round drawn later is not known, and a checkpoint whose path needs
it is not taken.

The ties known at an instant, across the three cups' qualifiers, form one tree: each club of a tie
is a club, or the winner or the loser of the tie it played just before, when that tie is in the
tree and undecided at the instant (read from the world: the club that came out of it, and
whether it won or lost it). Every draw plays the ties in kick-off order on the ties step's model
(`vesta.season.knockout.TieSampler`); a club reaches the cup's main stage when it wins its tie of
the cup's last qualifying round (the play-off round).
"""

import sqlite3
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vesta.season.knockout import Strength, Tie, TieSampler
from vesta.spine.ties import TieParameters, read_tie_rule
from vesta.spine.world import RESULT_STATUSES

SEASON_CONFIG = Path(__file__).resolve().parents[3] / "config" / "vesta" / "season.toml"
CUP_OF = {"CL": "CL", "CLQ": "CL", "EL": "EL", "ELQ": "EL", "ECL": "ECL", "ECLQ": "ECL"}
ROUNDS = ("preliminary", "Q1", "Q2", "Q3", "PO")
LAST_ROUND = "PO"


PLAY_OFF_STAGES = ("playoffs", "playoff round", "play-off round", "playoff")


def find_qualifying_round(code: str, stage: str) -> str | None:
    """The qualifying round of a world stage; None for a stage of the main competition. A
    qualifying competition (CLQ, ELQ, ECLQ) names its rounds "first round" and so on; before
    2020-21 the qualifiers sit in the cup itself, named "qualifying ...", "preliminary round" or
    "playoffs"."""
    stage = stage.lower()
    qualifying = code.endswith("Q")
    if "preliminary" in stage:
        return "preliminary"
    if stage in PLAY_OFF_STAGES:
        return "PO"
    if qualifying or "qualifying" in stage:
        for word, name in (("first", "Q1"), ("second", "Q2"), ("third", "Q3")):
            if word in stage:
                return name
    return None


@dataclass(frozen=True)
class QualifyingTie:
    tie_key: object
    cup: str  # CL, EL or ECL: the cup the tie qualifies for
    round_name: str
    kickoff: str  # its first match
    home: int  # the first match's home club
    away: int
    legs: int
    # kickoff, home, away, goals, played
    matches: tuple[tuple[str, int, int, int | None, int | None, bool], ...]
    winner: int | None  # from the goals, the season's away goals, then the world's winner


@dataclass(frozen=True)
class QualifyingSeason:
    season: int
    ties: tuple[QualifyingTie, ...]
    main_stage: dict[str, frozenset[int]]  # cup -> the clubs that played its main stage


def read_qualifying_season(world: sqlite3.Connection, season: int) -> QualifyingSeason:
    rows = world.execute(
        """
        SELECT c.code, LOWER(COALESCE(m.stage, '')), m.match_id, m.tie_id, m.kickoff_utc,
               m.home_team_id, m.away_team_id, m.home_goals, m.away_goals, m.status,
               m.winner_team_id, t.winner_team_id
        FROM matches m
        JOIN seasons s ON s.season_id = m.season_id
        JOIN competitions c ON c.competition_id = s.competition_id
        LEFT JOIN match_ties t ON t.tie_id = m.tie_id
        WHERE c.code IN ('CL', 'CLQ', 'EL', 'ELQ', 'ECL', 'ECLQ') AND s.season = ?
        ORDER BY m.kickoff_utc, m.match_id
        """,
        (season,),
    ).fetchall()
    grouped: dict[object, list] = {}
    main: dict[str, set[int]] = {}
    for code, stage, match_id, tie_id, kickoff, home, away, home_goals, away_goals, status, \
            match_winner, tie_winner in rows:  # fmt: skip
        cup = CUP_OF[code]
        name = find_qualifying_round(code, stage)
        if name is None:
            if code in ("CL", "EL", "ECL") and (stage.startswith("group")
                                                or stage == "league phase"):  # fmt: skip
                main.setdefault(cup, set()).update((home, away))
            continue
        key = tie_id if tie_id is not None else ("single", match_id)
        played = status in RESULT_STATUSES and home_goals is not None
        grouped.setdefault((cup, name, key), []).append(
            (kickoff, home, away, home_goals, away_goals, played, match_winner, tie_winner))
    ties = []
    for (cup, name, key), matches in grouped.items():
        matches.sort()
        rule = read_tie_rule({"CL": "CLQ", "EL": "ELQ", "ECL": "ECLQ"}[cup] if season >= 2020
                             else cup, season)  # fmt: skip
        winner = _find_tie_winner(matches, rule.away_goals)
        ties.append(QualifyingTie(key, cup, name, matches[0][0], matches[0][1], matches[0][2],
                                  len(matches),
                                  tuple(match[:6] for match in matches), winner))  # fmt: skip
    ties.sort(key=lambda tie: (tie.kickoff, str(tie.tie_key)))
    return QualifyingSeason(season, tuple(ties),
                            {cup: frozenset(clubs) for cup, clubs in main.items()})  # fmt: skip


def _find_tie_winner(matches, away_goals_rule) -> int | None:
    if not all(match[5] for match in matches):
        return None
    sides = (matches[0][1], matches[0][2])
    goals = dict.fromkeys(sides, 0)
    for _, home, away, home_goals, away_goals, *_ in matches:
        goals[home] += home_goals
        goals[away] += away_goals
    if goals[sides[0]] != goals[sides[1]]:
        return max(sides, key=goals.get)
    if len(matches) == 2 and away_goals_rule:
        away = {match[2]: match[4] for match in matches}
        if away[sides[0]] != away[sides[1]]:
            return max(sides, key=away.get)
    return matches[-1][7] or matches[-1][6]


# ---- the draws known at an instant ---------------------------------------------------------


def draw_dates(path: Path = SEASON_CONFIG) -> dict[tuple[str, int, str], str]:
    """(cup, season, round) -> the date of its draw, from UEFA's published calendar."""
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    return {(row["cup"], int(row["season"]), row["round"]): row["date"]
            for row in config.get("qualifying_draws", [])}  # fmt: skip


def find_checkpoints(season: QualifyingSeason, cup: str, dates: dict) -> dict[str, str]:
    """The eves of the cup's third qualifying round and play-off round whose path to the main
    stage was drawn by then: Q3's eve needs the play-off round drawn before Q3's first match."""
    found = {}
    first = {name: min((tie.kickoff for tie in season.ties
                        if tie.cup == cup and tie.round_name == name), default=None)
             for name in ("Q3", "PO")}  # fmt: skip
    for name in ("Q3", "PO"):
        if first[name] is None:
            continue
        instant = f"{first[name][:10]}T00:00:00Z"
        needed = ("Q3", "PO") if name == "Q3" else ("PO",)
        drawn = [dates.get((cup, season.season, round_name)) for round_name in needed]
        if all(date is not None and date < first[name][:10] for date in drawn):
            found[name] = instant
    return found


def find_known_ties(season: QualifyingSeason, instant: str, dates: dict) -> list[QualifyingTie]:
    """The qualifying ties of the three cups whose draw came before `instant` (the day before
    its first match at the latest) and that are not over at `instant`."""
    day = instant[:10]
    known = []
    for tie in season.ties:
        drawn = dates.get((tie.cup, season.season, tie.round_name))
        if drawn is None or drawn >= day:
            continue
        if all(match[0] < instant and match[5] for match in tie.matches):
            continue  # over: its clubs are known
        known.append(tie)
    return known


# ---- the simulation ------------------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    """A club, or the winner (`won`) or the loser of an earlier tie of the tree."""

    club: int | None = None
    tie: int | None = None
    won: bool = True


def link_ties(ties: list[QualifyingTie]) -> list[tuple[QualifyingTie, Slot, Slot]]:
    """Each tie with its two slots: a club whose previous tie in `ties` is the one it came out of
    is that tie's winner or loser."""
    order = sorted(ties, key=lambda tie: (tie.kickoff, str(tie.tie_key)))
    last_tie: dict[int, int] = {}
    linked = []
    for index, tie in enumerate(order):
        slots = []
        for club in (tie.home, tie.away):
            if club in last_tie:
                earlier = order[last_tie[club]]
                if earlier.winner is None:
                    raise ValueError(f"tie {earlier.tie_key}: no winner, its clubs cannot be"
                                     " linked")
                slots.append(Slot(tie=last_tie[club], won=earlier.winner == club))
            else:
                slots.append(Slot(club=club))
            last_tie[club] = index
        linked.append((tie, *slots))
    return linked


def simulate_qualifiers(
    linked: list[tuple[QualifyingTie, Slot, Slot]],
    instant: str,
    strength: Strength,
    rules: dict[str, object],
    parameters: TieParameters,
    draws: int,
    seed: int,
) -> dict[tuple[int, str], np.ndarray]:
    """Per club and cup, (draws,) whether the club won that cup's play-off round (reached its
    main stage); a pair never reached is absent."""
    generator = np.random.default_rng(seed)
    samplers = {cup: TieSampler(strength, rule, parameters, generator)
                for cup, rule in rules.items()}
    through: dict[tuple[int, str], np.ndarray] = {}
    for simulation in range(draws):
        winners: list[int] = []
        losers: list[int] = []
        for tie, first_slot, second_slot in linked:
            first = _resolve_slot_club(first_slot, winners, losers)
            second = _resolve_slot_club(second_slot, winners, losers)
            played = [match for match in tie.matches if match[0] < instant and match[5]]
            first_leg = None
            if tie.legs == 2 and len(played) == 1:
                first_leg = (played[0][3], played[0][4])
            node = Tie(tie.round_name, tie.legs, first, second, first_leg=first_leg)
            winner = samplers[tie.cup].draw_winner(node, first, second)
            winners.append(winner)
            losers.append(second if winner == first else first)
            if tie.round_name == LAST_ROUND:
                passed = through.setdefault((winner, tie.cup), np.zeros(draws, dtype=bool))
                passed[simulation] = True
    return through


def _resolve_slot_club(slot: Slot, winners: list[int], losers: list[int]) -> int:
    if slot.club is not None:
        return slot.club
    return winners[slot.tie] if slot.won else losers[slot.tie]


def find_subject_clubs(linked, cup: str, round_name: str) -> set[int]:
    """The clubs of the cup's ties of `round_name` and those entering its next round directly."""
    found = set()
    for tie, first, second in linked:
        if tie.cup != cup:
            continue
        if tie.round_name == round_name:
            found |= {tie.home, tie.away}
        elif ROUNDS.index(tie.round_name) == ROUNDS.index(round_name) + 1:
            found |= {slot.club for slot in (first, second) if slot.club is not None}
    return found
