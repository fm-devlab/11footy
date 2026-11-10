"""What the scorers publish before a match, if the player plays: he scores (`P3.scores`), his goals
(`P3.goals`), he scores the match's first goal (`P3.first_scorer`), he assists (`P5.assists`), he
scores or assists (`P5.scores_or_assists`), from the model that passed its verdicts
(development and blind).

Each side's players are the lineups' pool (`TeamLineup.pool`: the chance to play and the expected
minutes if he plays); their rates come from their league matches of this season and the one
before, with our expected goals of every shot as the morning rated them; the side's goals are the
champion's score matrix of the fixture. The groups' and the leagues' rates of a season come from
its five seasons before and do not change during it: built once a season and kept in the
cache folder (`players-priors-<season>.json`, with the release's identity).
"""

import bisect
import json
import math
import sqlite3
from dataclasses import asdict
from pathlib import Path

import numpy as np

from vesta.forecast_log.writing import PublishedDistribution
from vesta.lineups.model import read_world
from vesta.lineups.publication import TeamLineup
from vesta.players.model import (
    GroupRates,
    build_histories,
    estimate_league_rates,
    find_first_scoring_side,
    group_rates,
    read_happened,
    read_penalty_conversion,
    read_shot_players,
    share_out_side_goals,
    sum_shots_by_player,
    weigh_player,
)
from vesta.spine.timing import TimeRules
from vesta.spine.world import read_league_matches

TRAINING_SEASONS = 5
GOAL_CELLS = 6  # P3.goals: 0 to 5, then the tail of 6 or more
KEYS = ("P3.scores", "P3.goals", "P3.first_scorer", "P5.assists", "P5.scores_or_assists")
# a side's scorers are published while its latest league match with the source's shots is at
# most this many of its league matches back (the check of 4 November 2026); beyond it the side
# is held
STALE_MATCHES = 4


def _read_season_inputs(world: sqlite3.Connection, leagues: list[str], first: int, last: int,
                        rules: TimeRules, rated: dict, shot_section: dict, now: str | None = None,
                        sheets=None, covered: set[int] | None = None):  # fmt: skip
    """The league matches of the seasons (played before `now`), each player's appearances in
    them with what he did, and the facts the rates are read from; `sheets`, the lineups' world
    of the same seasons when the caller has it."""
    matches = [match for match in read_league_matches(world, leagues, first, last)
               if now is None or match.kickoff_utc < now]  # fmt: skip
    players = read_shot_players(world, shot_section["competitions"], shot_section["first_season"],
                                rated["last_season"], shot_section["layers"])  # fmt: skip
    shots = rated["shots"]
    if len(players) != len(shots):
        raise RuntimeError(f"the shots' players ({len(players)}) do not line up with the shots"
                           f" rated ({len(shots)})")  # fmt: skip
    sums = sum_shots_by_player(shots.match_id, rated["chances"], players)
    by_id = {match.match_id: match for match in matches}
    happened, own_goals = read_happened(world, set(by_id))
    sheets = sheets or read_world(world, first - 1, last, rules)
    histories, groups_of = build_histories(sheets, by_id, sums, happened, covered)
    return matches, by_id, happened, own_goals, sheets, histories, groups_of


def build_priors(world: sqlite3.Connection, leagues: list[str], season: int, rules: TimeRules,
                 rated: dict, shot_section: dict) -> dict:  # fmt: skip
    """The groups' rates and the leagues' rates of `season`, from its five seasons before."""
    first = season - TRAINING_SEASONS
    matches, by_id, happened, own_goals, sheets, histories, groups_of = _read_season_inputs(
        world, leagues, first, season - 1, rules, rated, shot_section)  # fmt: skip
    firsts = {
        match_id: find_first_scoring_side(by_id[match_id], happened, sheets) for match_id in by_id
    }
    assists_by_match: dict[int, int] = {}
    for (match_id, _), item in happened.items():
        assists_by_match[match_id] = assists_by_match.get(match_id, 0) + item.assists
    rates = group_rates([(groups_of[(player, index)], item)
                         for player, items in histories.items()
                         for index, item in enumerate(items)])  # fmt: skip
    leagues_found = estimate_league_rates(
        matches,
        assists_by_match,
        own_goals,
        firsts,
        read_penalty_conversion(world, leagues, season),
    )
    return {"groups": {str(group): asdict(item) for group, item in rates.items()},
            "leagues": leagues_found}  # fmt: skip


def load_or_build_priors(
    world: sqlite3.Connection,
    leagues: list[str],
    season: int,
    rules: TimeRules,
    rated: dict,
    shot_section: dict,
    folder: Path,
    identity: str,
) -> dict:
    """The season's rates from `folder`, built and kept there when missing or of another
    release."""
    path = folder / f"players-priors-{season}.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("identity") == identity:
            return stored["priors"]
    priors = build_priors(world, leagues, season, rules, rated, shot_section)
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"identity": identity, "priors": priors}), encoding="utf-8")
    return priors


class PlayersOfTheDay:
    """The scorers of the morning's league fixtures."""

    def __init__(self, world: sqlite3.Connection, leagues: list[str], season: int, now: str,
                 rules: TimeRules, rated: dict, shot_section: dict, folder: Path,
                 identity: str, sheets=None):  # fmt: skip
        priors = load_or_build_priors(world, leagues, season, rules, rated, shot_section, folder,
                                      identity)  # fmt: skip
        self.groups = {None if group == "None" else group: GroupRates(**values)
                       for group, values in priors["groups"].items()}  # fmt: skip
        self.leagues = priors["leagues"]
        covered = {int(item) for item in rated["shots"].match_id}
        matches, _, _, _, sheets, histories, _ = _read_season_inputs(
            world, leagues, season - 1, season, rules, rated, shot_section, now, sheets, covered
        )
        self.histories = histories
        self.uncovered = find_uncovered_last_matches(matches, sheets, covered)
        self.without_shots = count_trailing_without_shots(matches, covered)
        self.held: list[tuple[str, int, list[int]]] = []  # competition, team, matches
        self.stale: set[int] = set()  # the sides published on shots of earlier matches
        self.known_at = {player: [item.known_at for item in items]
                         for player, items in histories.items()}  # fmt: skip
        self.season = season
        self.now = now

    def build_side_distributions(self, lineup: TeamLineup, matrix: np.ndarray, home: bool,
                                 competition: str) -> list[PublishedDistribution]:  # fmt: skip
        """The scorers' distributions of a side's pool (version empty: the gate sets it)."""
        league = self.leagues.get(competition)
        if league is None or not lineup.pool:
            return []
        missing = sorted({self.uncovered[item[0]] for item in lineup.pool
                          if item[0] in self.uncovered})  # fmt: skip
        if missing and self.without_shots.get(lineup.team_id, 0) > STALE_MATCHES:
            # the source is out longer than the check allows: the side waits for it
            self.held.append((competition, lineup.team_id, missing))
            return []
        if missing:  # rated without the shots of their latest matches: said by a caveat
            self.stale.add(lineup.team_id)
        rates = [self.rate_player(player, group, competition) for player, group, *_ in lineup.pool]
        return build_scorer_distributions(
            [player for player, *_ in lineup.pool],
            rates,
            [item[3] for item in lineup.pool],
            [item[4] for item in lineup.pool],
            matrix,
            home,
            league,
        )

    def rate_player(self, player: int, group: str | None, competition: str) -> tuple[float, float]:
        """His goals and assists per 90 (penalties at the league's conversion), from what was
        known this morning: the lineup time rescales them by its own expected minutes."""
        history = self.histories.get(player, [])
        known = history[:bisect.bisect_right(self.known_at.get(player, []), self.now)]
        rates = self.groups.get(group, self.groups[None])
        return weigh_player(known, self.season, rates, self.leagues[competition]["conversion"],
                            90.0)  # fmt: skip


def build_scorer_distributions(
    players: list[int],
    rates: list[tuple[float, float]],
    plays: list[float],
    expected_minutes: list[float],
    matrix: np.ndarray,
    home: bool,
    league: dict,
) -> list[PublishedDistribution]:
    """The scorers' distributions of a side (version empty: the gate sets it): each player's
    goals and assists per 90 times his expected minutes if he plays, shared out of the side's
    goals of the score matrix by `side_expectations`."""
    raw_goals = np.array([goals * minutes / 90 for (goals, _), minutes
                          in zip(rates, expected_minutes, strict=True)])  # fmt: skip
    raw_assists = np.array([assists * minutes / 90 for (_, assists), minutes
                            in zip(rates, expected_minutes, strict=True)])  # fmt: skip
    side = share_out_side_goals(raw_goals, raw_assists, np.array(plays), matrix, home, league)
    found = []
    for index, player in enumerate(players):
        goals, assists = float(side.goals[index]), float(side.assists[index])
        found += [
            PublishedDistribution("P3.scores", player, "", {"yes": 1 - math.exp(-goals)}),
            PublishedDistribution("P3.goals", player, "", make_poisson_count(goals)),
            PublishedDistribution("P3.first_scorer", player, "",
                                  {"yes": min(float(side.first[index]), 1.0)}),
            PublishedDistribution("P5.assists", player, "", {"yes": 1 - math.exp(-assists)}),
            PublishedDistribution("P5.scores_or_assists", player, "",
                                  {"yes": 1 - math.exp(-goals - assists)})]  # fmt: skip
    return found


def count_trailing_without_shots(matches, covered: set[int]) -> dict[int, int]:
    """team -> how many of its latest league matches (of `matches`, by kick-off) have no shots of
    the expected goals' source."""
    found: dict[int, int] = {}
    for match in sorted(matches, key=lambda item: (item.kickoff_utc, item.match_id)):
        for side in (match.home_team_id, match.away_team_id):
            found[side] = 0 if match.match_id in covered else found.get(side, 0) + 1
    return found


def find_uncovered_last_matches(matches, sheets, covered: set[int]) -> dict[int, int]:
    """player -> his last league match (of `matches`, by kick-off) when it has no shots of the
    expected goals' source: the morning would read it as played with no expected goals, so his
    side's scorers wait for the source (`covered`, the matches with at least one shot)."""
    last: dict[int, int] = {}
    for match in sorted(matches, key=lambda item: (item.kickoff_utc, item.match_id)):
        for side in (match.home_team_id, match.away_team_id):
            for player, (started, minutes, _) in sheets.sheets.get((match.match_id, side),
                                                                   {}).items():  # fmt: skip
                if started or minutes > 0:
                    last[player] = match.match_id
    return {player: match_id for player, match_id in last.items() if match_id not in covered}


def make_poisson_count(mean: float) -> dict:
    """A count distribution: 0 to 5 goals, then the tail of 6 or more."""
    cells = [math.exp(-mean) * mean ** k / math.factorial(k) for k in range(GOAL_CELLS)]
    return {"probabilities": cells, "tail_from": GOAL_CELLS, "tail": max(1.0 - sum(cells), 0.0)}
