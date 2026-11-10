"""The tie-break rules against history: every finished league season of the world ranked by the
rule of `config/vesta/tie-break-rules.csv`, beside the world's recorded final positions.

    python -m vesta.season.standings_check <path of football-world.db>

The table of a season is rebuilt from its official results (an awarded result counts as awarded)
and the points deducted at its last matchday, then ranked by the competition's rule of that
season. Every team placed differently from the world's last standings is listed; a place that
only the drawing of lots decided is marked, as no rule can reproduce it. The check means something
only where the world's positions and deductions are the official ones, read from a source: where
the world ranks its own tables (by points, goal difference and goals, the build of 25 October
2026), it only measures how often the rule departs from that order. A row whose seasons all agree
is not proof of the rule, only of its agreement with the seasons played (the primary sources
decide).
"""

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vesta.season.standings import (
    SeasonMatches,
    TieBreakRule,
    count_season_totals,
    rank,
    read_tie_break_rule,
)
from vesta.spine.world import open_world

LOTS_SEEDS = (1, 2, 3)  # a place that changes with the seed of the lots was decided by them


@dataclass(frozen=True)
class Disagreement:
    competition: str
    season: int
    team_id: int
    ours: int  # 1 the first
    world: int
    by_lots: bool


def compare_season(
    team_ids: list[int],
    results: list[tuple[int, int, int, int]],
    deductions: dict[int, int],
    world_positions: dict[int, int],
    rule: TieBreakRule,
) -> list[tuple[int, int, int, bool]]:
    """(team, our place, the world's place, decided by lots) of every team placed differently."""
    column = {team: index for index, team in enumerate(team_ids)}
    rows = np.array(results, dtype=np.int64).reshape(-1, 4)
    matches = SeasonMatches(
        np.array([column[team] for team in rows[:, 0].tolist()], dtype=np.int64),
        np.array([column[team] for team in rows[:, 1].tolist()], dtype=np.int64),
        rows[None, :, 2],
        rows[None, :, 3],
    )
    totals = count_season_totals(
        matches, len(team_ids), np.array([deductions.get(team, 0) for team in team_ids])
    )
    orders = [rank(totals, matches, rule, np.random.default_rng(seed))[0] for seed in LOTS_SEEDS]
    differences = []
    for team, index in column.items():
        ours = int(orders[0][index]) + 1
        if ours != world_positions.get(team):
            by_lots = len({int(order[index]) for order in orders}) > 1
            differences.append((team, ours, world_positions.get(team, 0), by_lots))
    return differences


def check_world(world: sqlite3.Connection) -> tuple[int, list[Disagreement], list[str]]:
    """(seasons compared, the disagreements, the seasons without a rule or a table)."""
    compared, disagreements, skipped = 0, [], []
    seasons = world.execute(
        "SELECT s.season_id, c.code, s.season FROM seasons s JOIN competitions c USING"
        " (competition_id) WHERE c.kind = 'league' AND s.status = 'closed'"
        " ORDER BY c.code, s.season"
    ).fetchall()
    for season_id, competition, season in seasons:
        try:
            rule = read_tie_break_rule(competition, season)
        except ValueError:
            skipped.append(f"{competition} {season}: no rule")
            continue
        last = world.execute("SELECT MAX(after_matchday) FROM standings WHERE season_id = ?",
                             (season_id,)).fetchone()[0]  # fmt: skip
        if last is None:
            skipped.append(f"{competition} {season}: no standings")
            continue
        standing = world.execute(
            "SELECT team_id, position, points_deducted FROM standings WHERE season_id = ?"
            " AND after_matchday = ?", (season_id, last)).fetchall()  # fmt: skip
        results = world.execute(
            "SELECT home_team_id, away_team_id, home_goals, away_goals FROM matches WHERE"
            " season_id = ? AND status IN ('played', 'awarded') AND home_goals IS NOT NULL",
            (season_id,),
        ).fetchall()
        team_ids = sorted({row[0] for row in standing} | {row[0] for row in results}
                          | {row[1] for row in results})  # fmt: skip
        differences = compare_season(team_ids, results,
                                     {team: deducted for team, _, deducted in standing},
                                     {team: position for team, position, _ in standing},
                                     rule)  # fmt: skip
        compared += 1
        disagreements += [Disagreement(competition, season, *difference)
                          for difference in differences]  # fmt: skip
    return compared, disagreements, skipped


def main(arguments: list[str]) -> int:
    if len(arguments) != 1:
        print(__doc__)
        return 2
    with open_world(Path(arguments[0])) as world:
        compared, disagreements, skipped = check_world(world)
    print(f"{compared} league seasons compared; {len(skipped)} skipped")
    for line in skipped:
        print(f"  skipped: {line}")
    seasons = sorted({(item.competition, item.season) for item in disagreements})
    print(f"{len(seasons)} seasons with a team placed differently")
    for item in disagreements:
        note = " (decided by lots: no rule can reproduce it)" if item.by_lots else ""
        print(f"  {item.competition} {item.season} team {item.team_id}: ours {item.ours}, "
              f"the world's {item.world}{note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
