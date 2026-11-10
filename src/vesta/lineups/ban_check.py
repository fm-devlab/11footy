"""The lineups' ban rule check (2 November 2026, and for the four leagues 3 November 2026): on
a world with
suspensions_by_rule, the released lineups model with and without the rule, the same rows and
weights, seasons 2025 and 2026. With `--competitions`, only those leagues' bans are applied.

    python -m vesta.lineups.ban_check <world> [--competitions LL SA PL BL]
"""

import sys
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np

from vesta.lineups.model import collect_targets, read_bans, read_world, scale_to_eleven
from vesta.lineups.verdict import ROOT, TRAINING_SEASONS, Priors, _score_log_loss, estimate_minutes
from vesta.spine.timing import TimeRules
from vesta.spine.world import open_world

SEASONS = (2025, 2026)


def main() -> None:
    world_path = Path(sys.argv[1])
    chosen = frozenset(sys.argv[3:]) if sys.argv[2:3] == ["--competitions"] else None
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    leagues = set(spine["walk_forward"]["competitions"])
    first = SEASONS[0] - TRAINING_SEASONS
    with open_world(world_path) as connection:
        world = read_world(connection, first - 1, SEASONS[-1], TimeRules.from_config(spine))
        bans, read = read_bans(connection, chosen)
    print(f"leagues: {sorted(chosen) if chosen is not None else 'every league'}", flush=True)
    print(f"bans read: {read}; matches with a ban: {len(bans)}; bans: "
          f"{sum(len(players) for players in bans.values())}", flush=True)
    world.bans, world.bans_read = {}, False  # every row without the rule; the rule at scoring only
    rows_of, teams_of = {}, {}
    for season in range(first, SEASONS[-1] + 1):
        rows_of[season], teams_of[season] = collect_targets(world, leagues, {season}, 0)

    violations = []
    for season in SEASONS:
        training = range(season - TRAINING_SEASONS, season)
        priors = Priors([r for y in training for r in rows_of[y]],
                        [t for y in training for t in teams_of[y]])  # fmt: skip
        grouped = defaultdict(list)
        for row in rows_of[season]:
            grouped[(row.match_id, row.team_id)].append(row)
        loss = defaultdict(lambda: defaultdict(float))  # (key, arm) -> league -> sum
        count = defaultdict(int)
        banned_count = sides_with = 0
        for (match_id, _), players in grouped.items():
            design = np.column_stack([np.ones(len(players)), [r.features for r in players]])
            goalkeepers = np.array([r.goalkeeper for r in players])
            log_odds = design @ priors.weights
            without = scale_to_eleven(log_odds, goalkeepers)
            out = bans.get(match_id, frozenset())
            keep = np.array([r.player_id not in out for r in players])
            with_rule = np.zeros(len(players))
            if keep.any():
                with_rule[keep] = scale_to_eleven(log_odds[keep], goalkeepers[keep])
            banned_here = int((~keep).sum())
            banned_count += banned_here
            sides_with += banned_here > 0
            for index, row in enumerate(players):
                played = row.started or row.minutes > 0
                if not keep[index] and played:
                    violations.append((season, match_id, row.team_id, row.player_id, row.started,
                                       row.minutes))  # fmt: skip
                league = row.competition
                count[league] += 1
                for arm, start in (("without", float(without[index])),
                                   ("with", float(with_rule[index]))):
                    if arm == "with" and not keep[index]:
                        plays = 0.0  # the rule decides it: out of the match
                    else:
                        comes_on, _ = estimate_minutes(priors, row, start)
                        plays = start + (1 - start) * comes_on
                    for key, own, hit in (("L1.starts", start, row.started),
                                          ("L6.plays", plays, played)):
                        probability = own if hit else 1 - own
                        loss[(key, arm)][league] += _score_log_loss(probability)
        total = sum(count.values())
        print(f"\n{season}: {total} player rows; {banned_count} banned rows on {sides_with} sides",
              flush=True)
        for key in ("L1.starts", "L6.plays"):
            loss_without = sum(loss[(key, "without")].values()) / total
            loss_with = sum(loss[(key, "with")].values()) / total
            print(f"  {key}: without {loss_without:.5f}, with the rule {loss_with:.5f},"
                  f" difference {loss_with - loss_without:+.5f}", flush=True)
            for league in sorted(count):
                league_without = loss[(key, "without")][league] / count[league]
                league_with = loss[(key, "with")][league] / count[league]
                print(f"    {league}: {league_without:.5f} -> {league_with:.5f}"
                      f" ({league_with - league_without:+.5f})", flush=True)
    print(f"\nviolations (a banned player who played): {len(violations)}", flush=True)
    for item in violations:
        print("  ", item, flush=True)


if __name__ == "__main__":
    main()
