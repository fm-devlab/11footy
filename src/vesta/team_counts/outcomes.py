"""What happened to a team count: one rule for the verdicts' readers and the check job
(family A's plan, §2).

A count is read from the first layer of its scope's order that holds it for both sides; a match
without one stays pending, never guessed. The feed writes 0 where it holds no count, so two of its
zeros mean unknown:

- a stub: both feed rows of a match have fouls 0 and corners 0 (only the goals and the cards
  recorded); its counts are unknown but for the cards;
- a competition's season whose offsides were not recorded: more than `UNRECORDED_SHARE` of its
  counted matches have no offside on either side. Its offsides are unknown; with fewer than
  `MINIMUM_FOR_UNRECORDED` counted matches (a season's first days) the share is not known yet and
  the offsides stay pending.
"""

import sqlite3
from collections import defaultdict

from vesta.data_format import FEED
from vesta.team_counts.model import COLUMNS, LAYER_ORDER, is_stub

# the provider of record first (decision C3); the cups have the feed only, and in the leagues
# the report holds no offsides, so they fall to the feed
LAYERS_OF_SCOPE = {"leagues": LAYER_ORDER, "cups": (FEED,)}
UNRECORDED_SHARE = 0.5
# the fewest counted matches the share is read on: a recorded season's share is 0.15 or less, an
# unrecorded one's 0.85 or more, so four already tell them apart (P < 0.02); every competition's
# season of the verdict world from 2013 has at least four (PL 2016, the fewest)
MINIMUM_FOR_UNRECORDED = 4


def are_offsides_unrecorded(zero_matches: int, counted: int) -> bool | None:
    """Whether a competition's season did not record its offsides, from its counted matches and
    those with no offside on either side; None while too few are counted to tell."""
    if counted < MINIMUM_FOR_UNRECORDED:
        return None
    return zero_matches > UNRECORDED_SHARE * counted


def read_match_counts(world: sqlite3.Connection, match_ids: set[int], count: str,
                      scope: str) -> dict[int, tuple[int, int, str]]:  # fmt: skip
    """Per match of `match_ids` whose `count` is known: (home, away, layer). A match left out is
    pending: no layer holds both sides, a feed stub (but for the cards), or offsides of a
    season that did not record them or cannot tell yet."""
    if not match_ids:
        return {}
    marks = ", ".join("?" for _ in match_ids)
    sides: dict[int, tuple[int, int, int]] = {
        match_id: (season_id, home, away) for match_id, season_id, home, away in world.execute(
            f"SELECT match_id, season_id, home_team_id, away_team_id FROM matches"
            f" WHERE match_id IN ({marks})", tuple(match_ids))}  # fmt: skip
    values, feed_rows = _read_count_rows(world, f"t.match_id IN ({marks})", tuple(match_ids), count)
    unrecorded: dict[int, bool | None] = {}
    if count == "offsides":
        season_ids = {season_id for season_id, _, _ in sides.values()}
        unrecorded = {season_id: _read_offsides_unrecorded(world, season_id)
                      for season_id in season_ids}  # fmt: skip
    found = {}
    for match_id, (season_id, home, away) in sides.items():
        layer = next((name for name in LAYERS_OF_SCOPE[scope]
                       if values.get((match_id, home, name)) is not None
                       and values.get((match_id, away, name)) is not None), None)  # fmt: skip
        if layer is None:
            continue
        if layer == FEED and count != "cards" and is_stub(feed_rows.get(match_id, [])):
            continue
        if layer == FEED and count == "offsides" and unrecorded[season_id] is not False:
            continue
        found[match_id] = (values[(match_id, home, layer)], values[(match_id, away, layer)],
                           layer)  # fmt: skip
    return found




def _read_count_rows(world: sqlite3.Connection, where: str, parameters: tuple, count: str):
    """Each (match, team, layer)'s value of `count`, and each match's feed rows (fouls,
    corners)."""
    values: dict[tuple[int, int, str], int] = {}
    feed_rows: dict[int, list[tuple[int | None, int | None]]] = defaultdict(list)
    for match_id, team_id, layer, value, fouls, corners in world.execute(
            f"SELECT t.match_id, t.team_id, t.layer, {COLUMNS[count]},"
            f" t.fouls, t.corners"
            f" FROM match_team_statistics t WHERE {where}", parameters):  # fmt: skip
        if value is not None:
            values[(match_id, team_id, layer)] = int(value)
        if layer == FEED:
            feed_rows[match_id].append((fouls, corners))
    return values, feed_rows


def _read_offsides_unrecorded(world: sqlite3.Connection, season_id: int) -> bool | None:
    """`offsides_unrecorded` of one competition's season, on its feed rows in the world."""
    offsides: dict[int, list[int]] = defaultdict(list)
    feed_rows: dict[int, list[tuple[int | None, int | None]]] = defaultdict(list)
    for match_id, value, fouls, corners in world.execute(
            "SELECT t.match_id, t.offsides, t.fouls, t.corners FROM match_team_statistics t"
            " JOIN matches m ON m.match_id = t.match_id"
            f" WHERE m.season_id = ? AND t.layer = '{FEED}'",
            (season_id,)):  # fmt: skip
        feed_rows[match_id].append((fouls, corners))
        if value is not None:
            offsides[match_id].append(int(value))
    counted = [values for match_id, values in offsides.items()
               if len(values) == 2 and not is_stub(feed_rows[match_id])]  # fmt: skip
    return are_offsides_unrecorded(sum(values == [0, 0] for values in counted), len(counted))
