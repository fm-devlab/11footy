"""The families beyond result and goals for the fixtures of one round, on the user's PC: the team
counts, goals in time, the players and the lineups, built by the morning job's own code
(`vesta.jobs.pre_match`) from the dataset `vesta.run` reads, at the instant its fixtures are
forecast from.

The season's parts each family keeps (the team counts' corrections, the lineups', scorers' and
thresholds' priors, the champion's past) are kept in the run's cache under an identity made of
the release and the competitions the set holds: a later dataset of the same files reuses them, a
different set of files builds its own.
"""

import hashlib
import sqlite3
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from vesta.forecast_log.writing import PublishedDistribution
from vesta.jobs.fixtures import FixtureForecast
from vesta.jobs.past_cache import hash_release
from vesta.jobs.pre_match import (
    build_distributions,
    fit_current_timing,
    forecast_team_counts,
    forecast_who_plays,
    read_current_season,
)
from vesta.spine.goals_in_time import GoalsInTimeParameters, TimeShares

GOALS_IN_TIME_PREFIXES = ("M7.", "M8.", "M9.", "M10.", "M11.", "M12.", "M13.")
LINEUP_KEY = "L1.starts"  # the players' keys read the lineups: they go where it is served


@dataclass
class RoundFamilies:
    """What the families built for the fixtures forecast from one instant."""

    shares: TimeShares | None = None
    parameters: GoalsInTimeParameters | None = None
    team_counts: object = None
    cup_counts: object = None
    sides: dict = field(default_factory=dict)  # match -> each side's `TeamLineup`
    scorers: dict = field(default_factory=dict)  # match -> the players' distributions
    stale: set[int] = field(default_factory=set)  # sides whose scorers read earlier shots
    held: list = field(default_factory=list)  # (competition, team, matches) scorers held
    bans_read: bool = False
    failed: dict[str, str] = field(default_factory=dict)  # family -> why it was left out

    def build(self, forecast: FixtureForecast, versions: dict[tuple[str, int], str],
              competition_id: int) -> list[PublishedDistribution]:  # fmt: skip
        """The fixture's distributions whose (key, competition) `versions` holds, result and
        goals included, each with that version."""
        match_id = forecast.fixture.match_id
        return build_distributions(forecast, self.shares, self.parameters, versions,
                                   competition_id, self.team_counts, self.sides.get(match_id, ()),
                                   self.scorers.get(match_id, ()), self.cup_counts)  # fmt: skip


def make_identity(root: Path, codes: set[str]) -> str:
    """The cache's identity of the families' season parts: the release and the set's
    competitions."""
    held = hashlib.sha256(" ".join(sorted(codes)).encode()).hexdigest()[:16]
    return f"{hash_release(root)}-{held}"


def forecast_round_families(connection: sqlite3.Connection, forecasts: list[FixtureForecast],
                            as_of: str, versions: dict[tuple[str, int], str], ids: dict,
                            spine: dict, expected_goals: dict, cache: Path, identity: str,
                            rated: dict, past_matrices: dict[int, np.ndarray],
                            timing: dict) -> RoundFamilies:  # fmt: skip
    """The families of `forecasts`, all made at `as_of`, for the keys `versions` serves: only the
    families with a key there are built. `rated` and `past_matrices` are the champion's
    (`vesta.jobs.fixtures.forecast_league_fixtures`), empty without it; `timing` keeps goals in
    time's fit by season between calls. The lineups and the players are built for the fixtures
    of the competitions where `versions` serves the lineups (`ids`: code -> competition id).

    A family whose model cannot be fitted on the dataset (too few matches of a kind, a singular
    fit) is left out of the round and named in `failed`: the user's run publishes the rest."""
    keys = {key for key, _ in versions}
    found = RoundFamilies()
    if not forecasts or not keys:
        return found
    leagues = list(spine["walk_forward"]["competitions"])
    season = read_current_season(connection, leagues, as_of)
    if any(key.startswith(GOALS_IN_TIME_PREFIXES) for key in keys):
        if season not in timing:
            timing[season] = _contain(found, "goals in time",
                                      lambda: fit_current_timing(connection, spine, season))
        if timing[season] is not None:
            found.shares, added = timing[season]
            found.parameters = replace(GoalsInTimeParameters.from_config(spine),
                                       added_time_minutes=added)  # fmt: skip
    fixtures = [item.fixture for item in forecasts]
    counts = _contain(found, "team counts", lambda: forecast_team_counts(
        connection, leagues, season, as_of, spine, versions, fixtures, cache, identity))
    found.team_counts, found.cup_counts = counts or (None, None)
    with_lineups = {competition for key, competition in versions if key == LINEUP_KEY}
    chosen = [item for item in forecasts if ids[item.fixture.competition] in with_lineups]
    if not chosen:
        return found
    who_plays = _contain(found, "lineups and players", lambda: forecast_who_plays(
        connection, leagues, season, as_of, spine, expected_goals, cache, identity,
        {pair: version for pair, version in versions.items() if pair[1] in with_lineups},
        chosen, past_matrices, rated))  # fmt: skip
    if who_plays is None:
        return found
    lineups, players, _, found.sides, found.scorers = who_plays
    found.bans_read = lineups is not None and lineups.world.bans_read
    if players is not None:
        found.stale, found.held = players.stale, players.held
    return found


def _contain(found: RoundFamilies, family: str, step):
    """`step()`, or None with the family and its error in `found.failed`."""
    try:
        return step()
    except Exception as error:  # noqa: BLE001 - one family's failure leaves the others published
        found.failed[family] = f"{type(error).__name__}: {error}"
        return None
