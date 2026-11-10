"""What the player thresholds publish before a match, if the player plays: his shots
(`P1.shots`), shots on target (`P2.shots_on_target`), key passes (`P7.key_passes`), fouls
committed (`P8.fouls`) and drawn (`P8.fouls_drawn`), each a whole count distribution, and whether
he is booked (`P9.booked`), from the model that passed its verdicts
(development and blind).

The arithmetic is the verdict's (`vesta.players.thresholds`): his rates per 90 from his league
matches of this season and the one before, shrunk to his group's; the match's terms (his side's
expected goals against its recent level, from the champion's matrices; the opponent's and the
venue's terms of the weekly fouls and cards fits); his minutes if he plays, the lineups' published
distribution (`L6.minutes`), in bins; a negative binomial per count and group. The groups' rates
and the dispersions of a season come from its five seasons before and do not change during it:
built once a season and kept in the cache folder (`thresholds-priors-<season>.json`, with
the release's identity).
"""

import bisect
import json
import math
import sqlite3
import tomllib
from pathlib import Path

import numpy as np

from vesta.forecast_log.writing import PublishedDistribution
from vesta.lineups.model import read_world
from vesta.lineups.publication import TeamLineup
from vesta.players.model import compute_side_means
from vesta.players.thresholds import (
    KEYS as THRESHOLDS,
)
from vesta.players.thresholds import (
    GroupRates,
    MatchTerms,
    applies_to_player,
    build_histories,
    count_cells,
    estimate_own_rates,
    estimate_player_means,
    fit_sizes,
    group_rates,
    iterate_training_rows,
    read_counts,
    split_into_minute_bins,
)
from vesta.spine.timing import TimeRules, find_result_known_at, forecast_time
from vesta.spine.world import read_league_matches
from vesta.team_counts.model import CountMatch, FitParameters, read_count_matches

TRAINING_SEASONS = 5
# the keys of this release (vesta-2.6.0)
KEYS = ("P1.shots", "P2.shots_on_target", "P7.key_passes", "P8.fouls", "P8.fouls_drawn",
        "P9.booked", "P14.saves", "P16.offsides")  # fmt: skip
# candidate 2's keys: their expected counts scaled by the season's scale, fitted on the two
# seasons before it (`python -m vesta.players.thresholds_verdict <world> --scales-for <season>`)
SCALED = ("P14.saves", "P16.offsides")
SCALES_FILE = Path(__file__).resolve().parents[3] / "config/vesta/thresholds-scales.toml"
ROOT = Path(__file__).resolve().parents[3]


def _read_season_inputs(world: sqlite3.Connection, leagues: list[str], first: int, last: int,
                        rules: TimeRules, now: str | None = None, sheets=None):  # fmt: skip
    """The league matches of the seasons (played before `now`) and each player's appearances in
    them with his counts; `sheets`, the lineups' world of the same seasons when the caller has
    it."""
    matches = [match for match in read_league_matches(world, leagues, first, last)
               if now is None or match.kickoff_utc < now]  # fmt: skip
    by_id = {match.match_id: match for match in matches}
    sheets = sheets or read_world(world, first, last, rules)
    counted = read_counts(world, sheets.sheets, set(by_id))
    histories, groups_of = build_histories(sheets, by_id, counted)
    return matches, histories, groups_of


def build_priors(world: sqlite3.Connection, leagues: list[str], season: int,
                 rules: TimeRules) -> dict:  # fmt: skip
    """The groups' rates and the dispersions of `season`, from its five seasons before (and the
    season before them, which the first one's rates read), as the verdict fits them."""
    first = season - TRAINING_SEASONS
    _, histories, groups_of = _read_season_inputs(world, leagues, first - 1, season - 1, rules)
    training = set(range(first, season))
    rates = group_rates([(groups_of[(player, index)], item)
                         for player, items in histories.items()
                         for index, item in enumerate(items) if item.season in training])
    sizes = fit_sizes(iterate_training_rows(histories, groups_of, training, rates), rates)
    return {"groups": {str(group): {"per_ninety": item.per_ninety.tolist(),
                                    "per_match": item.per_match.tolist(),
                                    "saves_per_goal": item.saves_per_goal,
                                    "conceded_per_ninety": item.conceded_per_ninety}
                       for group, item in rates.items()},
            "sizes": [[count, group, None if math.isinf(size) else size]
                      for (count, group), size in sizes.items()]}  # fmt: skip


def load_or_build_priors(
    world: sqlite3.Connection,
    leagues: list[str],
    season: int,
    rules: TimeRules,
    folder: Path,
    identity: str,
) -> dict:
    """The season's priors from `folder`, built and kept there when missing or of another
    release."""
    path = folder / f"thresholds-priors-{season}.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("identity") == identity:
            return stored["priors"]
    priors = build_priors(world, leagues, season, rules)
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"identity": identity, "priors": priors}), encoding="utf-8")
    return priors


class ThresholdsOfTheDay:
    """The player thresholds of the morning's league fixtures."""

    def __init__(self, world: sqlite3.Connection, leagues: list[str], season: int, now: str,
                 rules: TimeRules, forecasts, past_matrices: dict[int, np.ndarray],
                 folder: Path, identity: str, sheets=None):  # fmt: skip
        """`forecasts`: the morning's league fixtures (`FixtureForecast`); `past_matrices`: the
        champion's matrices of the matches played (`vesta.jobs.fixtures.league_forecasts`)."""
        priors = load_or_build_priors(world, leagues, season, rules, folder, identity)
        self.groups = {None if group == "None" else group: GroupRates(
            np.array(values["per_ninety"]), np.array(values["per_match"]),
            values["saves_per_goal"], values["conceded_per_ninety"])
            for group, values in priors["groups"].items()}  # fmt: skip
        self.sizes = {(count, group): math.inf if size is None else size
                      for count, group, size in priors["sizes"]}  # fmt: skip
        matches, histories, _ = _read_season_inputs(world, leagues, season - 1, season, rules, now,
                                                    sheets)  # fmt: skip
        self.histories = histories
        self.known_at = {player: [item.known_at for item in items]
                         for player, items in histories.items()}  # fmt: skip
        self.season = season
        self.now = now
        self.scales = load_season_scales(season)
        self.terms = _build_match_terms(
            world, leagues, season, rules, matches, forecasts, past_matrices, now
        )

    def build_side_distributions(
        self, lineup: TeamLineup, match_id: int
    ) -> list[PublishedDistribution]:
        """The thresholds' distributions of a side's pool (version empty: the gate sets it)."""
        minutes_of = {item.subject_id: np.array(item.values["probabilities"])
                      for item in lineup.distributions if item.quantity_key == "L6.minutes"}
        players = [(player, group, minutes_of[player], self.rate_player(player, group))
                   for player, group, *_ in lineup.pool if player in minutes_of]  # fmt: skip
        return build_threshold_distributions(
            players, self.terms, match_id, lineup.team_id, self.sizes, self.scales
        )

    def rate_player(self, player: int, group: str | None) -> tuple[np.ndarray, float]:
        """His counts per 90 and his saves per goal conceded, from what was known this morning:
        the lineup time reads them with its own minutes."""
        history = self.histories.get(player, [])
        known = history[:bisect.bisect_right(self.known_at.get(player, []), self.now)]
        return estimate_own_rates(known, self.season, self.groups.get(group, self.groups[None]))


def build_threshold_distributions(
    players, terms, match_id: int, team_id: int, sizes: dict, scales: dict[str, float]
) -> list[PublishedDistribution]:
    """The thresholds' distributions of a side (version empty: the gate sets it): `players`,
    each (player, position group, his minutes distribution, his rates of `rates`)."""
    found = []
    for player, group, minutes, (per_ninety, ratio) in players:
        weights, bin_minutes = split_into_minute_bins(minutes)
        means = estimate_player_means(per_ninety, ratio, terms, match_id, team_id, bin_minutes)
        for index, (key, threshold) in enumerate(THRESHOLDS.items()):
            if key not in KEYS or not applies_to_player(threshold.who, group == "goalkeeper"):
                continue
            scale = 1.0
            if key in SCALED:
                if key not in scales:
                    continue  # no scale fitted for the season: not published
                scale = scales[key]
            size = sizes.get((threshold.count, group), sizes.get((threshold.count, None),
                                                                 math.inf))  # fmt: skip
            cells = count_cells(means[index] * scale, weights, size, threshold.cells)
            found.append(PublishedDistribution(key, player, "", _format_values(threshold, cells)))
    return found


def load_season_scales(season: int) -> dict[str, float]:
    """Candidate 2's scales of `season` (none when the file holds none for it)."""
    if not SCALES_FILE.exists():
        return {}
    found = tomllib.loads(SCALES_FILE.read_text(encoding="utf-8")).get(str(season), {})
    return {key: float(value) for key, value in found.items() if key in SCALED}


def _format_values(threshold, cells: np.ndarray) -> dict:
    if threshold.kind == "binary":
        return {"yes": float(1.0 - cells[0])}
    return {"probabilities": [float(value) for value in cells[:-1]],
            "tail_from": len(cells) - 1, "tail": float(cells[-1])}


def _build_match_terms(world, leagues, season, rules, matches, forecasts, past_matrices,
                       now: str) -> MatchTerms:  # fmt: skip
    """The match's terms of the morning's fixtures: each side's expected goals against its faded
    level over the matches played of this season and the one before (the champion's matrices),
    and the weekly fouls and cards fits on the matches known before the fixture's week."""
    terms = MatchTerms()
    ordered = [(match.kickoff_utc, match.match_id, match.home_team_id, match.away_team_id,
                past_matrices[match.match_id]) for match in matches
               if match.match_id in past_matrices]  # fmt: skip
    ordered += [(item.fixture.kickoff_utc, item.fixture.match_id, item.fixture.home_team_id,
                 item.fixture.away_team_id, item.matrix) for item in forecasts]  # fmt: skip
    ordered.sort(key=lambda row: (row[0], row[1]))
    terms.set_attack_factors([(match_id, home, away, *compute_side_means(matrix))
                              for _, match_id, home, away, matrix in ordered])  # fmt: skip
    config = tomllib.loads((ROOT / "config/vesta/team_counts.toml").read_text(encoding="utf-8"))
    fit = config["fit"]
    parameters = FitParameters(int(fit["window_days"]), float(fit["half_life_days"]),
                               float(fit["ridge"]), int(fit["minimum_matches"]), 11)  # fmt: skip
    fixtures = tuple(CountMatch(
        item.fixture.match_id, item.fixture.competition, item.fixture.season,
        item.fixture.home_team_id, item.fixture.away_team_id,
        forecast_time(item.fixture.kickoff_utc, "minute", item.fixture.kickoff_utc[:10], rules),
        find_result_known_at(item.fixture.kickoff_utc, "minute", item.fixture.kickoff_utc[:10],
                             rules),
        None, None, None) for item in forecasts)  # fmt: skip
    for count in ("fouls", "cards"):
        played = [match for match in read_count_matches(world, leagues, season - 3, season,
                                                        count, rules)
                  if match.known_at < now]  # fmt: skip
        terms.fit_count_factors(count, played, parameters, set(), fixtures)
    return terms

