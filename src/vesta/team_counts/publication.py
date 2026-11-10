"""What team counts publish before a match: the fits of this week and the season's corrections,
then each fixture's distributions, in the leagues and in the UEFA cups.

The fit of a league (or the cups' pooled fit) is the one of the week holding `now` (the matches
known before the week starts), as in the verdict. The corrections are fitted on the walk-forward
forecasts of the two seasons before the current one whose outcomes were known before the season's
first forecast, never on it: once a season (`season_corrections`, kept by `cached_corrections`),
or, without them, every morning as 2.2.0 did.

Each count publishes from the candidate its verdicts passed with (`CANDIDATES`):
the verdicts of 29 October 2026 for 2.2.0's corners and cards in the leagues, those of
31 October 2026 for family A and the cups.
"""

import json
import math
import sqlite3
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from vesta.forecast_log.writing import PublishedDistribution
from vesta.spine.dixon_coles import find_week_start
from vesta.spine.timing import TimeRules
from vesta.team_counts import cups
from vesta.team_counts.model import (
    KEYS,
    Calibration,
    FitParameters,
    LeagueFit,
    count_distributions,
    fit_calibration,
    fit_candidate_two,
    fit_week,
    read_count_matches,
    seconds_of,
    walk_forward,
)

ROOT = Path(__file__).resolve().parents[3]
# per scope, each count's candidate: 1 the sides' slopes and the tilt (2.2.0), 2 also the
# total's slope and the negative binomial's size per league (one for the cups), then the tilt
CANDIDATES = {
    "leagues": {"corners": 1, "cards": 1, "fouls": 1, "offsides": 1, "shots": 2,
                "shots_on_target": 2},
    "cups": {"cards": 1, "offsides": 1, "shots": 2, "shots_on_target": 2, "fouls": 2,
             "corners": 2},
}  # fmt: skip
QUALIFYING_CODES = ("CLQ", "ELQ", "ECLQ")  # the cups' qualifying competitions: group Q
# a season's corrections read the matches from this many seasons before it: the two seasons they
# forecast, and the fits' window (730 days) behind the first of them, which reaches the end of the
# season three before it (a cup season ends as late as August: CL 2019 in August 2020)
SEASONS_READ = 5
# the fits of a week of the season read from this many seasons before it, for the same reason
SEASONS_READ_WEEK = 3


@dataclass(frozen=True)
class SeasonCorrections:
    """A scope's corrections for one season: per count its calibration, and with candidate 2
    the negative binomial's sizes per league (or "cups"), which replace the weekly fits'."""

    scope: str
    season: int
    calibrations: dict[str, Calibration]
    sizes: dict[str, dict[str, float]] = field(default_factory=dict)

    def look_up_size(self, count: str, group: str, weekly: float) -> float:
        return self.sizes.get(count, {}).get(group, weekly)


def _load_config() -> dict:
    return tomllib.loads((ROOT / "config/vesta/team_counts.toml").read_text(encoding="utf-8"))


def _read(world: sqlite3.Connection, scope: str, count: str, season: int, spine: dict,
          leagues: list[str]) -> list:  # fmt: skip
    if scope == "leagues":
        return read_count_matches(world, leagues, season - SEASONS_READ, season, count,
                                  TimeRules.from_config(spine))  # fmt: skip
    return cups.read_cup_count_matches(world, spine, season - SEASONS_READ, season, count)


def fit_season_corrections(world: sqlite3.Connection, scope: str, season: int, now: str,
                           spine: dict, counts: tuple[str, ...]) -> SeasonCorrections:  # fmt: skip
    """The verdicts' corrections (`counts_verdict.corrections`) for `season`: on the walk-forward
    forecasts of the two seasons before whose outcomes were known before the season's first
    forecast (its first match played, or `now` before it)."""
    config = _load_config()
    leagues = list(spine["walk_forward"]["competitions"])
    calibrations, sizes = {}, {}
    for count in counts:
        parameters = FitParameters.from_config(config, count)
        matches = _read(world, scope, count, season, spine, leagues)
        start = min([seconds_of(match.forecast_time) for match in matches
                     if match.season == season and (scope == "leagues" or match.is_cup)]
                    + [seconds_of(now)])  # fmt: skip
        known = [match for match in matches if seconds_of(match.known_at) < start]
        walk = walk_forward if scope == "leagues" else cups.walk_forward
        forecasts = walk(known, parameters, {season - 2, season - 1})
        if CANDIDATES[scope][count] == 2:
            calibrations[count], sizes[count] = fit_candidate_two(
                forecasts, parameters.size,
                lambda match: match.competition if scope == "leagues" else "cups")  # fmt: skip
        else:
            calibrations[count] = fit_calibration(forecasts, parameters.size)
    return SeasonCorrections(scope, season, calibrations, sizes)


def load_or_build_corrections(world: sqlite3.Connection, scope: str, season: int, now: str,
                              spine: dict, counts: tuple[str, ...], folder: Path,
                              identity: str) -> SeasonCorrections:  # fmt: skip
    """The season's corrections from `folder`, built and kept there when missing, of another
    release or of other counts."""
    path = folder / f"team-counts-corrections-{scope}-{season}.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("identity") == identity and sorted(stored["calibrations"]) == sorted(counts):
            return SeasonCorrections(
                scope, season,
                {count: Calibration(tuple(values["home"]), tuple(values["away"]), values["tilt"],
                                    tuple(values["total"]))
                 for count, values in stored["calibrations"].items()},
                {count: {group: math.inf if value is None else value
                         for group, value in groups.items()}
                 for count, groups in stored["sizes"].items()})  # fmt: skip
    corrections = fit_season_corrections(world, scope, season, now, spine, counts)
    folder.mkdir(parents=True, exist_ok=True)
    kept = {"identity": identity,
            "calibrations": {count: {"home": list(value.home), "away": list(value.away),
                                     "tilt": value.tilt, "total": list(value.total)}
                             for count, value in corrections.calibrations.items()},
            "sizes": {count: {group: None if math.isinf(value) else value
                              for group, value in groups.items()}
                      for count, groups in corrections.sizes.items()}}  # fmt: skip
    written = path.with_suffix(".json.new")
    written.write_text(json.dumps(kept), encoding="utf-8")
    written.replace(path)
    return corrections


@dataclass(frozen=True)
class CountsOfTheDay:
    """Per count: its parameters, its correction, and each league's fit of the week; with
    `corrections`, the season's in place of `calibrations`."""

    parameters: dict[str, FitParameters]
    calibrations: dict[str, Calibration]
    fits: dict[str, dict[str, LeagueFit | None]]
    corrections: SeasonCorrections | None = None

    def build_fixture_distributions(self, competition: str, match_id: int, home_team_id: int,
                                    away_team_id: int) -> list[PublishedDistribution]:  # fmt: skip
        """Every team-count distribution of a fixture (version empty: the gate sets it); none
        for a league without a fit."""
        found = []
        for count, fits in self.fits.items():
            fit = fits.get(competition)
            if fit is None:
                continue
            if self.corrections is None:
                calibration, size_nb = self.calibrations[count], fit.size_nb
            else:
                calibration = self.corrections.calibrations[count]
                size_nb = self.corrections.look_up_size(count, competition, fit.size_nb)
            means = calibration.predict_means(fit.predict_means(home_team_id, away_team_id))
            found += count_distributions(count, match_id, home_team_id, away_team_id, means,
                                         size_nb, self.parameters[count].size, "",
                                         calibration.tilt)  # fmt: skip
        return found


def fit_counts_of_the_day(
    world: sqlite3.Connection,
    leagues: list[str],
    season: int,
    now: str,
    spine: dict,
    counts: tuple[str, ...] = tuple(KEYS),
    corrections: SeasonCorrections | None = None,
) -> CountsOfTheDay:
    """The leagues' fits of the week of `now` for `counts`; without `corrections`, each count's
    calibration refitted this morning on the two seasons before (2.2.0)."""
    config = _load_config()
    week = find_week_start(seconds_of(now))
    parameters, calibrations, fits = {}, {}, {}
    for count in counts:
        parameters[count] = FitParameters.from_config(config, count)
        first = season - (SEASONS_READ if corrections is None else SEASONS_READ_WEEK)
        matches = read_count_matches(world, leagues, first, season, count,
                                     TimeRules.from_config(spine))  # fmt: skip
        known = [match for match in matches
                 if match.home_count is not None and seconds_of(match.known_at) < week]  # fmt: skip
        if corrections is None:
            calibrations[count] = fit_calibration(
                walk_forward(known, parameters[count], {season - 2, season - 1}),
                parameters[count].size)  # fmt: skip
        fits[count] = {league: fit_week([match for match in known if match.competition == league],
                                        week, parameters[count])
                       for league in leagues}  # fmt: skip
    return CountsOfTheDay(parameters, calibrations, fits, corrections)


@dataclass(frozen=True)
class CupCountsOfTheDay:
    """Per count: its parameters, the cups' pooled fit of the week and the season's corrections."""

    parameters: dict[str, FitParameters]
    fits: dict[str, cups.PooledFit | None]
    corrections: SeasonCorrections

    def build_fixture_distributions(self, match_id: int, competition: str, home_team_id: int,
                                    away_team_id: int, *, neutral: bool, home_elo: float,
                                    away_elo: float) -> list[PublishedDistribution]:  # fmt: skip
        """Every team-count distribution of a cup fixture (version empty: the gate sets it).
        `home_elo` and `away_elo` are the spine's cups Elo before the kick-off, the cup forecast's
        own pair; a qualifying competition's fixture takes the qualifiers' level."""
        group = cups.QUALIFYING if competition in QUALIFYING_CODES else competition
        fixture = cups.CupCountMatch(match_id, competition, group, True, self.corrections.season,
                                     home_team_id, away_team_id, "", "", neutral, home_elo,
                                     away_elo, None, None, None)  # fmt: skip
        found = []
        for count, fit in self.fits.items():
            if fit is None:
                continue
            calibration = self.corrections.calibrations[count]
            size_nb = self.corrections.look_up_size(count, "cups", fit.size_nb)
            means = calibration.predict_means(fit.predict_means(fixture))
            found += count_distributions(count, match_id, home_team_id, away_team_id, means,
                                         size_nb, self.parameters[count].size, "",
                                         calibration.tilt)  # fmt: skip
        return found


def fit_cup_counts_of_the_day(world: sqlite3.Connection, season: int, now: str, spine: dict,
                              corrections: SeasonCorrections) -> CupCountsOfTheDay:  # fmt: skip
    """The cups' pooled fit of the week of `now` for every count of `corrections`: on the
    leagues' and the cups' matches with the feed's counts known before the week."""
    config = _load_config()
    week = find_week_start(seconds_of(now))
    parameters, fits = {}, {}
    for count in corrections.calibrations:
        parameters[count] = FitParameters.from_config(config, count)
        matches = cups.read_cup_count_matches(world, spine, season - SEASONS_READ_WEEK, season,
                                              count)  # fmt: skip
        fits[count] = cups.fit_week(matches, week, parameters[count])
    return CupCountsOfTheDay(parameters, fits, corrections)
