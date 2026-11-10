"""The pre-match publication of result and goals from a dataset in the Vesta data format (VDF S1).

    python -m vesta.jobs.dataset_pre_match <dataset.db> <forecast-log.db> [--hours 168] [--now T]

For a dataset other than our own: the capability
check (`vesta.capabilities`) decides which competitions are served, and every model refits from
the dataset with the released settings, nothing fitted on our data reused. In a league, the
baseline B0 (Dixon-Coles on goals, each competition's matrix calibrated on its own two seasons
before), the keys of its passed verdict (`vesta.capabilities.BASELINE_KEYS`); in a UEFA cup the
dataset holds with its leagues, the cups model (Elo on the scores). Every publication carries
the caveat `not_validated_on_dataset` on each of its kinds. Our own dataset keeps the daily job
(`vesta.jobs.pre_match`) and its released gate.

The dataset's log registers, once, a model version for this code (`vesta-<release>-dataset`,
family members the baseline and the cups) and the keys it serves in the dataset's
competitions, so the log stays self-describing; nothing of it claims a verdict on the dataset.
"""

import argparse
import math
import sqlite3
import sys
import tomllib
from dataclasses import replace
from pathlib import Path

import numpy as np

from vesta.capabilities import (
    BASELINE_KEYS,
    SPINE_KEYS,
    assess_competitions,
    find_latest_release,
    find_present_blocks,
)
from vesta.capabilities import _read_released_competitions as released_of
from vesta.code_version import read_code_version
from vesta.data_format import make_dataset_caveats, open_dataset
from vesta.forecast_log.store import open_log
from vesta.forecast_log.writing import (
    Publication,
    PublishedDistribution,
    publish,
    register_model_version,
    register_published_quantity,
)
from vesta.jobs.fixtures import (
    Fixture,
    FixtureForecast,
    forecast_cup_fixtures,
    format_now_utc,
    read_fixtures,
)
from vesta.quantities import load_quantities
from vesta.releases import _register_or_confirm
from vesta.spine.dixon_coles import DixonColesFamily, DixonColesParameters
from vesta.spine.point_in_time import ForecastRequest, replay
from vesta.spine.publication import build_spine_distributions
from vesta.spine.score_matrix import score_matrices
from vesta.spine.timing import TimeRules
from vesta.spine.walk_forward import (
    WalkForwardSettings,
    _build_events,
    _fit_calibration_or_neutral,
    _keep_forecastable,
)
from vesta.spine.world import read_league_matches

ROOT = Path(__file__).resolve().parents[3]
KINDS_OF = {"M1.result": "results"}  # every other spine key is of the kind goals
BASELINE_VERDICT = ("2026-11-08-b0-serve-development.md; 2026-11-08-b0-serve-blind.md (our five "
              "leagues; no verdict on this dataset)")  # fmt: skip


def forecast_fixtures_with_baseline(
    world: sqlite3.Connection, fixtures: list[Fixture], now: str, spine: dict, leagues: list[str]
) -> list[FixtureForecast]:
    """B0's forecasts of the league fixtures of `leagues`, made at `now`: Dixon-Coles on the
    goals known by then, each competition's matrix calibrated on its own forecasts of the
    `calibration_seasons` before, as its verdict did."""
    chosen = [fixture for fixture in fixtures if fixture.kind == "league"
              and fixture.competition in leagues]  # fmt: skip
    if not chosen:
        return []
    settings = replace(WalkForwardSettings.from_config(spine), competitions=tuple(leagues))
    last_season = max(fixture.season for fixture in chosen)
    matches = [match for match in read_league_matches(
        world, leagues, settings.first_training_season - 1, last_season)
        if match.kickoff_utc < now]  # fmt: skip
    facts, requests = _build_events(matches, TimeRules.from_config(spine), settings)
    facts = [fact for fact in facts if fact.known_at <= now]
    requests += [ForecastRequest(fixture.match_id, fixture.competition, fixture.season,
                                 fixture.home_team_id, fixture.away_team_id, now)
                 for fixture in chosen]  # fmt: skip
    rows = replay([DixonColesFamily(DixonColesParameters.from_config(spine))], facts, requests)

    def pick_intensities(features):
        return (features["dixon_coles_home_intensity"], features["dixon_coles_away_intensity"])

    fixture_ids = {fixture.match_id: fixture for fixture in chosen}
    past = _keep_forecastable([row for row in rows if row.request.match_id not in fixture_ids],
                              matches, pick_intensities)  # fmt: skip
    calibrations = {}
    forecasts = []
    for row in rows:
        fixture = fixture_ids.get(row.request.match_id)
        if fixture is None:
            continue
        home, away = pick_intensities(row.features)
        if not (math.isfinite(home) and math.isfinite(away)):
            continue
        key = (fixture.competition, fixture.season)
        if key not in calibrations:
            first = max(fixture.season - settings.calibration_seasons,
                        settings.first_training_season)
            calibrations[key] = _fit_calibration_or_neutral(
                [item for item in past if item.match.competition == fixture.competition
                 and first <= item.match.season < fixture.season],
                minimum_matches=settings.minimum_calibration_matches,
                fit_spread=False, fit_dispersion=False)  # fmt: skip
        matrix = score_matrices(np.array([home]), np.array([away]), calibrations[key])[0]
        forecasts.append(FixtureForecast(fixture, matrix, home, away, None))
    return forecasts


def register_dataset_version(log: sqlite3.Connection, version: str, published_from: str,
                             served: dict[int, tuple[str, ...]]) -> str:  # fmt: skip
    """The model version of this code on a foreign dataset, and each served (key, competition)
    pair, registered once; returns the version's name."""
    model_version = f"vesta-{version}-dataset"
    definitions = load_quantities()
    known = log.execute("SELECT 1 FROM model_versions WHERE model_version = ?",
                        (model_version,)).fetchone()  # fmt: skip
    if known is None:
        register_model_version(
            log, model_version, "the champion, baseline B0, the cups, the season, the nations"
            " (failed its verdicts)", published_from,
            "forecasts on a dataset no verdict covers: every model run on it as given",
            verdict_id=BASELINE_VERDICT)  # fmt: skip
    present = set(log.execute("SELECT quantity_key, competition_id FROM published_quantities"
                              " WHERE model_version = ?", (model_version,)).fetchall())
    for competition_id, keys in served.items():
        for key in keys:
            _register_or_confirm(log, definitions[key])
            if (key, competition_id) not in present:
                register_published_quantity(log, key, competition_id, model_version,
                                            published_from, False, False)  # fmt: skip
    return model_version


def run(dataset_path: Path, log_path: Path, hours: float, now: str, say=print) -> dict:
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    release = find_latest_release()
    version = tomllib.loads(release.read_text(encoding="utf-8"))["version"]
    world, header = open_dataset(dataset_path)
    if header.ours:
        world.close()
        raise SystemExit("our own dataset is published by vesta.jobs.pre_match and its gate")
    with world:
        present = find_present_blocks(world)
        served = [item for item in assess_competitions(world, header, present, spine,
                                                       released_of(release)) if item.served]
        leagues = [item.code for item in served if item.model == "B0"]
        cups = [item.code for item in served if item.model == "cups Elo"]
        ids = dict(world.execute("SELECT code, competition_id FROM competitions").fetchall())
        fixtures = read_fixtures(world, now, hours, leagues, cups)
        forecasts = (forecast_fixtures_with_baseline(world, fixtures, now, spine, leagues)
                     + forecast_cup_fixtures(world, fixtures, now, spine))  # fmt: skip
    world.close()
    keys_of = {ids[code]: BASELINE_KEYS for code in leagues}
    keys_of.update({ids[code]: SPINE_KEYS for code in cups})
    log = open_log(log_path)
    model_version = register_dataset_version(log, version, now, keys_of)
    commit = read_code_version(ROOT)  # no git: a ZIP download runs too
    counts = {"served_competitions": len(served), "fixtures": len(fixtures), "published": 0}
    for forecast in forecasts:
        fixture = forecast.fixture
        wanted = keys_of[ids[fixture.competition]]
        if _was_published_today(log, fixture.match_id, now[:10]):
            continue
        chosen = [
            PublishedDistribution(
                item.quantity_key, item.subject_id, model_version, item.values, item.window_key
            )
            for item in build_spine_distributions(
                forecast.matrix, fixture.match_id, fixture.home_team_id, fixture.away_team_id, ""
            )
            if item.quantity_key in wanted
        ]
        kinds = tuple(sorted({KINDS_OF.get(item.quantity_key, "goals") for item in chosen}))
        caveats = make_dataset_caveats(header, fixture.competition, set(), kinds, model_version)
        trigger = "first" if _is_first_publication(log, fixture.match_id) else "morning"
        publication = Publication(stage="pre_match", trigger=trigger, published_at=now,
                                  code_commit=commit, match_id=fixture.match_id,
                                  scheduled_kickoff_utc=fixture.kickoff_utc,
                                  world_build=f"{header.dataset} {header.built_at}")  # fmt: skip
        publish(log, publication, chosen, caveats)
        counts["published"] += 1
    log.close()
    say(f"{header.dataset}: {len(served)} competitions served ({', '.join(leagues + cups)}); "
        f"{len(fixtures)} fixtures; {counts['published']} published, each marked "
        "not validated on this dataset")  # fmt: skip
    return counts


def _is_first_publication(log: sqlite3.Connection, match_id: int) -> bool:
    return log.execute("SELECT 1 FROM publications WHERE match_id = ? AND stage = 'pre_match'",
                       (match_id,)).fetchone() is None  # fmt: skip


def _was_published_today(log: sqlite3.Connection, match_id: int, today: str) -> bool:
    return log.execute("SELECT 1 FROM publications WHERE match_id = ? AND stage = 'pre_match'"
                       " AND substr(published_at, 1, 10) = ?", (match_id, today)).fetchone() \
        is not None  # fmt: skip


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("log", type=Path)
    parser.add_argument("--hours", type=float, default=168.0)
    parser.add_argument("--now", default=None)
    options = parser.parse_args(arguments)
    run(options.dataset, options.log, options.hours, options.now or format_now_utc())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

