"""The user's one command: the forecasts of the next round of every competition the dataset
allows, made on the user's PC.

    python -m vesta.run <path> [<path> ...] [--log <forecast-log.db>] [--cache <folder>]
                        [--now <instant>]

Each path is a dataset file or a folder of them (the whole file, or any set of split files).
In order, one `PROGRESS <done> <total> <seconds left|-> <text>` line on stdout per step (the
window's contract, `src/desk/run.py`), everything else on stderr:

1. verify: each file opened alone, its header, its hash and its proof (`vesta.dataset_proof`);
   the code against its release manifest (`vesta.release_gate`);
2. merge: the set as one working copy (`vesta.datasets`);
3. capabilities: what each competition can be served by, and why not (`vesta.capabilities`);
4. fit and forecast: each served competition's round under way at the dataset's `built_at`,
   whole, from the facts at its first kick-off, and its next round, from `built_at`
   (`vesta.rounds`); with each fixture, the families the release publishes beyond result and
   goals where the capability check serves them (`vesta.capabilities.assess_families`): the
   team counts, goals in time, the players and the lineups (`vesta.round_families`), built from
   the same facts;
5. write: the local log beside the data, as the window finds it (`<file>.forecast-log.db` for one
   file, `vesta/forecast-log.db` for a set): one pre-match publication per fixture and round;
6. seasons: each league's season simulated from the same dataset, its released season keys
   (`vesta.season.league_job`), one `season_day` publication per league and dataset; a league
   B0 serves is simulated on B0's matrices, with `not_validated_on_dataset`.

A competition read from files proved ours, under the released code, is served by the released
models with their verdicts and no caveat. Anything else is refitted from the data, and every
publication carries `not_validated_on_dataset`. A fixture whose team plays another match
before it carries `before_previous_match`; one that had kicked off when the dataset was built is
published with the trigger `after_kickoff` and carries `forecast_after_kickoff`, never a forecast
of record.

The released champion reads the shots' `xg` as the dataset gives it (S3a): ours, written by the
weekly build; a foreign dataset's own, with the caveat. A league without it falls back to B0
(`vesta.capabilities`). B0 and the cups model run on any dataset. The national teams are
forecast by the nations model, which failed its verdicts (decided on 9 November
2026): never the release's version, always with `not_validated_on_dataset`.

A family's keys are the release's where the competition is validated and the five leagues its
fit pools over all come from files proved ours; else the dataset's, with the caveat. The
lineups read the bans and the elevens the dataset holds as of its `built_at`.

Exit codes: 0 done, 1 a format or verification error, 2 a failed run.
"""

import argparse
import sqlite3
import sys
import tomllib
from collections import Counter
from pathlib import Path

from vesta.capabilities import (
    BASELINE_KEYS,
    SPINE_KEYS,
    VALIDATED,
    assess_competitions,
    assess_families,
    find_present_blocks,
    has_player_names,
    validate,
)
from vesta.capabilities import _read_released_competitions as released_of
from vesta.code_version import read_code_version
from vesta.data_format import (
    NOT_VALIDATED_CAVEAT,
    FormatError,
    Header,
)
from vesta.dataset_proof import prove
from vesta.datasets import describe, find_dataset_paths, open_datasets
from vesta.forecast_log.store import open_log
from vesta.forecast_log.writing import Caveat, Publication, PublishedDistribution, publish
from vesta.jobs.dataset_pre_match import (
    forecast_fixtures_with_baseline,
    register_dataset_version,
)
from vesta.jobs.fixtures import (
    forecast_cup_fixtures,
    forecast_league_fixtures,
    forecast_nation_fixtures,
)
from vesta.jobs.pre_match import make_caveats, write_expected_elevens
from vesta.lineups.model import load_ban_competitions
from vesta.quantities import load_quantities
from vesta.release_gate import (
    check_manifest,
    find_latest_plan,
    read_release_version,
    register_release_gate,
)
from vesta.round_families import forecast_round_families, make_identity
from vesta.rounds import find_next_rounds
from vesta.season.league_job import (
    read_finished_since,
    read_previous_publication,
    simulate_league_seasons,
)

ROOT = Path(__file__).resolve().parents[2]
BEFORE_PREVIOUS = "before_previous_match"
AFTER_KICKOFF = "after_kickoff"  # the trigger of a match of the round under way (schema 12)
AFTER_KICKOFF_CAVEAT = "forecast_after_kickoff"
KINDS = {key: item.quantity.probability_kind for key, item in load_quantities().items()}
FAMILY_OF = {key: item.family_member for key, item in load_quantities().items()}


class Progress:
    """The window's progress lines: `PROGRESS <done> <total> <seconds left|-> <text>`."""

    def __init__(self, total: int, out=sys.stdout):
        self.done, self.total, self.out = 0, total, out

    def report_step(self, text: str, grow: int = 0) -> None:
        self.total += grow
        self.done += 1
        self.out.write(f"PROGRESS {self.done} {max(self.total, self.done)} - {text}\n")
        self.out.flush()


def _print_note(text: str) -> None:
    print(text, file=sys.stderr)


def run(arguments: list[str | Path], log_path: Path | None = None, cache: Path | None = None,
        now: str | None = None, out=sys.stdout, root: Path = ROOT) -> dict:  # fmt: skip
    spine = tomllib.loads((root / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    expected_goals = tomllib.loads(
        (root / "config/vesta/expected_goals.toml").read_text(encoding="utf-8")
    )
    commit = read_code_version(root)
    paths = find_dataset_paths(arguments)
    base = Path(arguments[0]) if Path(arguments[0]).is_dir() else Path(arguments[0]).parent
    # the window's rule (`desk.reading.find_log`): one file's log beside it, a set's in vesta/
    log_path = log_path or (paths[0].with_name(paths[0].name + ".forecast-log.db")
                            if len(paths) == 1 else base / "vesta" / "forecast-log.db")
    cache = cache or base / "vesta" / "cache"
    progress = Progress(len(paths) + 5, out)

    files, proofs = [], {}
    for path in paths:
        item = describe(path)
        files.append(item)
        proofs[item.path] = prove(item)
        progress.report_step(f"verified {path.name}: {proofs[item.path].reason}")
    manifest, why = check_manifest(root)
    _print_note(f"Vesta {read_code_version(root)}: {why}")

    connection, _ = open_datasets(paths, cache, files)
    built_at = min(item.header.built_at for item in files)
    datasets = {item.header.dataset for item in files}
    header = Header(files[0].header.version, datasets.pop() if len(datasets) == 1 else "mixed",
                    built_at)  # fmt: skip
    now = now or built_at
    progress.report_step(f"{len(files)} file(s) read as one dataset, as of {built_at}")

    proved = set()
    if manifest == "release":
        codes = set().union(*(item.codes for item in files))
        proved = {code for code in codes if all(proofs[item.path].ours for item in files
                                                if code in item.codes)}  # fmt: skip
    present = find_present_blocks(connection)
    errors, _ = validate(connection, present)
    if errors:
        connection.close()
        raise FormatError("; ".join(str(error) for error in errors[:5]))
    found = assess_competitions(
        connection, header, present, spine, released_of(find_latest_plan(root)), proved
    )
    families = assess_families(found, present, spine, find_latest_plan(root), built_at,
                               has_player_names(connection), proved)  # fmt: skip
    served = [item for item in found if item.served]
    champion_leagues = [item.code for item in served if item.model == "champion"]
    baseline_leagues = [item.code for item in served if item.model == "B0"]
    cups = [item.code for item in served if item.model == "cups Elo"]
    nations = [item.code for item in served if item.model == "nations Elo"]
    validation = {item.code: item.validation for item in served}
    progress.report_step(
        f"{len(served)} competitions served, {len(found) - len(served)} not served"
    )
    reasons: dict[str, list[str]] = {}
    for item in found:
        if not item.served:
            reasons.setdefault(item.reason, []).append(item.code)
    for reason, codes in sorted(reasons.items(), key=lambda pair: -len(pair[1])):
        shown = ", ".join(codes[:8]) + (f" and {len(codes) - 8} more" if len(codes) > 8 else "")
        _print_note(f"not served ({reason}): {shown}")
    family_reasons: dict[str, list[str]] = {}
    for entry in families:
        if not entry.served:
            family_reasons.setdefault(f"{entry.family}: {entry.reason}", []).append(entry.code)
    for reason, codes in family_reasons.items():
        _print_note(f"not served ({reason}): {', '.join(codes)}")
    rounds = find_next_rounds(connection, now, champion_leagues + baseline_leagues, cups,
                              nations)  # fmt: skip
    for key, reason in rounds.waiting.items():
        _print_note(f"{key}: waits: {reason}")
    before = {item.fixture.match_id for item in rounds.fixtures if item.before_previous_match}
    after_kickoff = {item.fixture.match_id for item in rounds.fixtures if item.after_kickoff}
    ids = dict(connection.execute("SELECT code, competition_id FROM competitions").fetchall())
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log = open_log(log_path)
    released_version, gate_pairs = (register_release_gate(log, connection, now, root) if proved
                                     else (None, {}))  # fmt: skip
    season_keys, season_leagues = read_season_part(root)
    season_leagues = [code for code in champion_leagues + baseline_leagues
                      if code in season_leagues]  # fmt: skip
    own = {ids[code]: BASELINE_KEYS for code in baseline_leagues}
    own.update({ids[code]: SPINE_KEYS for code in champion_leagues + cups + nations
                if validation[code] != VALIDATED})  # fmt: skip
    for code in season_leagues:
        if validation[code] != VALIDATED or code in baseline_leagues:
            own[ids[code]] = (*own.get(ids[code], ()), *season_keys)
    for entry in families:
        if entry.served and entry.validation != VALIDATED:
            own[ids[entry.code]] = (*own.get(ids[entry.code], ()), *entry.keys)
    dataset_version = register_dataset_version(log, read_release_version(root), now, own)
    versions = choose_versions(served, families, ids, gate_pairs, dataset_version)

    forecasts, timing = [], {}
    identity = make_identity(root, set().union(*(item.codes for item in files)))
    for as_of in sorted({item.as_of for item in rounds.fixtures}):  # each from its own facts
        fixtures = [item.fixture for item in rounds.fixtures if item.as_of == as_of]
        made = (forecast_fixtures_with_baseline(connection, fixtures, as_of, spine,
                                                baseline_leagues)
                + forecast_cup_fixtures(connection, fixtures, as_of, spine)
                + forecast_nations_or_none(connection, fixtures, as_of, spine))  # fmt: skip
        led = [item for item in fixtures if item.competition in champion_leagues]
        rated, past_matrices = {}, {}
        if led:  # the champion on the shots' xg, its past kept in the cache between runs
            champion, past_matrices = forecast_league_fixtures(
                connection, led, as_of, spine, expected_goals, kept_in=cache, release=commit,
                rated=rated)  # fmt: skip
            made += champion
        built = forecast_round_families(connection, made, as_of, versions, ids, spine,
                                        expected_goals, cache, identity, rated, past_matrices,
                                        timing)  # fmt: skip
        forecasts += [(forecast, built) for forecast in made]
        for family, why in built.failed.items():
            _print_note(f"{family} left out of the forecasts made at {as_of}: {why}")
        for competition, team, matches in built.held:
            _print_note(f"{competition}: the scorers of team {team} wait for the shots of its"
                        f" last league matches {', '.join(map(str, matches))}")  # fmt: skip
    round_count = len({item.round_key for item in rounds.fixtures})
    progress.report_step(f"{len(forecasts)} fixtures forecast in {round_count} rounds;"
                         f" {len(rounds.waiting)} rounds wait for a later dataset")  # fmt: skip

    published, by_family = 0, Counter()
    for forecast, built in forecasts:
        fixture = forecast.fixture
        if log.execute("SELECT 1 FROM publications WHERE match_id = ? AND stage = 'pre_match'",
                       (fixture.match_id,)).fetchone():  # fmt: skip
            continue  # its round's forecast stays as it was
        chosen = built.build(forecast, versions, ids[fixture.competition])
        caveats = make_round_caveats(chosen, dataset_version, fixture.match_id in before,
                                     fixture.match_id in after_kickoff)  # fmt: skip
        caveats += make_caveats(
            chosen, KINDS, built.bans_read and fixture.competition in load_ban_competitions(),
            bool({fixture.home_team_id, fixture.away_team_id} & built.stale))  # fmt: skip
        trigger = AFTER_KICKOFF if fixture.match_id in after_kickoff else "first"
        publication_id = publish(
            log, Publication(stage="pre_match", trigger=trigger, published_at=now,
                             code_commit=commit, match_id=fixture.match_id,
                             scheduled_kickoff_utc=fixture.kickoff_utc,
                             world_build=f"{header.dataset} {header.built_at}"),
            chosen, caveats)  # fmt: skip
        write_expected_elevens(log, publication_id, chosen, built.sides.get(fixture.match_id, ()))
        published += 1
        by_family.update({FAMILY_OF[item.quantity_key] for item in chosen})
    progress.report_step(f"{published} forecasts written to {log_path}: "
                         + ", ".join(f"{family} {count}" for family, count
                                     in sorted(by_family.items())))  # fmt: skip

    seasons = publish_seasons(connection, log, season_leagues, season_keys, ids, validation,
                              gate_pairs, dataset_version, now, spine, expected_goals, cache,
                              commit, f"{header.dataset} {header.built_at}",
                              tuple(code for code in season_leagues
                                    if code in baseline_leagues))  # fmt: skip
    connection.close()
    log.close()
    progress.report_step(f"{seasons} league seasons simulated and written")
    return {"files": len(files), "proved": sorted(proved),
            "served": len(served), "champion": sorted(champion_leagues),
            "fixtures": len(forecasts), "published": published, "seasons": seasons,
            "families": dict(by_family),
            "waiting": sorted(rounds.waiting), "released_version": released_version}


def forecast_nations_or_none(connection: sqlite3.Connection, fixtures: list, as_of: str,
                             spine: dict) -> list:  # fmt: skip
    """The nations model's forecasts of the national fixtures; none, with a note, when the
    dataset holds too few national matches before `as_of` to fit it."""
    try:
        return forecast_nation_fixtures(connection, fixtures, as_of, spine)
    except ValueError as error:
        _print_note(f"national teams not forecast from {as_of}: {error}")
        return []


def choose_versions(served: list, families: list, ids: dict, gate_pairs: dict,
                    dataset_version: str) -> dict[tuple[str, int], str]:  # fmt: skip
    """(key, competition id) -> the model version publishing it: the release's where the
    competition (or the family) is validated, else the dataset's. B0 is never the release's."""
    versions = {}
    for item in served:
        released = item.validation == VALIDATED and item.model != "B0"
        for key in BASELINE_KEYS if item.model == "B0" else SPINE_KEYS:
            pair = (key, ids[item.code])
            versions[pair] = (gate_pairs.get(pair) if released else None) or dataset_version
    for entry in (entry for entry in families if entry.served):
        for key in entry.keys:
            pair = (key, ids[entry.code])
            versions[pair] = (gate_pairs.get(pair) if entry.validation == VALIDATED
                              else None) or dataset_version  # fmt: skip
    return versions


def make_round_caveats(chosen: list[PublishedDistribution], dataset_version: str,
                       before_previous: bool, after_kickoff: bool) -> list[Caveat]:  # fmt: skip
    """The caveats of the user's run on each kind of a publication: `not_validated_on_dataset`
    on each kind of a distribution of the dataset's version; on every kind, a match made before
    a team's previous one, or after its kick-off."""
    found = {Caveat(KINDS[item.quantity_key], NOT_VALIDATED_CAVEAT, item.model_version)
             for item in chosen if item.model_version == dataset_version}  # fmt: skip
    first: dict[str, str] = {}
    for item in chosen:
        first.setdefault(KINDS[item.quantity_key], item.model_version)
    for wanted, code in ((before_previous, BEFORE_PREVIOUS),
                         (after_kickoff, AFTER_KICKOFF_CAVEAT)):  # fmt: skip
        if wanted:
            found |= {Caveat(kind, code, version) for kind, version in first.items()}
    return sorted(found, key=lambda caveat: (caveat.probability_kind, caveat.code,
                                             caveat.model_version))  # fmt: skip


def read_season_part(root: Path) -> tuple[tuple[str, ...], list[str]]:
    """The released season keys and their leagues (the latest plan's `season` part)."""
    plan = tomllib.loads(find_latest_plan(root).read_text(encoding="utf-8"))
    for part in plan["part"]:
        if part["family_member"] == "season":
            return tuple(part["keys"]), list(part["competitions"])
    return (), []


def publish_seasons(connection, log, leagues: list[str], keys: tuple[str, ...], ids: dict,
                    validation: dict, gate_pairs: dict, dataset_version: str, now: str,
                    spine: dict, expected_goals: dict, cache: Path, commit: str,
                    world_build: str, baseline_leagues: tuple[str, ...] = ()) -> int:  # fmt: skip
    """The season of each league the release's season part covers, simulated at `now` from the
    same dataset as the match forecasts, as the season job does (`vesta.season.league_job`):
    released versions on files proved ours, the dataset's version with
    `not_validated_on_dataset` otherwise. A league B0 serves (`baseline_leagues`: no shots' xg,
    as one league's file without it) is simulated on B0's matrices, never as the released
    season model: the dataset's version, with the caveat. A season already written from this
    dataset is left as it is. Returns the seasons written."""
    if not leagues or not keys:
        return 0
    allowed = {}
    for code in leagues:
        for key in keys:
            allowed[(key, ids[code])] = (gate_pairs.get((key, ids[code]))
                                         if validation[code] == VALIDATED
                                         and code not in baseline_leagues
                                         else None) or dataset_version  # fmt: skip
    previous = {}

    def is_written(season_id: int) -> bool:
        previous[season_id] = read_previous_publication(log, season_id)
        return previous[season_id] is not None and previous[season_id] >= now

    _, days = simulate_league_seasons(connection, leagues, now, spine, expected_goals, allowed,
                                      cache, commit, is_written, _print_note,
                                      baseline_leagues=baseline_leagues)  # fmt: skip
    for day in days:
        before = previous.get(day.season_id)
        version = day.distributions[0].model_version if day.distributions else dataset_version
        caveats = ([] if validation[day.competition] == VALIDATED
                   and day.competition not in baseline_leagues
                   else [Caveat("seasons", NOT_VALIDATED_CAVEAT, version)])  # fmt: skip
        publish(log, Publication(stage="season_day", trigger="morning" if before else "first",
                                 published_at=now, code_commit=commit, season_id=day.season_id,
                                 moved_by=read_finished_since(connection, day.finished, before)
                                 if before else [], world_build=world_build,
                                 tie_break_carried_forward=day.carried_forward),
                day.distributions, caveats)  # fmt: skip
    return len(days)


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--now", default=None)
    options = parser.parse_args(arguments)
    try:
        run(options.paths, options.log, options.cache, options.now)
    except (FormatError, FileNotFoundError, sqlite3.DatabaseError) as error:
        _print_note(f"error: {error}")
        return 1
    except Exception as error:  # noqa: BLE001 - the window needs an exit code, the text on stderr
        _print_note(f"failed: {type(error).__name__}: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
