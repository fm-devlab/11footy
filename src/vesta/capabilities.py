"""The capability check (VDF S1): is a dataset in the Vesta data format, which of result and
goals (M1-M5) can Vesta serve from it, per competition, and why not the others; and, for each
competition served, which families of the release beyond them (`FAMILIES`: the team counts, goals
in time, the players, the lineups) `vesta.run` publishes, and why not.

    PYTHONPATH=src python -m vesta.capabilities <dataset.db> [--release <release plan .toml>]

Read-only. It prints a table and writes `<dataset>.capabilities.json` beside the dataset. It
covers the header, the required block and B7 (cup structure); the season keys and the live
keys are not in it.
"""

import json
import re
import sqlite3
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from vesta.data_format import BLOCKS, KEYS, REQUIRED, VALUES, Header, has_column, open_dataset

ROOT = Path(__file__).resolve().parents[2]
SPINE_KEYS = ("M1.result", "M2.score", "M3.goals", "M4.both_score", "M5.goals.team")
RESULTS = ("played", "awarded")
KICKOFF = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
# B0 serves a league's result and goals where the champion cannot run: passed development and
# blind on 8 November 2026, every
# key but M4, worse than the frequency reference in the blind seasons
BASELINE_PASSED = True
BASELINE_KEYS = ("M1.result", "M2.score", "M3.goals", "M5.goals.team")

FULL_MODEL = ("the full model, with the shots' expected goals: the result, the score, the goals"
              " and both teams to score")
CUPS_MODEL = "the cups model, which rates every club by its results: the result, score and goals"
# The nations model serves the national teams though it failed its verdicts (decided
# on 9 November 2026): every publication of it is marked not validated
NATIONS_MODEL = ("the nations model, which rates every national team by its results (World"
                 " Football Elo): the result, the score, the goals and both teams to score. It did"
                 " not pass Vesta's tests: on the national matches of 2012 to 2021 its log loss"
                 " was 0.8969, worse than the 0.8870 of the model it had to match, and in the"
                 " blind test of 2024 and 2025 its chances were less well calibrated than past"
                 " frequencies (a calibration error of 0.0112 against 0.0039). Its forecasts are"
                 " published marked not validated")  # fmt: skip
NO_NATIONS_FIXTURE = ("no match of this competition is left to play in the dataset: there is"
                      " nothing to forecast")  # fmt: skip
NATION_KINDS = ("nations qualifying", "nations league", "nations tournament", "friendly")
VALIDATED = "validated"
NOT_VALIDATED = "not validated on this dataset"
FAILED = "did not pass Vesta's tests"


@dataclass
class Problem:
    table: str
    what: str
    rows: int
    examples: list = field(default_factory=list)

    def __str__(self) -> str:
        said = f"table {self.table}: {self.what}"
        if self.rows:
            said += f" ({self.rows} rows"
            said += f", for example {self.examples[0]})" if self.examples else ")"
        return said


@dataclass
class Competition:
    code: str
    kind: str
    season: int | None
    model: str | None  # champion, B0, cups Elo; None when not served
    served: bool
    reason: str
    validation: str | None


@dataclass
class Report:
    path: str
    header: dict
    blocks: dict[str, bool]
    errors: list[Problem]
    warnings: list[Problem]
    competitions: list[Competition]
    families: list["FamilyCoverage"] = field(default_factory=list)


def _read_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


def find_present_blocks(connection: sqlite3.Connection) -> dict[str, bool]:
    """A block is present when every one of its tables holds every one of its columns."""
    present = {}
    for block, tables in BLOCKS.items():
        present[block] = all(set(names) <= _read_columns(connection, table)
                             for table, names in tables.items())  # fmt: skip
    return present


def validate(connection: sqlite3.Connection, present: dict[str, bool]) -> tuple[list, list]:
    """(errors, warnings) of the format on the required block and B7."""
    errors: list[Problem] = []
    warnings: list[Problem] = []
    for table, names in BLOCKS[REQUIRED].items():
        missing = sorted(set(names) - _read_columns(connection, table))
        if missing:
            errors.append(Problem(table, f"missing columns {missing}", 0))
    if errors:
        return errors, warnings
    checked = [REQUIRED] + (["B7"] if present.get("B7") else [])
    tables = {table for block in checked for table in BLOCKS[block]}

    def count(table: str, what: str, condition: str, into: list, limit: int = 5) -> None:
        rows = connection.execute(f"SELECT * FROM {table} WHERE {condition} LIMIT {limit}")
        examples = [list(row) for row in rows]
        if examples:
            total = connection.execute(f"SELECT count(*) FROM {table} WHERE {condition}")
            into.append(Problem(table, what, total.fetchone()[0], examples))

    for table in sorted(tables & set(KEYS)):
        key = ", ".join(KEYS[table])
        duplicated = connection.execute(
            f"SELECT count(*) FROM (SELECT 1 FROM {table} GROUP BY {key} HAVING count(*) > 1)"
        ).fetchone()[0]
        if duplicated:
            errors.append(Problem(table, f"key ({key}) not unique", duplicated))
    for (table, column), allowed in VALUES.items():
        if table not in tables or column not in _read_columns(connection, table):
            continue
        listed = [value for value in allowed if value is not None]
        marks = ", ".join(repr(value) for value in listed)
        nulls = "" if None in allowed else f" OR {column} IS NULL"
        count(table, f"{column} outside the format's values", f"{column} NOT IN ({marks}){nulls}",
              errors)  # fmt: skip
    references = [("seasons", "competition_id", "competitions", "competition_id"),
                  ("matches", "season_id", "seasons", "season_id"),
                  ("matches", "home_team_id", "teams", "team_id"),
                  ("matches", "away_team_id", "teams", "team_id")]  # fmt: skip
    if "B7" in checked:
        references += [("matches", "venue_id", "venues", "venue_id"),
                       ("matches", "tie_id", "match_ties", "tie_id"),
                       ("match_ties", "season_id", "seasons", "season_id")]  # fmt: skip
    for table, column, target, key in references:
        count(
            table,
            f"{column} not found in {target}",
            f"{column} IS NOT NULL AND {column} NOT IN (SELECT {key} FROM {target})",
            errors,
        )
    count(
        "matches",
        "a result without both goals",
        f"status IN {RESULTS} AND (home_goals IS NULL OR away_goals IS NULL)",
        errors,
    )
    kickoffs = connection.execute("SELECT match_id, kickoff_utc FROM matches")
    bad_kickoffs = [row[0] for row in kickoffs if not KICKOFF.match(row[1] or "")]
    if bad_kickoffs:
        errors.append(Problem("matches", "kickoff_utc not YYYY-MM-DDTHH:MM:SSZ",
                              len(bad_kickoffs), bad_kickoffs[:5]))  # fmt: skip
    if "B7" in checked:
        count("matches", "extra time played without the 90-minute score",
              "extra_time_played = 1 AND status IN ('played', 'awarded')"
              " AND (home_goals_regular_time IS NULL OR away_goals_regular_time IS NULL)",
              warnings)  # fmt: skip
    return errors, warnings


def _read_released_competitions(release: Path) -> dict[str, str]:
    """competition code -> the family member releasing M1-M5 there."""
    plan = tomllib.loads(release.read_text(encoding="utf-8"))
    released = {}
    for part in plan["part"]:
        keys = set(part.get("keys", SPINE_KEYS))
        if part["family_member"] == "spine" and set(SPINE_KEYS) <= keys:
            for code in part["competitions"]:
                released[code] = "spine"
    return released


def _count_results(connection, competition_id: int, first: int, last: int) -> int:
    return connection.execute(
        "SELECT count(*) FROM matches m JOIN seasons s USING (season_id) WHERE s.competition_id = ?"
        " AND s.season BETWEEN ? AND ? AND m.status IN ('played', 'awarded')"
        " AND m.home_goals IS NOT NULL", (competition_id, first, last)).fetchone()[0]  # fmt: skip


def _count_rated_matches(connection, competition_id: int, season: int) -> int:
    """The matches of the competition's season and the one before with shots carrying `xg`
    (a season just begun has none played yet)."""
    return connection.execute(
        "SELECT count(DISTINCT m.match_id) FROM shots sh JOIN matches m USING (match_id)"
        " JOIN seasons s USING (season_id) WHERE s.competition_id = ? AND s.season BETWEEN ? AND ?"
        " AND sh.xg IS NOT NULL", (competition_id, season - 1, season)).fetchone()[0]  # fmt: skip


def _explain_basic_model(released: bool, present: dict[str, bool]) -> str:
    """Why a league gets the basic model (B0) and what it forecasts, in plain words."""
    if not released:
        why = "the full model is released for the five top leagues only"
    elif not present["B3"]:
        why = "the dataset holds no shots"
    elif not present["B5"]:
        why = "the dataset holds no lineups"
    else:
        why = "the dataset's shots carry no expected goals (xG)"
    return (f"the basic model, as {why}: the result, the score and the goals; not both teams to"
            " score, where the basic model did no better than past frequencies")


def assess_competitions(connection: sqlite3.Connection, header: Header, present: dict[str, bool],
                        spine: dict, released: dict[str, str],
                        proved: set[str] | None = None) -> list[Competition]:  # fmt: skip
    """M1-M5 per competition of the dataset, for its latest season. `proved`: with a set of files
    (`vesta.datasets`), the competitions read from files proved ours (`vesta.dataset_proof`);
    without, the header decides for every competition, as for one file."""
    walk = spine["walk_forward"]

    def is_ours(code: str) -> bool:
        return header.ours if proved is None else code in proved

    cups = spine["cups"]
    found = []
    rows = connection.execute(
        "SELECT c.competition_id, c.code, c.kind, MAX(s.season) FROM competitions c"
        " LEFT JOIN seasons s USING (competition_id) GROUP BY c.competition_id ORDER BY c.code"
    ).fetchall()
    for competition_id, code, kind, season in rows:

        def make_entry(model, served, reason, validation=None, code=code, kind=kind, season=season):
            return Competition(code, kind, season, model, served, reason, validation)

        if season is None:
            found.append(make_entry(None, False, "no season"))
            continue
        if kind == "league":
            first = season - int(walk["calibration_seasons"])
            before = _count_results(connection, competition_id, first, season - 1)
            minimum = int(walk["minimum_calibration_matches"])
            if before < minimum:
                found.append(make_entry(None, False, f"too little history: {before} results in"
                                        f" the {walk['calibration_seasons']} seasons before"
                                        f" {season}, and Vesta needs {minimum}"))  # fmt: skip
            elif (code in released and present["B3"] and present["B5"]) and (
                    _count_rated_matches(connection, competition_id, season)):
                # the rule of 9 November: a dataset's xg is read as given; validated only
                # on files proved ours
                found.append(make_entry("champion", True, FULL_MODEL,
                                        VALIDATED if is_ours(code) else NOT_VALIDATED))
            elif not BASELINE_PASSED:
                found.append(
                    make_entry(
                        "B0",
                        False,
                        "the basic model has not passed its tests yet",
                    )
                )
            else:
                found.append(
                    make_entry(
                        "B0",
                        True,
                        _explain_basic_model(code in released, present),
                        VALIDATED
                        if is_ours(code) and code in walk["competitions"]
                        else NOT_VALIDATED,
                    )
                )
        elif kind in ("cup", "cup qualifying") and code in cups["cups"]:
            if not present["B7"]:
                found.append(make_entry(None, False, "needs the cup structure (B7)"))
                continue
            leagues = [row[0] for row in connection.execute(
                f"SELECT code FROM competitions WHERE code IN "
                f"({', '.join('?' for _ in cups['leagues'])})", cups["leagues"])]  # fmt: skip
            training = _count_results(connection, competition_id, int(cups["first_cup_season"]),
                                      season - 1)  # fmt: skip
            if sorted(leagues) != sorted(cups["leagues"]):
                missing = sorted(set(cups["leagues"]) - set(leagues))
                found.append(
                    make_entry(None, False, "the cups model rates the clubs by their league"
                               f" results too: the dataset needs {', '.join(missing)}")
                )
            elif not training:
                found.append(
                    make_entry(
                        None,
                        False,
                        f"too little history: no season of this cup from"
                        f" {cups['first_cup_season']} to {season - 1}",
                    )
                )
            else:
                validation = (VALIDATED if is_ours(code) and code in released
                              and all(is_ours(league) for league in cups["leagues"])
                              else NOT_VALIDATED)  # fmt: skip
                found.append(make_entry("cups Elo", True, CUPS_MODEL, validation))
        elif kind in ("cup", "cup qualifying", "super cup"):
            found.append(
                make_entry(None, False, "Vesta does not have this cup's rules yet: its"
                           " matches are in the dataset, not forecast")
            )
        elif kind in NATION_KINDS:
            found.append(make_entry("nations Elo", True, NATIONS_MODEL, FAILED)
                         if _has_match_to_play(connection, competition_id)
                         else make_entry(None, False, NO_NATIONS_FIXTURE))  # fmt: skip
        else:
            found.append(make_entry(None, False, f"Vesta has no model for a competition of the"
                                    f" kind {kind!r}"))  # fmt: skip
    return found


def _has_match_to_play(connection: sqlite3.Connection, competition_id: int) -> bool:
    return connection.execute(
        "SELECT 1 FROM matches m JOIN seasons s USING (season_id) WHERE s.competition_id = ?"
        " AND m.status IN ('scheduled', 'postponed') LIMIT 1", (competition_id,)
    ).fetchone() is not None  # fmt: skip


@dataclass
class FamilyCoverage:
    """One family of keys beyond result and goals in one competition: served or not, and why."""

    code: str
    family: str
    served: bool
    reason: str
    validation: str | None
    keys: tuple[str, ...] = ()


# the families `vesta.run` publishes after the match forecasts, in the order it builds them
FAMILIES = ("team counts", "goals in time", "players", "lineups")
FAMILY_BLOCKS = {"team counts": ("B4",), "goals in time": ("B1", "B2", "B3"),
                 "players": ("B3", "B5", "B6"), "lineups": ("B5",)}  # fmt: skip
BLOCK_WORDS = {"B1": "half-time scores", "B2": "match events", "B3": "shots",
               "B4": "teams' match statistics", "B5": "lineups",
               "B6": "players' match statistics"}  # fmt: skip
FAMILY_WORDS = {
    "team counts": "the corners, cards, shots, shots on target, fouls and offsides",
    "goals in time": "when the goals come: each half's result and score, the first and the last"
                     " goal and the minutes they come in",
    "players": "the scorers and each player's shots, shots on target, key passes, fouls, cards,"
               " saves and offsides",
    "lineups": "each player's chance to start, his minutes and position, the team's formation"
               " and the expected eleven",
}  # fmt: skip
NO_NAMES = ("the dataset holds no players' names (the players file): they are published by"
            " their number")  # fmt: skip


def has_player_names(connection: sqlite3.Connection) -> bool:
    """Whether the set holds the players' names (the players file, or the whole file)."""
    if not has_column(connection, "players", "name"):
        return False
    return connection.execute("SELECT 1 FROM players WHERE name IS NOT NULL LIMIT 1"
                              ).fetchone() is not None  # fmt: skip


def read_family_parts(release: Path) -> dict[str, dict[str, tuple[str, ...]]]:
    """family -> competition code -> the keys the release publishes there."""
    plan = tomllib.loads(release.read_text(encoding="utf-8"))
    parts: dict[str, dict[str, tuple[str, ...]]] = {family: {} for family in FAMILIES}
    for part in plan["part"]:
        if part["family_member"] in parts:
            for code in part["competitions"]:
                parts[part["family_member"]][code] = tuple(part["keys"])
    return parts


def assess_families(found: list[Competition], present: dict[str, bool], spine: dict,
                    release: Path, built_at: str, has_names: bool,
                    proved: set[str] | None = None) -> list[FamilyCoverage]:  # fmt: skip
    """Each family of each competition `found` serves: what the release publishes there, from
    which data, and whether it is validated. A family pools its fit over the leagues of the
    walk-forward, so it is validated where the competition is and those leagues all come from
    files proved ours (`proved`; None: one file, the competition's own validation decides)."""
    parts = read_family_parts(release)
    leagues = set(spine["walk_forward"]["competitions"])
    pooled = proved is None or leagues <= proved
    entries = []
    for item in (item for item in found if item.served):
        for family in FAMILIES:
            keys = parts[family].get(item.code)
            missing = [BLOCK_WORDS[block] for block in FAMILY_BLOCKS[family]
                       if not present.get(block)]  # fmt: skip
            served, validation, reason = False, None, FAMILY_WORDS[family]
            if keys is None:
                cups = " and the European cups" if set(parts[family]) - leagues else ""
                reason += f": released for the five top leagues{cups} only"
            elif missing:
                reason += ": the dataset holds no " + " and no ".join(missing)
            elif family in ("players", "goals in time") and item.model != "champion":
                reason += (": they read the full model's forecast, which this competition does"
                           " not get")  # fmt: skip
            else:
                served = True
                validation = (VALIDATED if item.validation == VALIDATED and pooled
                              else NOT_VALIDATED)  # fmt: skip
                if validation == NOT_VALIDATED and item.validation == VALIDATED:
                    reason += ("; fitted on the leagues this dataset holds, not on the five top"
                               " leagues its tests used")  # fmt: skip
                if family == "lineups":
                    reason += (f"; the bans and the latest elevens are those the dataset holds"
                               f" as of {built_at}, and injuries are not included")  # fmt: skip
                if family == "players" and not has_names:
                    reason += f"; {NO_NAMES}"
            entries.append(FamilyCoverage(item.code, family, served, reason, validation,
                                          keys if served else ()))  # fmt: skip
    return entries


def describe_families(entries: list[FamilyCoverage], code: str) -> str:
    """The families of one competition in plain words, one sentence each."""
    said = []
    for entry in (entry for entry in entries if entry.code == code):
        state = f"served ({entry.validation})" if entry.served else "not served"
        said.append(f"{entry.family.capitalize()}: {state}, {entry.reason}.")
    return " ".join(said)


def check(path: Path, release: Path) -> Report:
    connection, header = open_dataset(path)
    with connection:
        present = find_present_blocks(connection)
        errors, warnings = validate(connection, present)
        spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
        found = (
            []
            if errors
            else assess_competitions(
                connection, header, present, spine, _read_released_competitions(release)
            )
        )
        families = assess_families(found, present, spine, release, header.built_at,
                                   has_player_names(connection))  # fmt: skip
    connection.close()
    return Report(str(path), asdict(header), present, errors, warnings, found, families)


def _print(report: Report) -> None:
    print(f"{report.path}: {report.header['version']}, dataset {report.header['dataset']!r}, "
          f"built {report.header['built_at']}")  # fmt: skip
    print("blocks: " + ", ".join(f"{block} {'yes' if yes else 'no'}"
                                 for block, yes in report.blocks.items()))  # fmt: skip
    for kind, problems in (("ERROR", report.errors), ("warning", report.warnings)):
        for problem in problems:
            print(f"{kind}: {problem.table}: {problem.what} ({problem.rows} rows) "
                  f"e.g. {problem.examples[:3]}")  # fmt: skip
    if report.errors:
        print("the dataset is not in the format: nothing is served")
        return
    print(f"\nresult and goals ({', '.join(SPINE_KEYS)}):")
    for item in report.competitions:
        state = "served" if item.served else "not served"
        mark = f" [{item.validation}]" if item.validation else ""
        print(f"  {item.code:10s} {item.kind:20s} {item.season!s:6s} {state:11s} "
              f"{item.model or '-':10s} {item.reason}{mark}")  # fmt: skip
    print("\nbeyond result and goals:")
    for item in report.families:
        state = f"served [{item.validation}]" if item.served else "not served"
        print(f"  {item.code:10s} {item.family:14s} {state}: {item.reason}")
    print("\nthe season keys and the live keys: not in the capability check")


def find_latest_release() -> Path:
    """The newest release plan of `config/vesta/releases`."""

    def parse_version(path: Path) -> tuple[int, ...]:
        return tuple(int(part) for part in path.stem.split("."))

    return max((ROOT / "config/vesta/releases").glob("*.toml"), key=parse_version)


def main(arguments: list[str]) -> int:
    if not arguments:
        print(__doc__)
        return 2
    release = find_latest_release()
    if "--release" in arguments:
        release = Path(arguments[arguments.index("--release") + 1])
    path = Path(arguments[0])
    report = check(path, release)
    _print(report)
    out = path.with_name(path.name + ".capabilities.json")
    out.write_text(json.dumps(asdict(report), indent=1, default=str), encoding="utf-8")
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
