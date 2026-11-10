"""The public track record: every pre-match forecast of record of the current Vesta (2.x), raw,
with what happened beside it (9 November 2026).

    python -m vesta.record <forecast-log.db> <dataset file or folder> [...]
                           --out <repository folder> --assets <release folder>
                           [--cache <folder>]

The same log and dataset give the same bytes. Into `--assets`, the files attached to each
dataset release (too large for the repository's history):
- `track-record.db`: the matches, quantities, forecasts and their lines, each kept once, and the
  view `forecast_lines` giving one row per line of every forecast of record (its probability,
  whether it happened, the outcome itself); `calibration` (when Vesta said a chance in a tenth,
  how often it happened) and `about`;
- `track-record.csv.gz`: the rows of `forecast_lines`, gzipped.
Into `--out`, the repository's `track-record/` folder: `README.md` (every column in plain words,
and where the two files are), `calibration.csv` and `summary.md` (the numbers 11footy's README
shows between its track-record markers: the matches counted, the result's log loss, how often
the favourite won, the calibration of every family).

Nothing is selected or removed: misses and hits, withdrawn forecasts (marked, with their cause)
and the ones still waiting for their match. The summaries are calibration and, for the match
result, its log loss and the favourite's share; no comparison with any reference or other
forecaster. Vesta v1's record is not exported. Names come from the
dataset (competitions, teams, players); nothing names a data provider or a path.

Exit codes: 0 written, 1 a format or file error, 2 a failed run.
"""

import argparse
import csv
import gzip
import io
import math
import sqlite3
import sys
from pathlib import Path

from vesta.data_format import FormatError, has_column
from vesta.datasets import find_dataset_paths, open_datasets
from vesta.forecast_log.store import open_for_reading

VERSIONS = "vesta-2.%"  # the current Vesta: every 2.x version, on our dataset or another
BUCKETS = 10
RESULT = "M1.result"
OUTCOMES = ("outcome=home", "outcome=draw", "outcome=away")
MARKERS = ("<!-- track-record:start -->", "<!-- track-record:end -->")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December")

COLUMNS = [
    ("match_id", "INTEGER", "the match's id in the dataset"),
    ("competition", "TEXT", "the competition's code"),
    ("competition_name", "TEXT", "the competition's name (its code where the dataset has none)"),
    ("season", "INTEGER", "the season (the year it starts)"),
    ("kickoff_utc", "TEXT", "the kick-off the forecast was made for, UTC"),
    ("home_team", "TEXT", "the home side"),
    ("away_team", "TEXT", "the away side"),
    ("published_at", "TEXT", "when Vesta published the forecast, UTC, before the kick-off"),
    ("model_version", "TEXT", "the Vesta version that made it (a '-dataset' version: made on a "
                              "dataset no verdict covered, with that caveat)"),
    ("quantity_key", "TEXT", "what is forecast (e.g. M1.result, the match result)"),
    ("quantity_name", "TEXT", "the same in words"),
    ("probability_kind", "TEXT", "the family of the quantity (results, goals, team counts...)"),
    ("subject", "TEXT", "who it is about: empty for the match, else the team or the player"),
    ("line", "TEXT", "the outcome the probability is for (e.g. outcome=home, over=2.5); for a "
                     "quantity forecast over windows of the match, `<window>:<line>`"),
    ("probability", "REAL", "the chance Vesta gave that line, from 0 to 1"),
    ("state", "TEXT", "counted (the match was played and the line checked), pending (not yet), "
                      "void (no outcome: cancelled, abandoned, awarded, or a player who did not "
                      "play), withdrawn (published, then withdrawn for a bug or a wrong datum)"),
    ("happened", "INTEGER", "1 when the line happened, 0 when it did not; empty unless counted"),
    ("withdrawn_because", "TEXT", "for a withdrawn forecast: bug or wrong_datum"),
    ("outcome", "TEXT", "what happened, as recorded when the match was checked (JSON)"),
    ("checked_at", "TEXT", "when the line was last checked, UTC"),
    ("subject_id", "INTEGER", "the id in the dataset of the match, the team or the player"),
]

CALIBRATION = [
    ("probability_kind", "TEXT", "the family of the quantities, or 'all' for every family"),
    ("said_from", "REAL", "the tenth of the probabilities: from this value"),
    ("said_to", "REAL", "to this one (the last tenth includes 1)"),
    ("said_mean", "REAL", "the mean probability Vesta gave the lines of the tenth"),
    ("happened_share", "REAL", "the share of those lines that happened"),
    ("count", "INTEGER", "how many counted lines are in the tenth"),
]


def read_record(log: sqlite3.Connection) -> list[tuple]:
    """Every line of every pre-match forecast of record of the current Vesta, in a fixed order:
    (match, key, subject, publication id, line, probability, model, published at, scheduled
    kick-off, state, happened, withdrawal cause, checked at, outcome, kind, name, subject kind).
    The log's `checks` view with each line's window (a key forecast over several windows of the
    match has one line per window: `<window>:<line>`)."""
    return log.execute(
        """
        SELECT r.match_id, pr.quantity_key, pr.subject_id, pr.publication_id,
               CASE pr.window_key WHEN '' THEN pr.line_key
                    ELSE pr.window_key || ':' || pr.line_key END,
               pr.probability, pr.model_version, p.published_at, p.scheduled_kickoff_utc,
               CASE WHEN w.withdrawal_id IS NOT NULL THEN 'withdrawn'
                    WHEN c.status = 'void' THEN 'void'
                    WHEN c.status = 'checked' THEN 'counted' ELSE 'pending' END,
               c.happened, w.cause, c.checked_at,
               (SELECT o.outcome_value FROM outcomes o WHERE o.match_id = r.match_id
                  AND o.quantity_key = pr.quantity_key AND o.subject_id = pr.subject_id
                  AND o.window_key = pr.window_key
                ORDER BY o.outcome_id DESC LIMIT 1),
               q.probability_kind, q.name, q.subject_kind
        FROM forecasts_of_record r
        JOIN publications p ON p.publication_id = r.publication_id
        JOIN probabilities pr ON pr.publication_id = r.publication_id
            AND pr.quantity_key = r.quantity_key AND pr.subject_id = r.subject_id
        JOIN quantities q ON q.quantity_key = pr.quantity_key
        LEFT JOIN line_checks c ON c.check_id = (
            SELECT MAX(last.check_id) FROM line_checks last
            WHERE last.publication_id = pr.publication_id
              AND last.quantity_key = pr.quantity_key AND last.subject_id = pr.subject_id
              AND last.line_key = pr.line_key)
        LEFT JOIN withdrawals w ON w.publication_id = r.publication_id
        WHERE p.stage = 'pre_match' AND pr.model_version LIKE ?
        ORDER BY r.match_id, pr.quantity_key, pr.subject_id, pr.window_key, pr.line_key
        """, (VERSIONS,)).fetchall()  # fmt: skip


def read_names(dataset: sqlite3.Connection, match_ids: set[int]) -> tuple[dict, dict, dict]:
    """(match -> competition, its name, season, home, away), team -> name, player -> name."""
    matches = {}
    name = "c.name" if has_column(dataset, "competitions", "name") else "c.code"  # optional
    for row in dataset.execute(
            f"SELECT m.match_id, c.code, {name}, s.season, h.name, a.name FROM matches m"
            " JOIN seasons s ON s.season_id = m.season_id"
            " JOIN competitions c ON c.competition_id = s.competition_id"
            " JOIN teams h ON h.team_id = m.home_team_id"
            " JOIN teams a ON a.team_id = m.away_team_id"):  # fmt: skip
        if row[0] in match_ids:
            matches[row[0]] = row[1:]
    teams = dict(dataset.execute("SELECT team_id, name FROM teams"))
    has_players = dataset.execute("SELECT 1 FROM sqlite_master WHERE type = 'table'"
                                  " AND name = 'players'").fetchone()  # fmt: skip
    players = dict(dataset.execute("SELECT player_id, name FROM players")) if has_players else {}
    return matches, teams, players


def build_rows(record: list[tuple], names: tuple[dict, dict, dict]) -> list[tuple]:
    matches, teams, players = names
    rows = []
    for (match_id, key, subject_id, _, line, probability, version, published_at, kickoff, state,
         happened, cause, checked_at, outcome, kind, name, subject_kind) in record:  # fmt: skip
        competition, competition_name, season, home, away = matches.get(
            match_id, (None, None, None, None, None))
        subject = {"team": teams, "player": players}.get(subject_kind)
        rows.append((match_id, competition, competition_name, season, kickoff, home, away,
                     published_at, version, key, name, kind,
                     "" if subject is None else subject.get(subject_id, str(subject_id)),
                     line, probability, state, happened if state == "counted" else None, cause,
                     outcome, checked_at, subject_id))  # fmt: skip
    return rows


def build_calibration(rows: list[tuple]) -> list[tuple]:
    """Per probability kind and for every kind, each tenth's mean probability, the share that
    happened and the count, on the counted lines only."""
    sums: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        if row[15] != "counted":
            continue
        tenth = min(int(row[14] * BUCKETS), BUCKETS - 1)
        for kind in (row[11], "all"):
            said, happened, count = sums.setdefault((kind, tenth), [0.0, 0.0, 0])
            sums[(kind, tenth)] = [said + row[14], happened + row[16], count + 1]
    return [(kind, tenth / BUCKETS, (tenth + 1) / BUCKETS, said / count, happened / count, count)
            for (kind, tenth), (said, happened, count) in sorted(sums.items())]  # fmt: skip


def score_results(rows: list[tuple]) -> tuple[int, float, float, str, str] | None:
    """The match result over its counted forecasts with three outcome lines and one that
    happened: (matches, mean log loss, the share the likeliest outcome won, first and last
    kick-off); None before the first."""
    forecasts: dict[tuple, dict[str, tuple]] = {}
    for row in rows:
        if row[9] == RESULT and row[15] == "counted" and row[13] in OUTCOMES:
            forecasts.setdefault((row[0], row[8]), {})[row[13]] = (row[14], row[16], row[4])
    loss, favourites, kickoffs = 0.0, 0, []
    for lines in forecasts.values():
        happened = [line for line, (_, hit, _) in lines.items() if hit == 1]
        if len(lines) != len(OUTCOMES) or len(happened) != 1:
            continue
        loss -= math.log(lines[happened[0]][0])
        favourites += max(OUTCOMES, key=lambda line: lines[line][0]) == happened[0]
        kickoffs.append(lines[happened[0]][2])
    if not kickoffs:
        return None
    return (len(kickoffs), loss / len(kickoffs), favourites / len(kickoffs), min(kickoffs),
            max(kickoffs))  # fmt: skip


def _day(timestamp: str) -> str:
    year, month, day = timestamp[:10].split("-")
    return f"{int(day)} {MONTHS[int(month) - 1]} {year}"


def build_summary(rows: list[tuple], calibration: list[tuple]) -> str:
    """The block 11footy's README shows between its markers, in plain words."""
    lines = [MARKERS[0], ""]
    results = score_results(rows)
    if results is None:
        lines += ["No match with a published forecast has been played yet: the first numbers "
                  "come with the first weekly update.", ""]
    else:
        matches, loss, favourite, first, last = results
        counted = sum(row[15] == "counted" for row in rows)
        lines += [f"From {_day(first)} to {_day(last)}, updated every week:", "",
                  f"- **{matches:,} matches** played, **{counted:,} probabilities** checked "
                  "against what happened.",
                  f"- The match result: log loss **{loss:.3f}**; Vesta's likeliest outcome "
                  f"happened in **{favourite:.0%}** of the matches.", "",
                  "| Vesta said | On average | It happened | Probabilities |",
                  "|---|---|---|---|",
                  *[f"| {low:.0%} to {high:.0%} | {said:.0%} | {share:.0%} | {count:,} |"
                    for kind, low, high, said, share, count in calibration if kind == "all"],
                  "",
                  "Every family's own calibration is in "
                  "[`track-record/calibration.csv`](track-record/calibration.csv).", ""]
    return "\n".join([*lines, MARKERS[1]]) + "\n"


# the database's tables, each a slice of the flat rows (their indexes in `build_rows`'s tuple)
TABLES = {
    "matches": (("match_id", "competition", "competition_name", "season", "home_team",
                 "away_team"), (0, 1, 2, 3, 5, 6)),
    "quantities": (("quantity_key", "quantity_name", "probability_kind"), (9, 10, 11)),
    "forecasts": (("forecast_id", "match_id", "quantity_key", "subject", "subject_id",
                   "kickoff_utc", "published_at", "model_version", "withdrawn_because",
                   "outcome"), (0, 9, 12, 20, 4, 7, 8, 17, 18)),
    "lines": (("forecast_id", "line", "probability", "state", "happened", "checked_at"),
              (13, 14, 15, 16, 19)),
}
KINDS = {name: kind for name, kind, _ in COLUMNS} | {"forecast_id": "INTEGER"}


def write_database(path: Path, rows: list[tuple], calibration: list[tuple],
                   about: list[tuple[str, str]]) -> None:  # fmt: skip
    """The rows kept once each (a match, a quantity, a forecast, its lines) and the view
    `forecast_lines` giving them back flat, as the CSV has them."""
    path.unlink(missing_ok=True)
    connection = sqlite3.connect(path)
    ids: dict[tuple, int] = {}
    values: dict[str, dict] = {table: {} for table in TABLES}
    for row in rows:
        forecast = ids.setdefault((row[0], row[9], row[20]), len(ids) + 1)
        for table, (_, picked) in TABLES.items():
            cells = tuple(row[index] for index in picked)
            if table in ("forecasts", "lines"):
                cells = (forecast, *cells)
            key = {"matches": row[0], "quantities": row[9], "forecasts": forecast,
                   "lines": (forecast, row[13])}[table]  # fmt: skip
            values[table].setdefault(key, cells)
    for table, (columns, _) in TABLES.items():
        connection.execute(f"CREATE TABLE {table} ("
                           + ", ".join(f"{name} {KINDS[name]}" for name in columns) + ")")
        connection.executemany(
            f"INSERT INTO {table} VALUES ({', '.join('?' for _ in columns)})",
            values[table].values(),
        )
    connection.execute(
        "CREATE VIEW forecast_lines AS SELECT "
        + ", ".join({"match_id": "f.match_id", "quantity_key": "f.quantity_key"}.get(name, name)
                    for name, _, _ in COLUMNS)
        + " FROM lines JOIN forecasts f USING (forecast_id) JOIN matches m"
          " ON m.match_id = f.match_id JOIN quantities q ON q.quantity_key = f.quantity_key"
          " ORDER BY forecast_id, line")  # fmt: skip
    for table, columns, items in (("calibration", CALIBRATION, calibration),
                                  ("about", [("key", "TEXT", ""), ("value", "TEXT", "")], about)):
        connection.execute(f"CREATE TABLE {table} ("
                           + ", ".join(f"{name} {kind}" for name, kind, _ in columns) + ")")
        connection.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' for _ in columns)})",
                               items)  # fmt: skip
    connection.commit()
    connection.execute("VACUUM")
    connection.close()


def write_csv(path: Path, columns: list[tuple], rows: list[tuple], packed: bool) -> None:
    """The rows under their columns' names; `packed`, gzipped with no name or time inside (the
    same rows give the same bytes)."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([name for name, _, _ in columns])
    writer.writerows(rows)
    data = buffer.getvalue().encode("utf-8")
    if not packed:
        path.write_bytes(data)
        return
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                                                 mtime=0) as gzipped:  # fmt: skip
        gzipped.write(data)


def build_readme(about: list[tuple[str, str]]) -> str:
    lines = ["# Vesta's track record", "",
             "Every pre-match forecast of record Vesta 2.x has published, raw: each probability it",
             "gave, and beside it what happened. Nothing is selected or removed: the misses with",
             "the hits, the withdrawn forecasts marked, the matches still to be played marked",
             "pending. Written each week from Vesta's forecast log by `python -m vesta.record`.",
             "", *[f"- {key}: {value}" for key, value in about], "",
             "The full record is attached to the latest dataset release of this repository",
             "(Releases): `track-record.db` (SQLite) and `track-record.csv.gz` (the same rows as",
             "CSV, gzipped). This folder holds this file, `calibration.csv` and `summary.md`",
             "(the numbers shown in 11footy's README).", "",
             "## track-record.db, view `forecast_lines` (and track-record.csv.gz)", "",
             "One row per line of a forecast: a probability and whether that line happened. The",
             "database keeps each match, quantity and forecast once (tables `matches`,",
             "`quantities`, `forecasts`, `lines`); the view joins them back.", "",
             "| column | meaning |", "|---|---|",
             *[f"| {name} | {meaning} |" for name, _, meaning in COLUMNS], "",
             "## calibration.csv (and track-record.db, table `calibration`)", "",
             "The one summary: when Vesta said a chance within a tenth (say 30% to 40%), how often",
             "it happened. When Vesta is well calibrated, happened_share is close to said_mean.",
             "Counted lines only.", "",
             "| column | meaning |", "|---|---|",
             *[f"| {name} | {meaning} |" for name, _, meaning in CALIBRATION], ""]  # fmt: skip
    return "\n".join(lines)


def run(log_path: Path, arguments: list[str | Path], out: Path, assets: Path,
        cache: Path | None = None) -> int:  # fmt: skip
    paths = find_dataset_paths(arguments)
    base = Path(arguments[0]) if Path(arguments[0]).is_dir() else Path(arguments[0]).parent
    dataset, files = open_datasets(paths, cache or base / "vesta" / "cache")
    log = open_for_reading(log_path)
    try:
        record = read_record(log)
        names = read_names(dataset, {row[0] for row in record})
    finally:
        log.close()
        dataset.close()
    rows = build_rows(record, names)
    calibration = build_calibration(rows)
    built_at = min(item.header.built_at for item in files)
    last = max((row[7] for row in rows), default="none")
    about = [("versions", "Vesta 2.x, every version"), ("dataset_built_at", built_at),
             ("last_published_at", last), ("forecast_lines", str(len(rows))),
             ("counted_lines", str(sum(row[15] == "counted" for row in rows)))]  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    write_database(assets / "track-record.db", rows, calibration, about)
    write_csv(assets / "track-record.csv.gz", COLUMNS, rows, packed=True)
    write_csv(out / "calibration.csv", CALIBRATION, calibration, packed=False)
    (out / "README.md").write_text(build_readme(about), encoding="utf-8", newline="\n")
    (out / "summary.md").write_text(build_summary(rows, calibration), encoding="utf-8",
                                    newline="\n")  # fmt: skip
    return len(rows)


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path)
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    options = parser.parse_args(arguments)
    try:
        written = run(options.log, options.datasets, options.out, options.assets, options.cache)
    except (FormatError, FileNotFoundError, sqlite3.DatabaseError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001 - the weekly script needs an exit code
        print(f"failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(f"{written} forecast lines written to {options.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
