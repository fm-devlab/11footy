"""Our expected goals written into our dataset, at the weekly build (S3a).

    python -m vesta.jobs.dataset_xg --world <football-world.db> <11footy.db>

The world's shots as the champion reads them (`[shots]` of `config/vesta/expected_goals.toml`),
each rated by the released xG model walk-forward (`rate_every_season`: a season's shots by the
trees fitted on the seasons before it), written to the dataset's `shots.xg` by shot id. The
dataset is the world's export, so it keeps the world's shot ids. The same world gives the same
numbers, so the job can be run again. Vesta never computes xG for a foreign dataset: there it
reads `shots.xg` as given (9 November).

Exit codes: 0 done, 1 refused (the dataset is not this world's export, a rated shot is missing
from it or is not a tracked shot), 2 a failed run.
"""

import argparse
import sqlite3
import sys
import tomllib
from pathlib import Path

import numpy as np

from vesta.data_format import FormatError, open_dataset
from vesta.expected_goals.model import ModelSettings
from vesta.expected_goals.shots import SHOT_ORDER, read_shots
from vesta.expected_goals.walk_forward import WalkForwardSettings, rate_every_season
from vesta.spine.world import open_world

ROOT = Path(__file__).resolve().parents[3]
TRACKED = "tracked"


class Refused(RuntimeError):
    """The dataset cannot take this world's expected goals."""


def read_shot_ids(world: sqlite3.Connection, section: dict, last_season: int) -> np.ndarray:
    """The ids of the shots `read_shots` reads, in its order."""
    competitions, layers = section["competitions"], section["layers"]
    return np.array([row[0] for row in world.execute(
        f"""
        SELECT sh.shot_id
        FROM shots sh
        JOIN matches m ON m.match_id = sh.match_id
        JOIN seasons s ON s.season_id = m.season_id
        JOIN competitions c ON c.competition_id = s.competition_id
        WHERE c.code IN ({", ".join("?" for _ in competitions)}) AND s.season BETWEEN ? AND ?
          AND sh.layer IN ({", ".join("?" for _ in layers)})
          AND sh.x IS NOT NULL AND sh.y IS NOT NULL
        ORDER BY {SHOT_ORDER}
        """, (*competitions, section["first_season"], last_season, *layers))],
        dtype="int64")  # fmt: skip


def rate_world_shots(world: sqlite3.Connection, expected_goals: dict
                     ) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """(shot ids, chances) of every shot the champion reads, through the world's latest season."""
    section = expected_goals["shots"]
    [last_season] = world.execute("SELECT MAX(season) FROM seasons").fetchone()
    shots = read_shots(world, section["competitions"], section["first_season"], last_season,
                       section["layers"])  # fmt: skip
    ids = read_shot_ids(world, section, last_season)
    if len(ids) != len(shots):
        raise RuntimeError(f"{len(ids)} shot ids for {len(shots)} shots read")
    chances, _ = rate_every_season(shots, ModelSettings.from_config(expected_goals),
                                   WalkForwardSettings.from_config(expected_goals))  # fmt: skip
    return ids, chances


def write_expected_goals(dataset: sqlite3.Connection, ids: np.ndarray,
                         chances: np.ndarray) -> int:  # fmt: skip
    """`shots.xg` of the rated shots, in one transaction; every other shot's left as it is."""
    dataset.execute("CREATE TEMP TABLE rated (shot_id INTEGER PRIMARY KEY, xg REAL NOT NULL)")
    try:
        dataset.executemany("INSERT INTO rated VALUES (?, ?)",
                            zip(ids.tolist(), chances.tolist(), strict=True))  # fmt: skip
        missing = dataset.execute(
            "SELECT count(*) FROM rated r LEFT JOIN shots s USING (shot_id)"
            " WHERE s.shot_id IS NULL"
        ).fetchone()[0]
        if missing:
            raise Refused(f"{missing} rated shots are not in the dataset")
        other = dataset.execute("SELECT count(*) FROM rated r JOIN shots s USING (shot_id)"
                                " WHERE s.layer IS NOT ?", (TRACKED,)).fetchone()[0]  # fmt: skip
        if other:
            raise Refused(f"{other} rated shots are not {TRACKED} shots in the dataset")
        with dataset:
            dataset.execute("UPDATE shots SET xg = (SELECT xg FROM rated r"
                            " WHERE r.shot_id = shots.shot_id)"
                            " WHERE shot_id IN (SELECT shot_id FROM rated)")  # fmt: skip
    finally:
        dataset.rollback()
        dataset.execute("DROP TABLE rated")
    return len(ids)


def run(world_path: Path, dataset_path: Path, root: Path = ROOT) -> int:
    expected_goals = tomllib.loads(
        (root / "config/vesta/expected_goals.toml").read_text(encoding="utf-8")
    )
    reading, header = open_dataset(dataset_path)
    reading.close()
    with open_world(world_path) as world:
        row = world.execute(
            "SELECT value FROM build_information WHERE key = 'built_at'"
        ).fetchone()
        built_at = row[0] if row else "an unknown instant"
        if built_at != header.built_at:
            raise Refused(f"the dataset was built at {header.built_at}, the world at {built_at}:"
                          " not its export")  # fmt: skip
        ids, chances = rate_world_shots(world, expected_goals)
    world.close()
    dataset = sqlite3.connect(dataset_path)
    try:
        return write_expected_goals(dataset, ids, chances)
    finally:
        dataset.close()


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("dataset", type=Path)
    options = parser.parse_args(arguments)
    try:
        written = run(options.world, options.dataset)
    except (Refused, FormatError, FileNotFoundError, sqlite3.DatabaseError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001 - the weekly script needs an exit code
        print(f"failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(f"{written} shots' xg written to {options.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
