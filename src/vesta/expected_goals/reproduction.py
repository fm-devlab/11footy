"""The reproduction of our expected goals on the football world.

    python -m vesta.expected_goals.reproduction <path of football-world.db>

Rates every shot of the five leagues walk-forward and prints, per season and in all, the shots,
the goals, the expected goals and the per-shot log loss and Brier score, beside the reference
figures on an earlier export (59,653 expected for 58,856 goals, log loss 0.2575). Reproduced to
the bit by this command.
"""

import sys
import time
import tomllib
from pathlib import Path

import numpy as np

from vesta.expected_goals.model import ModelSettings
from vesta.expected_goals.shots import read_shots
from vesta.expected_goals.walk_forward import WalkForwardSettings, check, rate_every_season
from vesta.spine.world import open_world

ROOT = Path(__file__).resolve().parents[3]
PREVIOUS = "the reference: 548,888 shots, 58,856 goals, 59,653 expected, log loss 0.2575"


def main(arguments: list[str]) -> int:
    if len(arguments) != 1:
        print(__doc__)
        return 2
    started = time.monotonic()
    configuration = tomllib.loads(
        (ROOT / "config/vesta/expected_goals.toml").read_text(encoding="utf-8")
    )
    section = configuration["shots"]
    with open_world(Path(arguments[0])) as world:
        last_season = world.execute("SELECT MAX(season) FROM seasons").fetchone()[0]
        shots = read_shots(
            world, section["competitions"], section["first_season"], last_season,
            section["layers"],
        )  # fmt: skip
    chances, models = rate_every_season(
        shots,
        ModelSettings.from_config(configuration),
        WalkForwardSettings.from_config(configuration),
    )
    for name, model in models.items():
        print(f"{name}: fitted on {model.shots:,} shots, {model.trees} trees")
    for season in sorted(set(shots.season.tolist())):
        chosen = shots.season == season
        name = f"{season}-{(season + 1) % 100:02d}"
        print(_format_line(name, check(chances[chosen], shots.goal[chosen])))
    print(_format_line("all", check(chances, shots.goal)))
    print(PREVIOUS)
    print(f"{time.monotonic() - started:.0f} seconds")
    return 0 if np.isfinite(chances).all() else 1


def _format_line(name: str, result) -> str:
    return (
        f"{name}: {result.shots:,} shots, {result.goals:,} goals, {result.expected_goals:,.1f}"
        f" expected, log loss {result.log_loss:.4f}, Brier {result.brier:.4f}"
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
