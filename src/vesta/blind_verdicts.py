"""The blind verdicts: the new models on the closed seasons, once, after the development verdicts.

    python -m vesta.blind_verdicts <football-world.db> [<report.md>]

The launch decision (25 October 2026, amended): the blind
tests are enough, with no closed-door period. So the closed seasons (`[seasons] closed` of
`config/vesta/research.toml`: 2024-25 and 2025-26) are read once for each new model, after the
development verdicts, by this command and by nothing else: the UEFA cups, the national teams (the
calendar years of the closed seasons), goals in time and the ties. The spine is not here: its
clean measure is forward, on 2026-27, in the track record.

The rules are the development verdicts' (`vesta.verdicts`), each new model against a reference,
the difference's interval below zero and the calibration within its tolerance, with one change
the closed seasons force: there are two, so a model must improve on both (`[blind]
min_seasons_improved`). The cups and the national teams, whose development check compared them
with reference numbers, are judged here against the training seasons' frequencies
of home win, draw and away win (each counted half a time more), the same reference goals in time
uses for its keys.

Run once: the report records the code's commit, and a second run on the same commit is refused,
so no model is tuned after seeing its blind result. Nothing is published and no tag is made by it.
"""

import math
import sys
import time
import tomllib
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from vesta.research.bootstrap import assign_block, bootstrap_mean
from vesta.research.calibration import measure_calibration_error
from vesta.research.repository import run_git
from vesta.spine.champion import forecast_world_with_champion
from vesta.spine.cups import CupsParameters, forecast_cups
from vesta.spine.match_forecast import RESULTS
from vesta.spine.nations import NationsParameters, forecast_nations
from vesta.spine.world import open_world, read_club_matches, read_nation_matches
from vesta.verdicts import (
    Rules,
    Verdict,
    _judge_goals_in_time,
    _judge_ties,
    _passes,
    report,
)

ROOT = Path(__file__).resolve().parents[2]
SMOOTHING = 0.5
SMALLEST = 1e-15


def make_blind_rules(research: dict) -> Rules:
    """The development rules, with the seasons improved the closed seasons allow."""
    return replace(Rules.from_config(research),
                   min_seasons_improved=int(research["blind"]["min_seasons_improved"]))  # fmt: skip


def run_blind(world: Path, spine: dict, research: dict,
              say: Callable[[str], None] = print) -> list[Verdict]:  # fmt: skip
    """Every new model's verdict on the closed seasons."""
    seasons = tuple(research["seasons"]["closed"])
    rules = make_blind_rules(research)
    shared: dict[str, list] = {}

    def run_champion_once() -> list:
        if "champion" not in shared:
            first = int(spine["live_chain"]["first_effects_season"])
            shared["champion"] = forecast_world_with_champion(
                world, tuple(range(first, max(seasons) + 1))
            )
        return shared["champion"]

    steps = [
        ("The UEFA cups", lambda: _judge_cups_blind(world, seasons, spine, rules)),
        ("The national teams", lambda: _judge_nations_blind(world, seasons, spine, rules)),
        (
            "Goals in time (step 3)",
            lambda: _judge_goals_in_time(
                world, seasons, run_champion_once(), spine, research, rules, say
            ),
        ),
        ("Ties (step 3b)", lambda: _judge_ties(world, seasons, spine, rules)),
    ]
    verdicts = []
    for name, step in steps:
        started = time.monotonic()
        say(f"{name}...")
        try:
            verdict = step()
            verdict.name = name
        except Exception as error:  # noqa: BLE001 - one verdict that cannot run stops no other
            verdict = Verdict(name, None, note=f"could not run: {type(error).__name__}: {error}")
        verdict.note = (verdict.note + f" ({time.monotonic() - started:.0f} seconds)").strip()
        verdicts.append(verdict)
    return verdicts


def judge_against_frequencies(forecasts: list, seasons: tuple[int, ...], rules: Rules) -> Verdict:
    """The forecasts of `seasons` against the result frequencies of the forecasts of the seasons
    before each (home win, draw, away win, each counted half a time more)."""
    rows = []
    for forecast in forecasts:
        if forecast.season not in seasons:
            continue
        earlier = [other for other in forecasts if other.season < forecast.season]
        counts = [sum(other.result == result for other in earlier) for result in RESULTS]
        reference = [(count + SMOOTHING) / (len(earlier) + 3 * SMOOTHING) for count in counts]
        happened = RESULTS.index(forecast.result)
        rows.append((forecast, reference, happened))
    if not rows:
        return Verdict("", None, [], "could not run: no forecast of the seasons checked")
    model = [-math.log(max(forecast.probabilities[happened], SMALLEST))
             for forecast, _, happened in rows]  # fmt: skip
    base = [-math.log(max(reference[happened], SMALLEST)) for _, reference, happened in rows]
    differences = [mine - theirs for mine, theirs in zip(model, base, strict=True)]
    blocks = [assign_block(forecast.competition, _to_instant(forecast.forecast_time))
              for forecast, *_ in rows]  # fmt: skip
    mean, low, high = bootstrap_mean(differences, blocks, rules.repetitions, rules.seed)
    improved = sum(
        math.fsum(diff for (forecast, *_), diff in zip(rows, differences, strict=True)
                  if forecast.season == season) < 0
        for season in seasons
    )  # fmt: skip
    model_lines = [(forecast.probabilities[index], happened == index)
                   for forecast, _, happened in rows for index in range(3)]  # fmt: skip
    base_lines = [(reference[index], happened == index)
                  for _, reference, happened in rows for index in range(3)]  # fmt: skip
    model_calibration = measure_calibration_error(model_lines)
    base_calibration = measure_calibration_error(base_lines)
    passed = _passes(high, improved, model_calibration, base_calibration, rules)
    table = ["| matches | model | reference | difference [interval] | seasons improved |"
             " calibration model / reference |", "|---|---|---|---|---|---|",
             f"| {len(rows)} | {math.fsum(model) / len(rows):.4f} | "
             f"{math.fsum(base) / len(rows):.4f} | {mean:+.4f} [{low:+.4f}, {high:+.4f}] | "
             f"{improved}/{len(seasons)} | {model_calibration:.4f} / {base_calibration:.4f} |"]
    return Verdict("", passed, table, "the result's log loss against the training seasons' "
                   "frequencies of home win, draw and away win")  # fmt: skip


def _to_instant(moment: str) -> str:
    """A forecast's time as an instant (a national match's is its date: noon UTC)."""
    return moment if "T" in moment else f"{moment}T12:00:00Z"


def _judge_cups_blind(world, seasons, spine, rules) -> Verdict:
    cups = CupsParameters.from_config(spine)
    first = int(spine["walk_forward"]["first_training_season"])
    with open_world(world) as connection:
        matches = read_club_matches(connection, list(cups.leagues), list(cups.cups),
                                    cups.first_league_season, cups.first_cup_season,
                                    max(seasons))  # fmt: skip
    span = tuple(range(max(first, cups.first_cup_season + 1), max(seasons) + 1))
    return judge_against_frequencies(forecast_cups(matches, cups, span), seasons, rules)


def _judge_nations_blind(world, seasons, spine, rules) -> Verdict:
    parameters = NationsParameters.from_config(spine)
    with open_world(world) as connection:
        matches = read_nation_matches(connection)
    years = tuple(range(min(seasons) - parameters.training_years, max(seasons) + 1))
    return judge_against_frequencies(forecast_nations(matches, parameters, years), seasons, rules)


def find_earlier_report(commit: str, folder: Path) -> Path | None:
    """A blind report of the same code, if one was written: the blind verdict is run once."""
    for path in sorted(folder.glob("*-blind-verdicts.md")):
        if f"Code: `{commit}`" in path.read_text(encoding="utf-8"):
            return path
    return None


def main(arguments: list[str]) -> int:
    if len(arguments) not in (1, 2):
        print(__doc__)
        return 2
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    research = tomllib.loads((ROOT / "config/vesta/research.toml").read_text(encoding="utf-8"))
    commit = run_git(ROOT, "rev-parse", "HEAD").strip()
    earlier = find_earlier_report(commit, ROOT / "docs/research")
    if earlier is not None:
        print(f"the blind verdicts of {commit} were run already ({earlier}); they run once")
        return 1
    world = Path(arguments[0])
    seasons = tuple(research["seasons"]["closed"])
    verdicts = run_blind(world, spine, research)
    path = Path(arguments[1]) if len(arguments) == 2 else (
        ROOT / "docs/research" / f"{datetime.now(UTC):%Y-%m-%d}-blind-verdicts.md")  # fmt: skip
    text = report(verdicts, world, seasons).replace("# The verdicts", "# The blind verdicts", 1)
    text = text.replace("Development seasons:", "Closed seasons:", 1)
    path.write_text(text, encoding="utf-8")
    print(f"written {path}")
    return 0 if all(verdict.passed is not None for verdict in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
