"""Every verdict of the spine and its family, end to end, in one command and one report.

    python -m vesta.verdicts <football-world.db> [<report.md>]

On the development seasons only (`config/vesta/research.toml`; the closed seasons are never
read): the reproduction of B0 against the reference 0.99229, the champion against the reference
0.98174, the cups (each season against the reference) and the national teams (the years
2012-2021 against 0.8870), goals in time (every step-3 key against the training frequencies),
the ties (against the training share). The champion is computed once and shared. Each verdict
is run on its own: one that cannot run is written in the report with its reason, and the others
go on.

Pass or fail, by the discipline (`src/vesta/research/README.md`):
- a reproduction passes when its log loss is within `[reproduction] tolerance` of the
  reference (a cup season or the nations: no worse than the reference by more than it);
- a check against a reference passes when the difference's interval lies below zero, at least
  `min_seasons_improved` seasons improved and the calibration error is no worse than the
  reference's by more than `calibration_tolerance`.
The report goes to `docs/research/<date>-verdicts.md` unless another path is given; nothing is
published and no tag is made by it.
"""

import hashlib
import sys
import time
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from vesta.research.repository import run_git
from vesta.spine.baseline import PREVIOUS_MATCHES as B0_MATCHES
from vesta.spine.baseline import PREVIOUS_VERDICT as BASELINE_VERDICT
from vesta.spine.champion import PREVIOUS_VERDICT as CHAMPION_VERDICT
from vesta.spine.champion import forecast_world_with_champion
from vesta.spine.cups import CupsParameters, forecast_cups
from vesta.spine.cups_and_nations import NATION_YEARS, PREVIOUS_CUPS, PREVIOUS_NATIONS
from vesta.spine.dixon_coles import DixonColesParameters
from vesta.spine.goals_in_time_check import score_goals_in_time_forecasts
from vesta.spine.match_forecast import average_log_loss
from vesta.spine.nations import NationsParameters, forecast_nations
from vesta.spine.ties import TieParameters
from vesta.spine.ties_check import check_ties, read_ties
from vesta.spine.timing import TimeRules
from vesta.spine.walk_forward import WalkForwardSettings, forecast_with_baseline
from vesta.spine.world import (
    open_world,
    read_club_matches,
    read_league_matches,
    read_nation_matches,
)

ROOT = Path(__file__).resolve().parents[2]
# the diagnostics measured on the uncorrected world with its duplicated matches left out
# (the spine's first runs, 25 October 2026), stated beside the corrected world's numbers
DEDUPLICATED = {"B0": (0.99213, 8976), "champion": (0.98379, 8976)}


@dataclass
class Verdict:
    name: str
    passed: bool | None  # None: it could not run
    rows: list[str] = field(default_factory=list)  # Markdown lines
    note: str = ""


@dataclass(frozen=True)
class Rules:
    tolerance: float
    min_seasons_improved: int
    calibration_tolerance: float
    repetitions: int
    seed: int

    @classmethod
    def from_config(cls, research: dict) -> "Rules":
        verdict = research["verdict"]
        return cls(float(research["reproduction"]["tolerance"]),
                   int(verdict["min_seasons_improved"]), float(verdict["calibration_tolerance"]),
                   int(verdict["bootstrap_repetitions"]), int(verdict["seed"]))  # fmt: skip


def run_verdicts(world: Path, seasons: tuple[int, ...], spine: dict, research: dict,
                 say: Callable[[str], None] = print) -> list[Verdict]:  # fmt: skip
    rules = Rules.from_config(research)
    closed = set(research["seasons"]["closed"])
    if closed & set(seasons):
        raise ValueError(f"the closed seasons {sorted(closed & set(seasons))} are read only by"
                         " the blind verdicts (python -m vesta.blind_verdicts)")  # fmt: skip
    shared: dict[str, list] = {}

    def run_champion_once() -> list:
        if "champion" not in shared:
            first = int(spine["live_chain"]["first_effects_season"])
            span = tuple(range(min(first, min(seasons)), max(seasons) + 1))
            shared["champion"] = forecast_world_with_champion(world, span)
        return shared["champion"]

    steps = [
        ("B0 (the previous baseline)", lambda: _judge_baseline(world, seasons, spine, rules)),
        ("The champion", lambda: _judge_champion(run_champion_once(), seasons, rules)),
        ("The UEFA cups", lambda: _judge_cups(world, seasons, spine, rules)),
        ("The national teams", lambda: _judge_nations(world, spine, rules)),
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


def _judge_reproduction(name, forecasts, seasons, previous, previous_matches, rules,
                        diagnostic=None) -> Verdict:  # fmt: skip
    rows = ["| season | log loss | matches |", "|---|---|---|"]
    for season in seasons:
        chosen = [forecast for forecast in forecasts if forecast.season == season]
        if chosen:
            rows.append(f"| {season}-{(season + 1) % 100:02d} | {average_log_loss(chosen):.5f} | "
                        f"{len(chosen)} |")  # fmt: skip
    chosen = [forecast for forecast in forecasts if forecast.season in seasons]
    if not chosen:
        return Verdict(name, None, rows, "could not run: no match of the seasons forecast")
    loss = average_log_loss(chosen)
    rows.append(f"| all | {loss:.5f} | {len(chosen)} |")
    passed = abs(loss - previous) <= rules.tolerance
    note = (f"the reference: {previous:.5f} on {previous_matches}; tolerance "
            f"{rules.tolerance}")
    if diagnostic is not None:
        note += (f"; the deduplicated diagnostic on the uncorrected world: {diagnostic[0]:.5f} on "
                 f"{diagnostic[1]}")
    if not passed:
        note += ("; a miss is investigated and explained match by match, never retuned to pass "
                 "(25 October 2026)")
    return Verdict(name, passed, rows, note)


def _judge_baseline(world, seasons, spine, rules) -> Verdict:
    settings = WalkForwardSettings.from_config(spine)
    with open_world(world) as connection:
        matches = read_league_matches(connection, list(settings.competitions),
                                      settings.first_training_season - 1, max(seasons))  # fmt: skip
    forecasts = forecast_with_baseline(matches, TimeRules.from_config(spine),
                                       DixonColesParameters.from_config(spine), settings,
                                       seasons)  # fmt: skip
    return _judge_reproduction("", forecasts, seasons, BASELINE_VERDICT, B0_MATCHES, rules,
                               DEDUPLICATED["B0"])


def _judge_champion(forecasts, seasons, rules) -> Verdict:
    return _judge_reproduction("", forecasts, seasons, CHAMPION_VERDICT, B0_MATCHES, rules,
                               DEDUPLICATED["champion"])


def _judge_cups(world, seasons, spine, rules) -> Verdict:
    cups = CupsParameters.from_config(spine)
    with open_world(world) as connection:
        matches = read_club_matches(connection, list(cups.leagues), list(cups.cups),
                                    cups.first_league_season, cups.first_cup_season,
                                    max(seasons))  # fmt: skip
    forecasts = forecast_cups(matches, cups, seasons)
    rows = ["| season | log loss | matches | the reference |", "|---|---|---|---|"]
    passed = True
    for season in seasons:
        chosen = [forecast for forecast in forecasts if forecast.season == season]
        previous = PREVIOUS_CUPS.get(season)
        if not chosen:
            continue
        loss = average_log_loss(chosen)
        rows.append(f"| {season}-{(season + 1) % 100:02d} | {loss:.4f} | {len(chosen)} | "
                    + (f"{previous[1]:.4f} on {previous[0]} |" if previous else "none |"))
        if previous:
            passed &= loss <= previous[1] + rules.tolerance
    if not forecasts:
        return Verdict("", None, rows, "could not run: no cup match forecast")
    return Verdict("", passed, rows, f"each season no worse than the reference by more "
                   f"than {rules.tolerance}")  # fmt: skip


def _judge_nations(world, spine, rules) -> Verdict:
    with open_world(world) as connection:
        matches = read_nation_matches(connection)
    forecasts = forecast_nations(matches, NationsParameters.from_config(spine), NATION_YEARS)
    if not forecasts:
        return Verdict("", None, [], "could not run: no national match forecast")
    loss = average_log_loss(forecasts)
    rows = ["| years | log loss | matches | the reference |", "|---|---|---|---|",
            f"| {NATION_YEARS[0]}-{NATION_YEARS[-1]} | {loss:.4f} | {len(forecasts)} | "
            f"{PREVIOUS_NATIONS[1]:.4f} on {PREVIOUS_NATIONS[0]} |"]  # fmt: skip
    note = f"no worse than the reference by more than {rules.tolerance}"
    return Verdict("", loss <= PREVIOUS_NATIONS[1] + rules.tolerance, rows, note)


def _passes(high: float, improved: int, model_calibration: float, reference_calibration: float,
            rules: Rules) -> bool:  # fmt: skip
    calibrated = model_calibration <= reference_calibration + rules.calibration_tolerance
    return high < 0 and improved >= rules.min_seasons_improved and calibrated


def _judge_goals_in_time(world, seasons, forecasts, spine, research, rules, say) -> Verdict:
    results = score_goals_in_time_forecasts(world, seasons, forecasts, spine, research, say)
    rows = ["| key | metric | n | model | reference | difference [interval] | seasons improved |"
            " calibration model / reference | verdict |", "|---|---|---|---|---|---|---|---|---|"]
    passed_keys = 0
    for result in results:
        passed = _passes(result.interval_high, result.seasons_improved,
                         result.calibration_error_model, result.calibration_error_reference,
                         rules)  # fmt: skip
        passed_keys += passed
        rows.append(f"| {result.key} | {result.metric} | {result.forecasts} | {result.model:.4f} | "
                    f"{result.reference:.4f} | {result.difference:+.4f} "
                    f"[{result.interval_low:+.4f}, {result.interval_high:+.4f}] | "
                    f"{result.seasons_improved}/{result.seasons} | "
                    f"{result.calibration_error_model:.4f} / "
                    f"{result.calibration_error_reference:.4f} | "
                    f"{'pass' if passed else 'fail'} |")  # fmt: skip
    if not results:
        return Verdict("", None, rows, "could not run: no key could be checked")
    return Verdict("", passed_keys == len(results), rows,
                   f"{passed_keys} of {len(results)} keys pass; each key is published only on "
                   "its own pass")  # fmt: skip


def _judge_ties(world, seasons, spine, rules) -> Verdict:
    cups = CupsParameters.from_config(spine)
    with open_world(world) as connection:
        matches = read_club_matches(connection, list(cups.leagues), list(cups.cups),
                                    cups.first_league_season, cups.first_cup_season,
                                    max(seasons))  # fmt: skip
        ties = read_ties(connection, list(cups.cups))
    forecasts = {forecast.match_id: forecast
                 for forecast in forecast_cups(matches, cups, seasons)}  # fmt: skip
    result = check_ties(ties, forecasts, seasons, TieParameters.from_config(spine),
                        rules.repetitions, rules.seed)  # fmt: skip
    mean, low, high = result.difference
    passed = _passes(high, result.seasons_improved, result.calibration_model,
                     result.calibration_reference, rules)  # fmt: skip
    rows = ["| ties | model | reference | difference [interval] | seasons improved |"
            " calibration model / reference |", "|---|---|---|---|---|---|",
            f"| {result.ties} | {result.model:.4f} | {result.reference:.4f} | {mean:+.4f} "
            f"[{low:+.4f}, {high:+.4f}] | {result.seasons_improved}/{result.seasons} | "
            f"{result.calibration_model:.4f} / {result.calibration_reference:.4f} |"]  # fmt: skip
    return Verdict("", passed, rows, "against the training share of ties won by the first leg's"
                   " home side")  # fmt: skip


def report(verdicts: list[Verdict], world: Path, seasons: tuple[int, ...]) -> str:
    """The report: the world with its sha256 and the code's commit first, so a verdict says
    exactly what it judged."""
    sha256 = hash_world(world) if world.exists() else "missing"
    commit = run_git(ROOT, "rev-parse", "HEAD").strip()
    lines = [f"# The verdicts, {datetime.now(UTC):%Y-%m-%d %H:%M} UTC", "",
             f"World: `{world}` (sha256 {sha256}). Code: `{commit}`. Development seasons: "
             f"{', '.join(map(str, seasons))}. Written by `python -m vesta.verdicts`; nothing "
             "published, no tag.", "",
             "| verdict | result |", "|---|---|"]  # fmt: skip
    for verdict in verdicts:
        outcome = {True: "pass", False: "fail", None: "did not run"}[verdict.passed]
        lines.append(f"| {verdict.name} | {outcome} |")
    for verdict in verdicts:
        lines += ["", f"## {verdict.name}", ""]
        if verdict.rows:
            lines += verdict.rows + [""]
        lines.append(verdict.note)
    return "\n".join(lines) + "\n"


def hash_world(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main(arguments: list[str]) -> int:
    if len(arguments) not in (1, 2):
        print(__doc__)
        return 2
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    research = tomllib.loads((ROOT / "config/vesta/research.toml").read_text(encoding="utf-8"))
    seasons = tuple(research["seasons"]["development"])
    world = Path(arguments[0])
    verdicts = run_verdicts(world, seasons, spine, research)
    path = Path(arguments[1]) if len(arguments) == 2 else (
        ROOT / "docs/research" / f"{datetime.now(UTC):%Y-%m-%d}-verdicts.md")  # fmt: skip
    path.write_text(report(verdicts, world, seasons), encoding="utf-8")
    print(f"written {path}")
    return 0 if all(verdict.passed is not None for verdict in verdicts) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


