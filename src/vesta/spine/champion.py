"""The reproduction of the champion on the football world (the spine, step 5).

    python -m vesta.spine.champion <path of football-world.db>

Rates every shot with our expected goals (walk-forward), reads who played in every match, then
forecasts the development seasons walk-forward with the champion (Dixon-Coles on our expected
goals mixed with goals and a prior for promoted clubs, blended with the ratings of the last
eleven, one pooled calibration with spread and dispersion), and prints the result's log loss per
season and in all beside the reference verdict on the same seasons (0.98174 on 8,955
matches). Reproduced to the bit.
"""

import sys
import time
import tomllib
from pathlib import Path

from vesta.expected_goals.model import ModelSettings
from vesta.expected_goals.shots import read_shots
from vesta.expected_goals.walk_forward import (
    WalkForwardSettings as ShotWalkForwardSettings,
)
from vesta.expected_goals.walk_forward import (
    rate_every_season,
    sum_expected_goals_by_match,
)
from vesta.spine.dixon_coles import DixonColesParameters
from vesta.spine.match_forecast import MatchForecast, average_log_loss
from vesta.spine.player_ratings import PlayerRatingsParameters
from vesta.spine.timing import TimeRules
from vesta.spine.walk_forward import (
    MatchFacts,
    WalkForwardSettings,
    forecast_with_champion,
)
from vesta.spine.world import open_world, read_league_matches, read_match_players

ROOT = Path(__file__).resolve().parents[3]
PREVIOUS_VERDICT = 0.98174
PREVIOUS_MATCHES = 8955


def main(arguments: list[str]) -> int:
    if len(arguments) != 1:
        print(__doc__)
        return 2
    started = time.monotonic()
    seasons = tuple(_read_toml("config/vesta/research.toml")["seasons"]["development"])
    forecasts = forecast_world_with_champion(Path(arguments[0]), seasons)
    for season in seasons:
        chosen = [forecast for forecast in forecasts if forecast.season == season]
        print(f"{season}-{(season + 1) % 100:02d}: {average_log_loss(chosen):.5f} on {len(chosen)}")
    print(
        f"all: {average_log_loss(forecasts):.5f} on {len(forecasts)} matches; the reference "
        f"champion: {PREVIOUS_VERDICT:.5f} on {PREVIOUS_MATCHES}"
    )
    print(f"{time.monotonic() - started:.0f} seconds")
    return 0


def forecast_world_with_champion(world_path: Path, seasons: tuple[int, ...],
                                 rated: dict | None = None) -> list[MatchForecast]:  # fmt: skip
    """The champion's forecast of every league match of `seasons`, walk-forward, from the world:
    our expected goals rated walk-forward, who played in every match, the champion's families.
    With `rated`, it also receives the shots read (`shots`) and their expected goals
    (`chances`)."""
    spine = _read_toml("config/vesta/spine.toml")
    matches, facts = read_champion_inputs(world_path, seasons, rated)
    return forecast_with_champion(
        matches,
        facts,
        TimeRules.from_config(spine),
        DixonColesParameters.from_config(spine, "dixon_coles_expected_goals"),
        PlayerRatingsParameters.from_config(spine),
        int(spine["champion_calibration"]["seasons"]),
        WalkForwardSettings.from_config(spine),
        seasons,
    )


def read_champion_inputs(world_path: Path, seasons: tuple[int, ...], rated: dict | None = None
                         ) -> tuple[list, dict[int, MatchFacts]]:  # fmt: skip
    """The league matches to `seasons` and each one's facts (our expected goals, rated
    walk-forward, and who played), as the champion reads them."""
    spine = _read_toml("config/vesta/spine.toml")
    expected_goals = _read_toml("config/vesta/expected_goals.toml")
    settings = WalkForwardSettings.from_config(spine)
    shot_section = expected_goals["shots"]
    with open_world(world_path) as world:
        matches = read_league_matches(
            world, list(settings.competitions), settings.first_training_season - 1, max(seasons)
        )
        shots = read_shots(world, shot_section["competitions"], shot_section["first_season"],
                           max(seasons), shot_section["layers"])  # fmt: skip
        players = read_match_players(world, {match.match_id for match in matches})
    chances, _ = rate_every_season(
        shots,
        ModelSettings.from_config(expected_goals),
        ShotWalkForwardSettings.from_config(expected_goals),
    )
    if rated is not None:
        rated.update(shots=shots, chances=chances)
    by_match = sum_expected_goals_by_match(shots, chances)
    facts = {
        match.match_id: MatchFacts(
            *by_match.get(match.match_id, (float("nan"), float("nan"))),
            players.get(match.match_id, ()),
        )
        for match in matches
    }
    return matches, facts


def _read_toml(relative: str) -> dict:
    return tomllib.loads((ROOT / relative).read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
