"""The reproduction of the baseline B0 on the football world.

    python -m vesta.spine.baseline <path of football-world.db>

Forecasts the development seasons walk-forward with Dixon-Coles on goals and the calibrated
matrix, and prints the result's log loss per season and in all, beside the reference
verdict on the same seasons (0.99229 on 8,955 matches). Reproduced to the bit by this command.
"""

import sys
import tomllib
from pathlib import Path

from vesta.spine.dixon_coles import DixonColesParameters
from vesta.spine.match_forecast import average_log_loss
from vesta.spine.timing import TimeRules
from vesta.spine.walk_forward import WalkForwardSettings, forecast_with_baseline
from vesta.spine.world import open_world, read_league_matches

ROOT = Path(__file__).resolve().parents[3]
PREVIOUS_VERDICT = 0.99229
PREVIOUS_MATCHES = 8955


def main(arguments: list[str]) -> int:
    if len(arguments) != 1:
        print(__doc__)
        return 2
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    research = tomllib.loads((ROOT / "config/vesta/research.toml").read_text(encoding="utf-8"))
    settings = WalkForwardSettings.from_config(spine)
    seasons = tuple(research["seasons"]["development"])
    with open_world(Path(arguments[0])) as world:
        matches = read_league_matches(
            world, list(settings.competitions), settings.first_training_season - 1, max(seasons)
        )
    forecasts = forecast_with_baseline(
        matches,
        TimeRules.from_config(spine),
        DixonColesParameters.from_config(spine),
        settings,
        seasons,
    )
    for season in seasons:
        chosen = [forecast for forecast in forecasts if forecast.season == season]
        print(f"{season}-{(season + 1) % 100:02d}: {average_log_loss(chosen):.5f} on {len(chosen)}")
    print(
        f"all: {average_log_loss(forecasts):.5f} on {len(forecasts)} matches; the reference "
        f"B0: {PREVIOUS_VERDICT:.5f} on {PREVIOUS_MATCHES}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
