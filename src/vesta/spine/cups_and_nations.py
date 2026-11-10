"""The reproduction of the spine for the UEFA cups and the national teams on the football world.

    python -m vesta.spine.cups_and_nations <path of football-world.db>

Prints the result's log loss of each, season by season (cups) and year by year (nations), beside
the reference numbers on the same seasons and years: the cups on the development seasons
2019-20 to 2023-24 only (the reference's Elo Poisson; the closed seasons 2024-25 and 2025-26
stay closed), the national teams on the development years 2012-2021 (0.8870 on 9,033 matches).
Reproduced to the bit by this command.
"""

import sys
import tomllib
from pathlib import Path

from vesta.spine.cups import CupsParameters, forecast_cups
from vesta.spine.match_forecast import average_log_loss
from vesta.spine.nations import NationsParameters, forecast_nations
from vesta.spine.world import open_world, read_club_matches, read_nation_matches

ROOT = Path(__file__).resolve().parents[3]
CUP_SEASONS = (2019, 2020, 2021, 2022, 2023)
# the reference, per season: (matches, log loss of its Elo Poisson)
PREVIOUS_CUPS = {2019: (721, 0.9652), 2020: (540, 0.9289), 2021: (816, 0.9675),
                 2022: (804, 0.9753), 2023: (806, 0.9309)}  # fmt: skip
NATION_YEARS = tuple(range(2012, 2022))
PREVIOUS_NATIONS = (9033, 0.8870)


def main(arguments: list[str]) -> int:
    if len(arguments) != 1:
        print(__doc__)
        return 2
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    cups = CupsParameters.from_config(spine)
    with open_world(Path(arguments[0])) as world:
        club_matches = read_club_matches(world, list(cups.leagues), list(cups.cups),
                                         cups.first_league_season, cups.first_cup_season,
                                         max(CUP_SEASONS))  # fmt: skip
        nation_matches = read_nation_matches(world)
    cup_forecasts = forecast_cups(club_matches, cups, CUP_SEASONS)
    print("The UEFA cups:")
    for season in CUP_SEASONS:
        chosen = [forecast for forecast in cup_forecasts if forecast.season == season]
        matches, loss = PREVIOUS_CUPS[season]
        print(
            f"  {season}-{(season + 1) % 100:02d}: {average_log_loss(chosen):.4f} on {len(chosen)};"
            f" the reference: {loss:.4f} on {matches}"
        )
    previous = sum(m * loss for m, loss in PREVIOUS_CUPS.values())
    previous_matches = sum(m for m, _ in PREVIOUS_CUPS.values())
    print(f"  all: {average_log_loss(cup_forecasts):.4f} on {len(cup_forecasts)}; the reference: "
          f"{previous / previous_matches:.4f} on {previous_matches}")
    nation_forecasts = forecast_nations(
        nation_matches, NationsParameters.from_config(spine), NATION_YEARS
    )
    print(f"The national teams, {NATION_YEARS[0]}-{NATION_YEARS[-1]}: "
          f"{average_log_loss(nation_forecasts):.4f} on {len(nation_forecasts)}; the reference: "
          f"{PREVIOUS_NATIONS[1]:.4f} on {PREVIOUS_NATIONS[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
