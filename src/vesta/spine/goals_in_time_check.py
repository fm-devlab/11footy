"""The check of goals in time (step 3) on the football world.

    python -m vesta.spine.goals_in_time_check <path of football-world.db>

Forecasts the development seasons with the champion, reads the first-half shares and the shares
of the manners of goals from each season's training seasons, draws every match's goals in time,
and prints, for every step-3 quantity, its deciding metric beside the reference of the training
seasons' frequencies, the difference with its interval, the seasons improved and the calibration
of its lines (`vesta.spine.goals_in_time_verdict`). Reproduced to the bit by this command.
"""

import sys
import time
import tomllib
from pathlib import Path

import numpy as np

from vesta.quantities import load_quantities
from vesta.spine.champion import forecast_world_with_champion
from vesta.spine.goal_minutes import TimingSeasons, fit_profile, measure_added_minutes
from vesta.spine.goal_minutes import read_goal_history as read_history
from vesta.spine.goals_in_time import (
    GoalsInTimeParameters,
    TimeShares,
    fit_first_half_shares,
    fit_method_shares,
    split_regular_minutes,
    uses_minute_profile,
)
from vesta.spine.goals_in_time_verdict import check_goals_in_time
from vesta.spine.world import (
    open_world,
    read_goal_methods,
    read_half_time_scores,
    read_league_matches,
    read_match_stories,
)

ROOT = Path(__file__).resolve().parents[3]


def main(arguments: list[str]) -> int:
    if len(arguments) != 1:
        print(__doc__)
        return 2
    started = time.monotonic()
    spine = _read_toml("config/vesta/spine.toml")
    research = _read_toml("config/vesta/research.toml")
    seasons = tuple(research["seasons"]["development"])
    forecasts = forecast_world_with_champion(Path(arguments[0]), seasons)
    results = score_goals_in_time_forecasts(Path(arguments[0]), seasons, forecasts, spine, research)
    print("key | metric | n | model | reference | difference [interval] | seasons improved | "
          "calibration error model / reference")
    for result in results:
        print(f"{result.key} | {result.metric} | {result.forecasts} | {result.model:.4f} | "
              f"{result.reference:.4f} | {result.difference:+.4f} [{result.interval_low:+.4f}, "
              f"{result.interval_high:+.4f}] | {result.seasons_improved}/{result.seasons} | "
              f"{result.calibration_error_model:.4f} / {result.calibration_error_reference:.4f}")
    print(f"{time.monotonic() - started:.0f} seconds")
    return 0


def score_goals_in_time_forecasts(world_path: Path, seasons: tuple[int, ...], forecasts: list,
                                  spine: dict, research: dict, say=print) -> list:  # fmt: skip
    """The step-3 results of `forecasts` (the champion's, of `seasons`), the shares and the added
    time fitted on the `[timing]` seasons before each."""
    timing = TimingSeasons.from_config(spine)
    first_training = timing.first_season
    competitions = list(spine["walk_forward"]["competitions"])
    parameters = GoalsInTimeParameters.from_config(spine)
    forecasts = [forecast for forecast in forecasts if forecast.season in seasons]
    with open_world(world_path) as world:
        history = read_history(world, competitions, timing.first_season, max(seasons))
        shares, added = {}, {}
        for season in seasons:
            # the added time read from the live chain's profile of the same seasons
            before = history.season < season
            profile = fit_profile(
                history.select(before),
                history.select(before & (history.season >= timing.find_recent_seasons(season)[0])),
                timing.smoothing_minutes,
            )
            added[season] = measure_added_minutes(profile)
            scores = read_half_time_scores(world, competitions, first_training, season - 1)
            home, away = fit_first_half_shares(
                *(np.array(column) for column in (scores.home_goals, scores.away_goals,
                                                  scores.home_half_time, scores.away_half_time)),
                parameters.first_half_share_without_data,
            )  # fmt: skip
            methods = read_goal_methods(world, competitions, first_training, season - 1)
            minutes = split_regular_minutes(profile) if uses_minute_profile(spine) else None
            shares[season] = TimeShares(home, away, fit_method_shares(methods),
                                        regular_minutes=minutes)
            say(f"{season}: added time {added[season][0]:.2f} and {added[season][1]:.2f}; "
                  f"first-half shares {home:.4f} {away:.4f}; manners "
                  + ", ".join(f"{share:.3f}" for share in shares[season].methods))
        season_of = {match.match_id: match.season for match in read_league_matches(
            world, competitions, first_training, max(seasons))}  # fmt: skip
        stories = read_match_stories(world, set(season_of))
    checked = sum(1 for forecast in forecasts if forecast.match_id in stories)
    say(f"{checked} of {len(forecasts)} forecast matches have a story to check")
    return check_goals_in_time(
        forecasts, stories, season_of, shares, load_quantities(), parameters,
        int(research["verdict"]["bootstrap_repetitions"]), int(research["verdict"]["seed"]),
        first_training, added,
    )  # fmt: skip


def _read_toml(relative: str) -> dict:
    return tomllib.loads((ROOT / relative).read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
