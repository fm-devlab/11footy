"""The check of ties (step 3b, M14.qualifies.team) on the football world.

Every tie of the UEFA cups in the world's `match_ties` whose matches the cups model forecast: its
chance of going through for the side at home in the first leg, read at the first leg as the ties
step publishes it (`vesta.spine.ties`: both legs' matrices, the competition's rules of that season,
extra time, the shoot-out; a single match the same with one matrix), scored by its log loss
against the world's winner. The reference is the training seasons' share of ties that side won
(each outcome counted half a time more), the seasons before the one checked. The difference,
model minus reference, gets the block bootstrap of the discipline, the count of seasons improved
and the calibration of the chance of going through.
"""

import math
import sqlite3
from dataclasses import dataclass

from vesta.research.bootstrap import assign_block, bootstrap_mean
from vesta.research.calibration import measure_calibration_error
from vesta.spine.match_forecast import MatchForecast
from vesta.spine.ties import (
    TieParameters,
    forecast_first_leg_through,
    read_tie_rule,
    resolve_single_match,
)

SMALLEST = 1e-15
SMOOTHING = 0.5


@dataclass(frozen=True)
class TieRecord:
    tie_id: int
    competition: str
    season: int
    first_leg: int
    second_leg: int | None
    winner: int | None
    first_home: int
    kickoff_utc: str
    decided_by: str = ""
    deciding_home: int | None = None  # the home side of the match that decides the tie


@dataclass(frozen=True)
class TiesResult:
    ties: int
    model: float
    reference: float
    difference: tuple[float, float, float]
    seasons_improved: int
    seasons: int
    calibration_model: float
    calibration_reference: float


def read_ties(world: sqlite3.Connection, competitions: list[str]) -> list[TieRecord]:
    """The decided ties of `competitions` with their first leg's home side."""
    marks = ", ".join("?" for _ in competitions)
    rows = world.execute(
        f"""
        SELECT t.tie_id, c.code, s.season, t.first_leg_match_id, t.second_leg_match_id,
               t.winner_team_id, m.home_team_id, m.kickoff_utc, COALESCE(t.decided_by, ''),
               COALESCE(d.home_team_id, m.home_team_id)
        FROM match_ties t
        JOIN matches m ON m.match_id = t.first_leg_match_id
        LEFT JOIN matches d ON d.match_id = t.second_leg_match_id
        JOIN seasons s ON s.season_id = t.season_id
        JOIN competitions c ON c.competition_id = s.competition_id
        WHERE c.code IN ({marks}) AND t.winner_team_id IS NOT NULL
        ORDER BY m.kickoff_utc, t.tie_id
        """,
        competitions,
    ).fetchall()
    return [TieRecord(*row) for row in rows]


def check_ties(ties: list[TieRecord], forecasts: dict[int, MatchForecast], seasons: tuple[int, ...],
               parameters: TieParameters | dict[int, TieParameters], repetitions: int,
               seed: int) -> TiesResult:  # fmt: skip
    """The ties of `seasons` whose matches hold a forecast, against the training seasons'
    reference; `parameters` one set, or one per season forecast (`fitted_parameters`)."""
    rows = []
    for tie in ties:
        if tie.season not in seasons or tie.first_leg not in forecasts:
            continue
        if tie.second_leg is not None and tie.second_leg not in forecasts:
            continue
        chance = forecast_through_chance(
            tie, forecasts, _pick_season_parameters(parameters, tie.season)
        )
        earlier = [other for other in ties if other.season < tie.season]
        won = sum(other.winner == other.first_home for other in earlier)
        reference = (won + SMOOTHING) / (len(earlier) + 2 * SMOOTHING)
        happened = tie.winner == tie.first_home
        rows.append((tie, chance, reference, happened))
    if not rows:
        raise ValueError("no tie of the seasons checked holds the forecasts of its matches")

    def measure_loss(probability: float, happened: bool) -> float:
        return -math.log(max(probability if happened else 1.0 - probability, SMALLEST))

    model = [measure_loss(chance, happened) for _, chance, _, happened in rows]
    reference = [measure_loss(base, happened) for _, _, base, happened in rows]
    differences = [mine - theirs for mine, theirs in zip(model, reference, strict=True)]
    blocks = [assign_block(tie.competition, tie.kickoff_utc) for tie, *_ in rows]
    improved = 0
    by_season = sorted({tie.season for tie, *_ in rows})
    for season in by_season:
        chosen = [diff for (tie, *_), diff in zip(rows, differences, strict=True)
                  if tie.season == season]  # fmt: skip
        improved += sum(chosen) < 0
    return TiesResult(
        ties=len(rows), model=math.fsum(model) / len(rows),
        reference=math.fsum(reference) / len(rows),
        difference=bootstrap_mean(differences, blocks, repetitions, seed),
        seasons_improved=improved, seasons=len(by_season),
        calibration_model=measure_calibration_error(
            [(chance, happened) for _, chance, _, happened in rows]),
        calibration_reference=measure_calibration_error(
            [(base, happened) for _, _, base, happened in rows]),
    )  # fmt: skip


def _pick_season_parameters(parameters, season: int) -> TieParameters:
    return parameters[season] if isinstance(parameters, dict) else parameters


def find_deciding_match(tie: TieRecord) -> int:
    return tie.second_leg if tie.second_leg is not None else tie.first_leg


def read_extra_time_goals(world: sqlite3.Connection) -> dict[int, int]:
    """The goals of extra time (periods 3 and 4) of every match that has any."""
    rows = world.execute(
        "SELECT match_id, COUNT(*) FROM match_events WHERE period IN (3, 4)"
        " AND kind IN ('goal', 'penalty_goal', 'own_goal') GROUP BY match_id"
    ).fetchall()
    return dict(rows)


def fit_tie_parameters(ties: list[TieRecord], forecasts: dict[int, MatchForecast],
                       extra_time_goals: dict[int, int], seasons: tuple[int, ...],
                       base: TieParameters) -> dict[int, TieParameters]:  # fmt: skip
    """Per season forecast, extra time's pace and the shoot-out's home share fitted on the ties
    of the seasons before it (H1): the pace the extra-time goals over the goals the deciding
    match's intensities expect in `extra_time_minutes`, the share the shoot-outs won by the
    deciding match's home side, each outcome counted half a time more."""
    fitted = {}
    for season in seasons:
        earlier = [tie for tie in ties if tie.season < season
                   and tie.decided_by in ("extra_time", "shootout")]  # fmt: skip
        scored = expected = 0.0
        for tie in earlier:
            forecast = forecasts.get(find_deciding_match(tie))
            if forecast is None:
                continue
            scored += extra_time_goals.get(find_deciding_match(tie), 0)
            expected += ((forecast.home_intensity + forecast.away_intensity)
                         * base.extra_time_minutes / 90.0)  # fmt: skip
        shootouts = [tie for tie in earlier if tie.decided_by == "shootout"]
        won = sum(tie.winner == tie.deciding_home for tie in shootouts)
        fitted[season] = TieParameters(
            extra_time_minutes=base.extra_time_minutes,
            extra_time_pace=(scored + SMOOTHING) / (expected + SMOOTHING) if expected else
            base.extra_time_pace,
            shootout_home_share=(won + SMOOTHING) / (len(shootouts) + 2 * SMOOTHING),
        )  # fmt: skip
    return fitted


def forecast_through_chance(tie: TieRecord, forecasts: dict[int, MatchForecast],
                            parameters: TieParameters) -> float:  # fmt: skip
    """The chance the first leg's home side goes through, as the ties step publishes it."""
    first = forecasts[tie.first_leg]
    if tie.second_leg is None:
        outcomes = resolve_single_match(first.matrix, first.home_intensity, first.away_intensity,
                                        parameters)  # fmt: skip
        return sum(outcomes.through["home"].values())
    second = forecasts[tie.second_leg]
    rule = read_tie_rule(tie.competition, tie.season)
    through = forecast_first_leg_through(first.matrix, second.matrix,
                                         (second.home_intensity, second.away_intensity), rule,
                                         parameters)  # fmt: skip
    return through["home"]
