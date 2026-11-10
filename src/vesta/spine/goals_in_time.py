"""Goals in time: every match completed from its score matrix, so the halves, the timing and the
manner of the goals agree with the final score (the family plan's step 3).

For every score i-j the population holds max(minimum, ceil(N P(i,j)))
simulated matches that end i-j, each weighing P(i,j) divided by their number: a quantity that
depends only on the final score equals the matrix exactly. Each simulated goal then:
- falls in the first half with the first-half share of its side (else in the second), uniformly
  on the playing time of its half, added time included; the added time of each half is a gamma of
  shape `added_time_shape` and mean `added_time_minutes`; a goal is reported as the match record
  does, minutes 1-45 and 46-90, a goal in added time at 45 or 90;
- has a manner: own goal, penalty, direct free kick, header, from outside
  the box, or from open play inside the box, drawn with the shares of the training goals.
Each match draws from its own seeded streams, so a forecast is reproduced to the bit.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from vesta.spine.score_matrix import SIZE

HALF_MINUTES = 45.0
SLOTS = SIZE  # the most goals of one side in a cell (the matrix stops at ten)
METHODS = (
    "penalty", "own_goal", "direct_free_kick", "header", "outside_box", "open_play_inside_box",
)  # fmt: skip
# the identity of each component's random numbers, never reused
TIMES_STREAM, METHODS_STREAM = 1, 2
SHARE_BANDS = (1.0, 1.8)  # H3: a side's expected goals below 1.0, 1.0 to 1.8, above 1.8


@dataclass(frozen=True)
class GoalsInTimeParameters:
    """`[goals_in_time]` of `config/vesta/spine.toml`."""

    simulations: int
    minimum_per_score: int
    seed: int
    added_time_minutes: tuple[float, float]
    added_time_shape: float
    first_half_share_without_data: float

    @classmethod
    def from_config(cls, spine: dict) -> "GoalsInTimeParameters":
        section = spine["goals_in_time"]
        return cls(
            simulations=int(section["simulations"]),
            minimum_per_score=int(section["minimum_per_score"]),
            seed=int(section["seed"]),
            added_time_minutes=tuple(float(value) for value in section["added_time_minutes"]),
            added_time_shape=float(section["added_time_shape"]),
            first_half_share_without_data=float(section["first_half_share_without_data"]),
        )


@dataclass(frozen=True)
class TimeShares:
    """What the training matches say: each side's share of goals in the first half, and the
    shares of the manners of a goal (in the order of `METHODS`)."""

    first_half_home: float
    first_half_away: float
    methods: tuple[float, ...]
    # candidates (H2, H3), absent from the published model: the weight of each regular minute of
    # each half (two tuples of 45) for the minute within the half; each side's first-half share
    # in the bands of its expected goals (`SHARE_BANDS`), home and away
    regular_minutes: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    shares_by_band: tuple[tuple[float, ...], tuple[float, ...]] | None = None


def fit_first_half_shares(
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_half_time: np.ndarray,
    away_half_time: np.ndarray,
    without_data: float,
) -> tuple[float, float]:
    """The maximum-likelihood share of each side's goals in the first half (its first-half goals
    are binomial given its goals): first-half goals over goals, on the matches whose half-time
    score fits inside the final score."""
    usable = (home_half_time <= home_goals) & (away_half_time <= away_goals)
    shares = []
    for goals, first_half in ((home_goals, home_half_time), (away_goals, away_half_time)):
        total = float(goals[usable].sum())
        shares.append(float(first_half[usable].sum()) / total if total > 0 else without_data)
    return shares[0], shares[1]


def fit_method_shares(methods: list[str]) -> tuple[float, ...]:
    """The share of each manner among the training goals."""
    counts = [sum(1 for method in methods if method == name) for name in METHODS]
    if sum(counts) == 0:
        raise ValueError("no training goal to read the manners of goals from")
    return tuple(count / sum(counts) for count in counts)


@dataclass(frozen=True)
class Population:
    """The simulated matches of one match: their final score, weight, and each side's goals in
    time order (`*_time` minutes from kick-off, nan where no goal; `*_minute` as reported, 0 where
    no goal; `*_half` 1 or 2, 0 where no goal; `*_method` an index of `METHODS`, -1 where no
    goal)."""

    home_goals: np.ndarray
    away_goals: np.ndarray
    weight: np.ndarray
    home_time: np.ndarray
    away_time: np.ndarray
    home_minute: np.ndarray
    away_minute: np.ndarray
    home_half: np.ndarray
    away_half: np.ndarray
    home_method: np.ndarray
    away_method: np.ndarray


def simulate(
    matrix: np.ndarray, shares: TimeShares, parameters: GoalsInTimeParameters, match_id: int
) -> Population:
    """The population of one match from its matrix (rows home goals, columns away goals)."""
    probability = np.asarray(matrix, dtype="float64").reshape(-1)
    if abs(math.fsum(probability.tolist()) - 1.0) > 1e-9:
        raise ValueError(f"match {match_id}: the score matrix does not sum to one")
    wanted = np.ceil(parameters.simulations * probability)
    per_score = np.maximum(parameters.minimum_per_score, wanted).astype("int64")
    cell = np.repeat(np.arange(SIZE * SIZE), per_score)
    home_goals, away_goals = cell // SIZE, cell % SIZE
    weight = (probability / per_score)[cell]
    if shares.shares_by_band is not None:  # H3: each side's share by its expected goals
        matrix = probability.reshape(SIZE, SIZE)
        expected = (matrix.sum(axis=1) @ np.arange(SIZE), matrix.sum(axis=0) @ np.arange(SIZE))
        bands = [int(np.searchsorted(SHARE_BANDS, value, side="right")) for value in expected]
        shares = replace(shares, first_half_home=shares.shares_by_band[0][bands[0]],
                         first_half_away=shares.shares_by_band[1][bands[1]])  # fmt: skip
    times = _simulate_goal_times(home_goals, away_goals, shares, parameters, match_id)
    methods = _simulate_goal_methods(home_goals, away_goals, shares, parameters, match_id)
    return Population(home_goals, away_goals, weight, *times, *methods)


def _simulate_goal_times(home_goals, away_goals, shares, parameters, match_id):
    generator = np.random.default_rng([parameters.seed, match_id, TIMES_STREAM])
    count = len(home_goals)
    shape = parameters.added_time_shape
    added = np.stack([generator.gamma(shape, mean / shape, count)
                      for mean in parameters.added_time_minutes], axis=1)  # fmt: skip
    slots = np.arange(SLOTS)[None, :]
    arrays = {}
    for side, goals, share in (("home", home_goals, shares.first_half_home),
                               ("away", away_goals, shares.first_half_away)):  # fmt: skip
        # every slot draws, used or not, so the streams never depend on the score
        first_half = generator.random((count, SLOTS)) < share
        position = generator.random((count, SLOTS))
        used = slots < goals[:, None]
        length = np.where(first_half, HALF_MINUTES + added[:, :1], HALF_MINUTES + added[:, 1:])
        into_half = position * length
        if shares.regular_minutes is not None:  # H2: the regular minutes by the profile
            into_half = _place_by_profile(into_half, first_half, shares.regular_minutes)
        time = np.where(first_half, into_half, HALF_MINUTES + added[:, :1] + into_half)
        minute = (np.clip(np.ceil(into_half), 1, HALF_MINUTES)
                  + np.where(first_half, 0.0, HALF_MINUTES))  # fmt: skip
        order = np.argsort(np.where(used, time, np.inf), axis=1, kind="stable")
        used = np.take_along_axis(used, order, axis=1)
        arrays[side] = (
            np.where(used, np.take_along_axis(time, order, axis=1), np.nan),
            np.where(used, np.take_along_axis(minute, order, axis=1), 0).astype("int16"),
            np.where(used, np.take_along_axis(np.where(first_half, 1, 2), order, axis=1),
                     0).astype("int8"),
        )  # fmt: skip
    return (arrays["home"][0], arrays["away"][0], arrays["home"][1], arrays["away"][1],
            arrays["home"][2], arrays["away"][2])  # fmt: skip


def _place_by_profile(into_half: np.ndarray, first_half: np.ndarray,
                      regular_minutes: tuple[tuple[float, ...], ...]) -> np.ndarray:  # fmt: skip
    """A goal in the regular part of its half moved to the minute the profile gives at the same
    quantile (the uniform draw's share of the 45 minutes read through the profile's cumulative
    weight); a goal in added time stays where it is."""
    moved = into_half.copy()
    for half, weights in ((True, regular_minutes[0]), (False, regular_minutes[1])):
        cumulative = np.r_[0.0, np.cumsum(weights)] / np.sum(weights)
        regular = (first_half == half) & (into_half < HALF_MINUTES)
        quantile = into_half[regular] / HALF_MINUTES
        moved[regular] = np.interp(quantile, cumulative, np.arange(len(cumulative)))
    return moved


def fit_shares_by_band(home_goals, away_goals, home_half_time, away_half_time, home_expected,
                       away_expected, without_data: float) -> tuple[tuple[float, ...], ...]:
    """H3: each side's first-half share in each band of its own expected goals (`SHARE_BANDS`),
    by the same maximum likelihood as `fit_first_half_shares`."""
    shares = []
    for goals, half_time, expected in ((home_goals, home_half_time, home_expected),
                                       (away_goals, away_half_time, away_expected)):  # fmt: skip
        band = np.searchsorted(SHARE_BANDS, expected, side="right")
        side = []
        for index in range(len(SHARE_BANDS) + 1):
            chosen = (band == index) & (half_time <= goals)
            total = float(goals[chosen].sum())
            side.append(float(half_time[chosen].sum()) / total if total > 0 else without_data)
        shares.append(tuple(side))
    return shares[0], shares[1]


def uses_minute_profile(spine: dict) -> bool:
    """Whether the published model places a goal within its half by the live chain's profile
    (`[goals_in_time] minute_weights = "chain_profile"`, H2 kept on 26 October 2026) rather
    than evenly."""
    return spine["goals_in_time"].get("minute_weights") == "chain_profile"


def split_regular_minutes(profile: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """H2: the weight of each regular minute of each half, read from the live chain's profile
    (its slots 1-45 and 46-90)."""
    return (tuple(float(value) for value in profile[:45]),
            tuple(float(value) for value in profile[55:100]))  # fmt: skip


def _simulate_goal_methods(home_goals, away_goals, shares, parameters, match_id):
    generator = np.random.default_rng([parameters.seed, match_id, METHODS_STREAM])
    count = len(home_goals)
    slots = np.arange(SLOTS)[None, :]
    cumulative = np.cumsum(shares.methods)
    methods = []
    for goals in (home_goals, away_goals):
        drawn = np.searchsorted(cumulative, generator.random((count, SLOTS)) * cumulative[-1],
                                side="right")  # fmt: skip
        methods.append(np.where(slots < goals[:, None], np.minimum(drawn, len(METHODS) - 1), -1))
    return methods[0].astype("int8"), methods[1].astype("int8")
