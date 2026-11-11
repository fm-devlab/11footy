"""The expected-goals model: the shot's facts, the boosted trees, the penalty rate.

The chance of a shot outside the penalties is a gradient-boosted model of fourteen facts of the shot
and its move; a penalty's chance is the share of penalties scored in the training shots. The
inputs are built as numbers: the three categories (situation, body part, the action before) as
their index in a fixed list, so the trees always see the same codes.

LightGBM is imported only where a model is fitted: a user's run reads the dataset's own expected
goals and never fits one, and on macOS LightGBM needs the system's OpenMP (`brew install libomp`).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from vesta.expected_goals.shots import ShotTable

if TYPE_CHECKING:
    import lightgbm

PITCH_LENGTH_METRES, PITCH_WIDTH_METRES, GOAL_WIDTH_METRES = 105.0, 68.0, 7.32
SITUATIONS = ("open_play", "corner", "set_piece", "direct_free_kick", "penalty")
BODY_PARTS = ("right_foot", "left_foot", "head", "other")
ACTIONS = ("Pass", "Cross", "Aerial", "Standard", "TakeOn", "Chipped", "Rebound", "BallRecovery",
           "HeadPass", "Throughball", "BallTouch", "LayOff", "Dispossessed", "Tackle", "Foul",
           "CornerAwarded", "BlockedPass", "Interception", "Other")  # fmt: skip
FEATURES = ("distance", "angle", "x", "y_off", "situation", "body", "last_action", "minute",
            "state", "home", "assisted", "gap", "after_stop", "same_minute")  # fmt: skip
CATEGORICAL = (4, 5, 6)


@dataclass(frozen=True)
class ModelSettings:
    """`[model]` of `config/vesta/expected_goals.toml`."""

    learning_rate: float
    leaves_per_tree: int
    min_child_samples: int
    subsample: float
    subsample_freq: int
    colsample_bytree: float
    reg_lambda: float
    most_trees: int
    early_stopping_trees: int
    trees_without_validation: int
    penalty_rate_without_penalties: float
    seed: int

    @classmethod
    def from_config(cls, expected_goals: dict) -> ModelSettings:
        return cls(**expected_goals["model"])

    def to_booster_parameters(self) -> dict:
        """The parameters in LightGBM's own names (the same values as the scikit-learn
        names, which LightGBM maps to these)."""
        return {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": self.learning_rate,
            "num_leaves": self.leaves_per_tree,
            "min_data_in_leaf": self.min_child_samples,
            "bagging_fraction": self.subsample,
            "bagging_freq": self.subsample_freq,
            "feature_fraction": self.colsample_bytree,
            "lambda_l2": self.reg_lambda,
            "seed": self.seed,
            "num_threads": 1,
            "deterministic": True,
            "force_row_wise": True,
            "verbose": -1,
        }


@dataclass(frozen=True)
class FittedModel:
    booster: lightgbm.Booster
    penalty_rate: float
    shots: int
    trees: int

    def rate(self, shots: ShotTable) -> np.ndarray:
        """The chance of each shot: the trees outside penalties, the rate for a penalty."""
        chances = self.booster.predict(build_inputs(shots), num_iteration=self.trees)
        return np.where(shots.situation == "penalty", self.penalty_rate, chances)


def build_inputs(shots: ShotTable) -> np.ndarray:
    """The fourteen inputs of every shot, as a matrix of numbers (columns of `FEATURES`)."""
    along = (1.0 - shots.x) * PITCH_LENGTH_METRES
    across = (shots.y - 0.5) * PITCH_WIDTH_METRES
    distance = np.hypot(along, across)
    opening = np.arctan2(
        GOAL_WIDTH_METRES * along, along * along + across * across - (GOAL_WIDTH_METRES / 2) ** 2
    )  # the angle the goal mouth opens from the shot
    angle = np.where(opening < 0, opening + np.pi, opening)
    return np.column_stack(
        [
            distance,
            angle,
            shots.x,
            np.abs(shots.y - 0.5),
            _encode_categories(shots.situation, SITUATIONS, "open_play"),
            _encode_categories(shots.body_part, BODY_PARTS, "other"),
            _encode_categories(shots.last_action, ACTIONS, "Other"),
            np.clip(shots.minute, 1, 100),
            np.clip(shots.state, -3, 3),
            shots.home,
            shots.assisted,
            np.clip(shots.gap, 0, 10),
            shots.after_stop,
            np.clip(shots.same_minute, 1, 5),
        ]
    ).astype("float64")


def fit(train: ShotTable, validation: ShotTable | None, settings: ModelSettings) -> FittedModel:
    """A model fitted on `train`. With `validation`, the number of trees is the one where the
    validation shots stop improving, and the model is refitted on train and validation together
    with that number; without, `trees_without_validation`."""
    lightgbm = _load_lightgbm()
    penalties = train.situation == "penalty"
    penalty_rate = (
        float(train.goal[penalties].mean())
        if penalties.any()
        else settings.penalty_rate_without_penalties
    )
    open_play = train.select(~penalties)
    if validation is not None and len(validation):
        checked = validation.select(validation.situation != "penalty")
        training = _make_dataset(open_play)
        probe = lightgbm.train(
            settings.to_booster_parameters(),
            training,
            num_boost_round=settings.most_trees,
            valid_sets=[_make_dataset(checked, reference=training)],
            callbacks=[lightgbm.early_stopping(settings.early_stopping_trees, verbose=False)],
        )
        trees = int(probe.best_iteration or settings.most_trees)
        whole = _join_tables(open_play, checked)
    else:
        trees, whole = settings.trees_without_validation, open_play
    final = lightgbm.train(
        settings.to_booster_parameters(), _make_dataset(whole), num_boost_round=trees
    )
    return FittedModel(final, penalty_rate, len(whole), trees)


def _load_lightgbm():
    return importlib.import_module("lightgbm")


def _make_dataset(shots: ShotTable, reference: lightgbm.Dataset | None = None) -> lightgbm.Dataset:
    return _load_lightgbm().Dataset(
        build_inputs(shots),
        label=shots.goal,
        feature_name=list(FEATURES),
        categorical_feature=list(CATEGORICAL),
        reference=reference,
        free_raw_data=False,
    )


def _encode_categories(
    values: np.ndarray, categories: tuple[str, ...], otherwise: str
) -> np.ndarray:
    """The index of each value in `categories`; a value outside it counts as `otherwise`."""
    index = {category: position for position, category in enumerate(categories)}
    fallback = index[otherwise]
    return np.array([index.get(value, fallback) for value in values], dtype="float64")


def _join_tables(first: ShotTable, second: ShotTable) -> ShotTable:
    return ShotTable(
        **{
            name: np.concatenate([getattr(first, name), getattr(second, name)])
            for name in first.__dataclass_fields__
        }
    )
