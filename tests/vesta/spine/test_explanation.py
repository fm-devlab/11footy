"""The explanation of a match's spine forecast (2d): factors that are the model's own terms, the
chain from the terms to the line, the sensitivity, the data note; written into the log."""

import numpy as np
import pytest

from vesta.forecast_log.store import open_log
from vesta.forecast_log.writing import (
    Publication,
    publish,
    record_explanation,
    register_model_version,
    register_quantity,
)
from vesta.forecast_reading import read_line
from vesta.quantities import load_quantities
from vesta.spine.explanation import ForecastParts, explain_match
from vesta.spine.publication import build_spine_distributions
from vesta.spine.score_matrix import Calibration

TEAM = {"level": 0.2, "home_advantage": 0.15, "home_attack": 0.3, "home_defence": 0.1,
        "away_attack": -0.1, "away_defence": -0.05, "home_known": True, "away_known": True}
RATINGS = {"level": 0.18, "home_advantage": 0.12, "home_attack": 0.2, "home_defence": 0.05,
           "away_attack": 0.0, "away_defence": 0.0}  # fmt: skip
CALIBRATION = Calibration(-0.05, 0.08, 1.0, 1.0, 0.0, 1.0)
PARTS = ForecastParts(TEAM, RATINGS, 0.5, CALIBRATION)


def _by_line(explanations):
    return {(item.quantity_key, item.line_key): item for item in explanations}


def test_the_chain_ends_on_the_spines_own_line():
    explained = _by_line(explain_match(PARTS, 1, 900, "spine-test"))
    definitions = load_quantities()
    published = {item.quantity_key: item for item in build_spine_distributions(
        PARTS.build_matrix(PARTS.blend_terms()), 900, 11, 12, "spine-test")}  # fmt: skip
    for (key, line), item in explained.items():
        if not line:
            continue
        kind = definitions[key].quantity.distribution_kind
        assert item.chain[-1]["values"]["probability"] == pytest.approx(
            read_line(kind, published[key].values, line), abs=1e-4)
    home = explained[("M1.result", "outcome=home")]
    blended = home.chain[2]["values"]
    assert blended["weight_on_ratings"] == 0.5
    assert home.chain[0]["label"].startswith("team model")


def test_the_factors_are_the_models_terms_with_their_weight_in_points():
    home = _by_line(explain_match(PARTS, 1, 900, "spine-test"))[("M1.result", "outcome=home")]
    labels = [factor["label"] for factor in home.factors]
    assert sorted(labels) == sorted(["home advantage", "the home side's attack",
                                     "the home side's defence", "the away side's attack",
                                     "the away side's defence"])  # fmt: skip
    by_label = {factor["label"]: factor for factor in home.factors}
    assert by_label["the home side's attack"]["direction"] == "for"  # a strong home attack
    assert by_label["home advantage"]["direction"] == "for"
    assert by_label["the away side's defence"]["direction"] == "for"  # a weak away defence
    weights = [factor["weight_points"] for factor in home.factors]
    assert weights == sorted(weights, reverse=True) and weights[0] > 0
    assert home.sentence.startswith("Vesta gives a home win")
    assert labels[0] in home.sentence
    assert "1.28 times the league's average club" in by_label["the home side's attack"]["note"]
    # a team of the league's average in every term: no factor weighs anything
    plain = ForecastParts(dict(TEAM, **dict.fromkeys(("home_advantage", "home_attack",
                                                       "home_defence", "away_attack",
                                                       "away_defence"), 0.0)),
                          None, 0.5, CALIBRATION)  # fmt: skip
    flat = _by_line(explain_match(plain, 1, 900, "spine-test"))[("M1.result", "outcome=home")]
    assert all(factor["weight_points"] == 0.0 for factor in flat.factors)


def test_the_sensitivity_and_the_data_note_say_what_moves_and_what_is_missing():
    home = _by_line(explain_match(PARTS, 1, 900, "spine-test"))[("M1.result", "outcome=home")]
    moves = {change["change"]: change["effect_points"] for change in home.sensitivity}
    assert moves["the home side's expected goals a tenth higher"] > 0
    assert moves["the away side's expected goals a tenth higher"] < 0
    assert moves["no home advantage"] < 0
    assert {"the team model alone", "the player ratings alone"} <= set(moves)
    assert home.data_note is None
    promoted = ForecastParts(dict(TEAM, away_known=False), None, 0.5, CALIBRATION)
    note = explain_match(promoted, 1, 900, "spine-test")[0].data_note
    assert "away side is new to the league" in note and "team model alone" in note
    whole = _by_line(explain_match(PARTS, 1, 900, "spine-test"))[("M1.result", "")]
    assert whole.sentence.startswith("Vesta's likeliest outcome is a home win")


def test_the_explanations_are_written_beside_their_publication(tmp_path):
    log = open_log(tmp_path / "forecast-log.db")
    register_model_version(log, "spine-test", "spine", "2026-10-24T00:00:00Z", "test")
    definitions = load_quantities()
    for key in ("M1.result", "M2.score", "M3.goals", "M4.both_score", "M5.goals.team"):
        register_quantity(log, definitions[key].quantity)
    publication = Publication(stage="pre_match", trigger="first",
                              published_at="2026-10-24T08:00:00Z", code_commit="abc1234",
                              match_id=900,
                              scheduled_kickoff_utc="2026-10-25T15:00:00Z")  # fmt: skip
    publication_id = publish(log, publication, build_spine_distributions(
        PARTS.build_matrix(PARTS.blend_terms()), 900, 11, 12, "spine-test"))  # fmt: skip
    for explanation in explain_match(PARTS, publication_id, 900, "spine-test"):
        record_explanation(log, explanation)
    assert log.execute("SELECT COUNT(*) FROM explanations").fetchone()[0] == 6
    assert np.isfinite(log.execute("SELECT COUNT(*) FROM explanations WHERE line_key = ''"
                                   ).fetchone()[0])  # fmt: skip
    log.close()


def test_a_cup_or_national_match_is_explained_by_its_ratings_and_its_ground():
    from vesta.spine.explanation import EloParts, explain_elo_match

    parts = EloParts(1834.0, 1712.0, 0.25, 0.9, 0.2, 1.0, CALIBRATION,
                     ("the away side is rated from its country's mean",))  # fmt: skip
    home = _by_line(explain_elo_match(parts, 1, 901, "cups-test"))[("M1.result", "outcome=home")]
    by_label = {factor["label"]: factor for factor in home.factors}
    assert set(by_label) == {"the rating difference", "home ground"}
    assert by_label["the rating difference"]["direction"] == "for"
    assert "rated 1834" in by_label["the rating difference"]["note"]
    assert {change["change"] for change in home.sensitivity} == {
        "the home side rated 50 points higher", "the away side rated 50 points higher",
        "on neutral ground"}
    assert home.data_note == "the away side is rated from its country's mean"
    neutral = EloParts(1834.0, 1712.0, 0.25, 0.9, 0.2, 0.0, CALIBRATION)
    away = _by_line(explain_elo_match(neutral, 1, 901, "nations-test"))[("M1.result",
                                                                         "outcome=away")]
    ground = {factor["label"]: factor for factor in away.factors}["home ground"]
    assert ground["weight_points"] == 0.0 and "neutral ground" in ground["note"]
    assert away.data_note is None

