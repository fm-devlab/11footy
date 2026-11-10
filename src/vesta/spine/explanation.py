"""Why the spine says what it says (deliverable 2d): the explanation of a match's forecast.

Every number of the spine comes from one score matrix, and the matrix from two intensities that
are sums of log terms of the model (`ForecastParts`): the league's level, the home advantage, the
attack and defence of each side, for the team model (Dixon-Coles on our expected goals) and for
the player ratings of the last elevens, blended with `blend_weight` on the ratings, then the
matrix's calibration. The explanation reads only those quantities, nothing else:

- factors: each of the home advantage and the four attack and defence terms (blended), with its
  weight in points, the change of the line's probability when that term alone is set to the
  league's average (zero), the other terms kept; its direction, for or against the line; its note
  says the term as a multiple of the league's average club. One term at a time: the weights do
  not add up to the probability (the terms act together through the matrix);
- chain: the team model's terms and intensities, the ratings' terms and intensities, the blend,
  the calibrated matrix's expected goals and likeliest scores, the line's probability;
- sensitivity: the line in points when the home side's or the away side's expected goals rise by
  a tenth, without the home advantage, with the team model alone, with the player ratings alone;
- data note: a club the team model does not hold (given a promoted club's prior), or a side
  without player ratings (the team model alone).

Written with every pre-match publication, for M1 (the whole distribution and its three outcomes)
and the key lines: three goals or more (M3 `over=2.5`) and both teams score (M4 `yes`).

The cups and the national teams (`explain_elo_match`) read their own model the same way: the
factors are the rating difference and home ground, the chain the two Elo ratings, the Poisson and
the matrix.
"""

import math
from dataclasses import dataclass

import numpy as np

from vesta.forecast_log.writing import Explanation
from vesta.forecast_reading import read_line
from vesta.spine.score_matrix import SIZE, Calibration, score_matrices

TERMS = ("home_advantage", "home_attack", "home_defence", "away_attack", "away_defence")
LABELS = {
    "home_advantage": "home advantage",
    "home_attack": "the home side's attack",
    "home_defence": "the home side's defence",
    "away_attack": "the away side's attack",
    "away_defence": "the away side's defence",
}
EXPLAINED = (
    ("M1.result", "", "ordered"),
    ("M1.result", "outcome=home", "ordered"),
    ("M1.result", "outcome=draw", "ordered"),
    ("M1.result", "outcome=away", "ordered"),
    ("M3.goals", "over=2.5", "count"),
    ("M4.both_score", "yes", "binary"),
)
SAID = {"outcome=home": "a home win", "outcome=draw": "a draw", "outcome=away": "an away win",
        "over=2.5": "three goals or more", "yes": "both teams to score"}  # fmt: skip
EXPECTED_GOALS_STEP = 0.10  # the sensitivity's rise of one side's expected goals


@dataclass(frozen=True)
class ForecastParts:
    """The terms of one match's forecast: the team model's (`DixonColesFamily.parts`), the
    player ratings' (`PlayerRatingsFamily.parts`, None when a side has none), the weight of the
    ratings in the blend, the matrix's calibration."""

    team: dict
    ratings: dict | None
    weight: float
    calibration: Calibration

    def blend_terms(self) -> dict[str, float]:
        weight = self.weight if self.ratings is not None else 0.0
        names = ("level", *TERMS)
        if self.ratings is None:
            return {name: float(self.team[name]) for name in names}
        return {name: (1 - weight) * float(self.team[name]) + weight * float(self.ratings[name])
                for name in names}  # fmt: skip

    def build_matrix(self, terms: dict[str, float]) -> np.ndarray:
        home, away = compute_intensities(terms)
        return score_matrices(np.array([home]), np.array([away]), self.calibration)[0]

    def build_terms(self) -> dict[str, float]:
        return self.blend_terms()

    @property
    def removable(self) -> dict[str, str]:
        return LABELS


def compute_intensities(terms: dict[str, float]) -> tuple[float, float]:
    home = math.exp(terms["level"] + terms["home_advantage"] + terms["home_attack"]
                    - terms["away_defence"])  # fmt: skip
    away = math.exp(terms["level"] + terms["away_attack"] - terms["home_defence"])
    return home, away


def explain_match(parts: ForecastParts, publication_id: int, match_id: int,
                  model_version: str) -> list[Explanation]:  # fmt: skip
    """The explanations of a match's spine forecast, one per explained line."""
    terms = parts.blend_terms()
    matrix = parts.build_matrix(terms)
    explanations = []
    for key, line, kind in EXPLAINED:
        chosen = line or _find_favoured_outcome(matrix)
        probability = _read_matrix_line(matrix, key, chosen, kind)
        factors = _rank_factors(parts, terms, key, chosen, kind, probability)
        sentence = _compose_sentence(chosen, probability, factors, whole=not line)
        explanations.append(Explanation(
            publication_id=publication_id, quantity_key=key, subject_id=match_id,
            model_version=model_version, sentence=sentence, factors=factors,
            chain=_build_chain(parts, terms, matrix, key, chosen, probability),
            sensitivity=_measure_sensitivity(parts, terms, key, chosen, kind, probability),
            line_key=line, data_note=_describe_data_gaps(parts),
        ))  # fmt: skip
    return explanations


def _read_matrix_line(matrix: np.ndarray, key: str, line: str, kind: str) -> float:
    if key == "M1.result":
        values = [["home", float(np.tril(matrix, -1).sum())], ["draw", float(np.trace(matrix))],
                  ["away", float(np.triu(matrix, 1).sum())]]  # fmt: skip
    elif key == "M3.goals":
        total = np.zeros(2 * SIZE - 1)
        for home in range(SIZE):
            total[home : home + SIZE] += matrix[home]
        values = {"probabilities": total.tolist(), "tail_from": len(total), "tail": 0.0}
    else:
        values = {"yes": min(max(float(matrix[1:, 1:].sum()), 0.0), 1.0)}
    return read_line(kind, values, line)


def _find_favoured_outcome(matrix: np.ndarray) -> str:
    shares = {"outcome=home": np.tril(matrix, -1).sum(), "outcome=draw": np.trace(matrix),
              "outcome=away": np.triu(matrix, 1).sum()}  # fmt: skip
    return max(shares, key=shares.get)


def _rank_factors(parts, terms, key, line, kind, probability) -> list[dict]:
    factors = []
    for name in TERMS:
        without = dict(terms, **{name: 0.0})
        effect = probability - _read_matrix_line(parts.build_matrix(without), key, line, kind)
        factors.append({
            "label": LABELS[name],
            "weight_points": round(abs(effect) * 100, 1),
            "direction": "for" if effect >= 0 else "against",
            "note": _describe_term(name, terms[name]),
        })  # fmt: skip
    return sorted(factors, key=lambda factor: -factor["weight_points"])


def _describe_term(name: str, value: float) -> str:
    if name == "home_advantage":
        return f"the home side scores {math.exp(value):.2f} times what it would on neutral ground"
    side = "the home side" if name.startswith("home") else "the away side"
    if name.endswith("attack"):
        return f"{side} scores {math.exp(value):.2f} times the league's average club"
    return f"{side} concedes {math.exp(-value):.2f} times the league's average club"


def _compose_sentence(line: str, probability: float, factors: list[dict], whole: bool) -> str:
    opening = (f"Vesta's likeliest outcome is {SAID[line]}, at {probability:.0%}" if whole
               else f"Vesta gives {SAID[line]} {probability:.0%}")  # fmt: skip
    top = [factor for factor in factors if factor["weight_points"] > 0][:2]
    if not top:
        return opening + "."
    reasons = " and ".join(f"{factor['label']} ({factor['direction']} it, "
                           f"{factor['weight_points']:g} points)" for factor in top)  # fmt: skip
    return f"{opening}: the largest factors are {reasons}." if len(top) > 1 else (
        f"{opening}: the largest factor is {reasons}.")  # fmt: skip


def _build_chain(parts, terms, matrix, key, line, probability) -> list[dict]:
    chain = [{"step": 1, "label": "team model (Dixon-Coles on our expected goals)",
              "values": _round_terms_and_intensities(parts.team)}]  # fmt: skip
    if parts.ratings is not None:
        chain.append({"step": 2, "label": "player ratings of the last elevens",
                      "values": _round_terms_and_intensities(parts.ratings)})  # fmt: skip
    home, away = compute_intensities(terms)
    chain.append({"step": len(chain) + 1, "label": "blend of the two",
                  "values": {"weight_on_ratings": parts.weight if parts.ratings else 0.0,
                             "home_intensity": round(home, 4),
                             "away_intensity": round(away, 4)}})  # fmt: skip
    goals = np.arange(SIZE)
    likeliest = np.argsort(-matrix.ravel(), kind="stable")[:3]
    chain.append({"step": len(chain) + 1, "label": "calibrated score matrix",
                  "values": {"home_expected_goals": round(float(matrix.sum(axis=1) @ goals), 3),
                             "away_expected_goals": round(float(matrix.sum(axis=0) @ goals), 3),
                             "likeliest_scores": [[f"{cell // SIZE}-{cell % SIZE}",
                                                   round(float(matrix.ravel()[cell]), 4)]
                                                  for cell in likeliest]}})  # fmt: skip
    chain.append({"step": len(chain) + 1, "label": f"{key} {line}",
                  "values": {"probability": round(probability, 4)}})  # fmt: skip
    return chain


def _round_terms_and_intensities(source: dict) -> dict:
    terms = {name: float(source[name]) for name in ("level", *TERMS)}
    home, away = compute_intensities(terms)
    return {**{name: round(value, 4) for name, value in terms.items()},
            "home_intensity": round(home, 4), "away_intensity": round(away, 4)}  # fmt: skip


def _measure_sensitivity(parts, terms, key, line, kind, probability) -> list[dict]:
    changes = [
        ("the home side's expected goals a tenth higher",
         dict(terms, home_attack=terms["home_attack"] + math.log1p(EXPECTED_GOALS_STEP))),
        ("the away side's expected goals a tenth higher",
         dict(terms, away_attack=terms["away_attack"] + math.log1p(EXPECTED_GOALS_STEP))),
        ("no home advantage", dict(terms, home_advantage=0.0)),
    ]  # fmt: skip
    if parts.ratings is not None:
        for label, weight in (("the team model alone", 0.0), ("the player ratings alone", 1.0)):
            other = ForecastParts(parts.team, parts.ratings, weight, parts.calibration)
            changes.append((label, other.blend_terms()))
    return [{"change": label,
             "effect_points": round((_read_matrix_line(parts.build_matrix(changed), key, line, kind)
                                     - probability) * 100, 1)}
            for label, changed in changes]  # fmt: skip


def _describe_data_gaps(parts: ForecastParts) -> str | None:
    notes = []
    for side in ("home", "away"):
        if not parts.team.get(f"{side}_known", True):
            notes.append(f"the {side} side is new to the league: the team model gives it the "
                         "attack and defence of a promoted club")  # fmt: skip
    if parts.ratings is None:
        notes.append("a side has no player ratings of its last eleven: the team model alone")
    return "; ".join(notes) or None


@dataclass(frozen=True)
class EloParts:
    """The terms of a cup or national match's forecast (`vesta.spine.cups`, `vesta.spine.nations`):
    the two Elo ratings, the mapping's level (with the qualifying round's for a cup), its slope on
    the rating difference, its venue weight, the venue (+1 home ground, 0 neutral), the
    calibration; `notes` what the data note says (a club rated from its country's mean, a team
    with few matches)."""

    home_rating: float
    away_rating: float
    level: float
    slope: float
    venue_weight: float
    venue: float
    calibration: Calibration
    notes: tuple[str, ...] = ()

    @property
    def removable(self) -> dict[str, str]:
        return ELO_LABELS

    def build_terms(self) -> dict[str, float]:
        return {"level": self.level,
                "rating_difference": self.slope * (self.home_rating - self.away_rating) / 400.0,
                "home_ground": self.venue_weight * self.venue}  # fmt: skip

    def build_matrix(self, terms: dict[str, float]) -> np.ndarray:
        home = math.exp(terms["level"] + terms["rating_difference"] + terms["home_ground"])
        away = math.exp(terms["level"] - terms["rating_difference"] - terms["home_ground"])
        return score_matrices(np.array([home]), np.array([away]), self.calibration)[0]


ELO_LABELS = {"rating_difference": "the rating difference", "home_ground": "home ground"}


def explain_elo_match(parts: EloParts, publication_id: int, match_id: int,
                      model_version: str) -> list[Explanation]:  # fmt: skip
    """The explanations of a cup or national match's forecast, one per explained line: the
    factors are the rating difference and home ground, each set to nothing (equal ratings,
    neutral ground) one at a time."""
    terms = parts.build_terms()
    matrix = parts.build_matrix(terms)
    explanations = []
    for key, line, kind in EXPLAINED:
        chosen = line or _find_favoured_outcome(matrix)
        probability = _read_matrix_line(matrix, key, chosen, kind)
        factors = []
        for name, label in ELO_LABELS.items():
            effect = probability - _read_matrix_line(
                parts.build_matrix(dict(terms, **{name: 0.0})), key, chosen, kind
            )
            factors.append({"label": label, "weight_points": round(abs(effect) * 100, 1),
                            "direction": "for" if effect >= 0 else "against",
                            "note": _describe_elo_term(name, parts)})  # fmt: skip
        factors.sort(key=lambda factor: -factor["weight_points"])
        changes = [("the home side rated 50 points higher",
                    dict(terms, rating_difference=terms["rating_difference"]
                         + parts.slope * 50 / 400.0)),
                   ("the away side rated 50 points higher",
                    dict(terms, rating_difference=terms["rating_difference"]
                         - parts.slope * 50 / 400.0))]  # fmt: skip
        if parts.venue:
            changes.append(("on neutral ground", dict(terms, home_ground=0.0)))
        explanations.append(
            Explanation(
                publication_id=publication_id,
                quantity_key=key,
                subject_id=match_id,
                model_version=model_version,
                sentence=_compose_sentence(chosen, probability, factors, whole=not line),
                factors=factors,
                chain=_build_elo_chain(parts, terms, matrix, key, chosen, probability),
                sensitivity=[
                    {
                        "change": label,
                        "effect_points": round(
                            (
                                _read_matrix_line(parts.build_matrix(changed), key, chosen, kind)
                                - probability
                            )
                            * 100,
                            1,
                        ),
                    }
                    for label, changed in changes
                ],
                line_key=line,
                data_note="; ".join(parts.notes) or None,
            )
        )
    return explanations


def _describe_elo_term(name: str, parts: EloParts) -> str:
    if name == "home_ground":
        return ("the home side plays at home" if parts.venue else
                "on neutral ground: no home advantage")  # fmt: skip
    gap = parts.home_rating - parts.away_rating
    return (f"the home side is rated {parts.home_rating:.0f}, the away side "
            f"{parts.away_rating:.0f} ({abs(gap):.0f} points {'more' if gap >= 0 else 'fewer'}"
            " for the home side)")  # fmt: skip


def _build_elo_chain(parts, terms, matrix, key, line, probability) -> list[dict]:
    goals = np.arange(SIZE)
    likeliest = np.argsort(-matrix.ravel(), kind="stable")[:3]
    home = math.exp(terms["level"] + terms["rating_difference"] + terms["home_ground"])
    away = math.exp(terms["level"] - terms["rating_difference"] - terms["home_ground"])
    return [
        {"step": 1, "label": "Elo ratings",
         "values": {"home_rating": round(parts.home_rating, 1),
                    "away_rating": round(parts.away_rating, 1)}},
        {"step": 2, "label": "Poisson on the rating difference and the venue",
         "values": {"level": round(parts.level, 4), "slope": round(parts.slope, 4),
                    "venue_weight": round(parts.venue_weight, 4), "venue": parts.venue,
                    "home_intensity": round(home, 4), "away_intensity": round(away, 4)}},
        {"step": 3, "label": "calibrated score matrix",
         "values": {"home_expected_goals": round(float(matrix.sum(axis=1) @ goals), 3),
                    "away_expected_goals": round(float(matrix.sum(axis=0) @ goals), 3),
                    "likeliest_scores": [[f"{cell // SIZE}-{cell % SIZE}",
                                          round(float(matrix.ravel()[cell]), 4)]
                                         for cell in likeliest]}},
        {"step": 4, "label": f"{key} {line}", "values": {"probability": round(probability, 4)}},
    ]  # fmt: skip
