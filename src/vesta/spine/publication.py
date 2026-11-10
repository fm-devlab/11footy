"""What the spine publishes for a match, read from its one score matrix.

Every quantity of the spine (`config/vesta/quantities.csv`, family member `spine`) is a sum of
cells of the same matrix, so the numbers of a match agree by construction: the result (M1, with
double chance as its lines), the score (M2, with the exact scores and the winning margins), the
goals of the match (M3), both teams score (M4), the goals of each team (M5). The matrix stops at
ten goals a side with the mass beyond folded into the tenth (`score_matrix`), and the counts read
from it stop there too, so every published tail is zero.
"""

import numpy as np

from vesta.forecast_log.writing import PublishedDistribution
from vesta.spine.score_matrix import SIZE, format_published_matrix


def build_spine_distributions(
    matrix: np.ndarray, match_id: int, home_team_id: int, away_team_id: int, model_version: str
) -> list[PublishedDistribution]:
    """The spine's distributions of one match from its matrix (11 x 11, rows home goals)."""
    home_goals = matrix.sum(axis=1)
    away_goals = matrix.sum(axis=0)
    total = np.array(
        [np.trace(np.fliplr(matrix), offset=SIZE - 1 - goals) for goals in range(2 * SIZE - 1)]
    )
    result = [
        ["home", float(np.tril(matrix, -1).sum())],
        ["draw", float(np.trace(matrix))],
        ["away", float(np.triu(matrix, 1).sum())],
    ]
    return [
        PublishedDistribution("M1.result", match_id, model_version, result),
        PublishedDistribution("M2.score", match_id, model_version,
                              format_published_matrix(matrix)),  # fmt: skip
        PublishedDistribution("M3.goals", match_id, model_version, _count(total)),
        PublishedDistribution("M4.both_score", match_id, model_version,
                              {"yes": _clip_probability(float(matrix[1:, 1:].sum()))}),  # fmt: skip
        PublishedDistribution("M5.goals.team", home_team_id, model_version, _count(home_goals)),
        PublishedDistribution("M5.goals.team", away_team_id, model_version, _count(away_goals)),
    ]


def _count(probabilities: np.ndarray) -> dict:
    return {"probabilities": probabilities.tolist(), "tail_from": len(probabilities), "tail": 0.0}


def _clip_probability(probability: float) -> float:
    """A sum of cells can stray beyond [0, 1] by the last bit."""
    return min(max(probability, 0.0), 1.0)
