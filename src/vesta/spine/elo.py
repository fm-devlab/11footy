"""The Elo rating shared by the national teams and the club cups.

World Football Elo, used for both: the expected result of the home side, a goal-margin factor,
and the change of the two ratings after a match. The weights, the home advantage and the first
ratings are each model's own (`config/vesta/spine.toml` [nations], [cups]).
"""

ELO_SCALE = 400.0


def compute_expected_result(rating_difference: float) -> float:
    """The home side's expected result (1 a win, 0.5 a draw) for a difference of ratings that
    already includes the home advantage."""
    return 1.0 / (10.0 ** (-rating_difference / ELO_SCALE) + 1.0)


def weigh_margin(goal_difference: int) -> float:
    """1 for a margin of 0 or 1, 1.5 for 2, (11 + n) / 8 for n of 3 or more."""
    margin = abs(goal_difference)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


def compute_rating_change(
    home_rating: float,
    away_rating: float,
    home_goals: int,
    away_goals: int,
    weight: float,
    home_advantage: float,
) -> float:
    """What the home side gains (and the away side loses): weight x margin factor x (result -
    expected result). `home_advantage` is zero on neutral ground."""
    result = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
    expected = compute_expected_result(home_rating - away_rating + home_advantage)
    return weight * weigh_margin(home_goals - away_goals) * (result - expected)
