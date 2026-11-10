"""Which seasons a model learns from, which it is judged on, and which no selection may read.

Walk-forward: a season is forecast by a model fitted only on the seasons before it, and every
row it learns from must be known before the season's first forecast. The closed seasons
(`config/vesta/research.toml` [seasons] closed) are never read while a hypothesis is chosen;
they give a clean measure once, by their own command.
"""

from collections.abc import Iterable


class ClosedSeasonError(PermissionError):
    """A closed season asked for by a step that chooses between models."""






def refuse_closed(seasons: Iterable[int], closed: Iterable[int]) -> None:
    """Raise `ClosedSeasonError` when a step that selects asks for a closed season."""
    asked = sorted(set(seasons) & set(closed))
    if asked:
        raise ClosedSeasonError(
            f"seasons {asked} are closed: no selection reads them; they are measured once, by "
            "the clean measure"
        )


