"""The order of importance of our competitions (`config/competition-order.csv`): the window
lists our dataset's competitions in it, and only those the file shows:

- `always`: listed while it has a match in the last four years (`desk.reading.RECENT_REACH`);
- `around_matches`: a tournament, qualifying or friendly played only at times, listed only from
  90 days before one of its matches to 30 days after (`desk.reading.BEFORE_MATCHES`,
  `AFTER_MATCHES`);
- `no`, or a code the file does not name: never listed for our dataset.
"""

import csv
import unicodedata
from functools import cache
from pathlib import Path

ORDER_FILE = Path(__file__).resolve().parents[2] / "config" / "competition-order.csv"


@cache
def read_competition_order() -> dict[str, int]:
    """Each listed competition's place, by its code; none when the file is missing."""
    if not ORDER_FILE.exists():
        return {}
    with ORDER_FILE.open(encoding="utf-8", newline="") as handle:
        return {row["code"]: int(row["position"]) for row in csv.DictReader(handle)}


@cache
def read_competition_shown() -> dict[str, str]:
    """Each listed competition's rule (`always`, `around_matches` or `no`), by its code."""
    if not ORDER_FILE.exists():
        return {}
    with ORDER_FILE.open(encoding="utf-8", newline="") as handle:
        return {row["code"]: row["shown"] for row in csv.DictReader(handle)}


def rank_by_importance(competition: dict) -> tuple:
    """Listed competitions first in that order, the others after them by name, read
    without accents or case ("Évence" among the E's, not after the Z's)."""
    order = competition["display_order"]
    return order is None, order if order is not None else 0, simplify_name(competition["name"])


def simplify_name(name: str) -> str:
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()
