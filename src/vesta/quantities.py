"""The definitions of the quantities Vesta publishes (`config/vesta/quantities.csv`).

One row per quantity (`docs/FORECAST_LOG.md` §5): its key, the catalogue item it answers, the
kind of its distribution, the lines it shows (separated by `|`), and two flags of the record:
whether it moved in play and was a live headline while live forecasting was part of the product
(until 8 November 2026; every key is pre-match since). A key is never reused for another meaning;
a model adds its rows when it is published.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from vesta.forecast_log.writing import Quantity

QUANTITIES_FILE = Path(__file__).resolve().parents[2] / "config" / "vesta" / "quantities.csv"


@dataclass(frozen=True)
class QuantityDefinition:
    quantity: Quantity
    family_member: str
    moves_live: bool
    live_headline: bool


def load_quantities(path: Path = QUANTITIES_FILE) -> dict[str, QuantityDefinition]:
    """Every definition of the file, by quantity key."""
    definitions: dict[str, QuantityDefinition] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            key = row["quantity_key"]
            if key in definitions:
                raise ValueError(f"{path}: the quantity {key} is defined twice")
            definitions[key] = QuantityDefinition(
                quantity=Quantity(
                    quantity_key=key,
                    catalogue_id=row["catalogue_id"],
                    probability_kind=row["probability_kind"],
                    subject_kind=row["subject_kind"],
                    name=row["name"],
                    period=row["period"],
                    distribution_kind=row["distribution_kind"],
                    conditional_on_playing=row["conditional_on_playing"] == "1",
                    outcome_rule=row["outcome_rule"],
                    lines=tuple(line for line in row["lines"].split("|") if line),
                    breakdown=row["breakdown"],
                    unit=row["unit"] or None,
                ),
                family_member=row["family_member"],
                moves_live=row["moves_live"] == "1",
                live_headline=row["live_headline"] == "1",
            )
    return definitions
