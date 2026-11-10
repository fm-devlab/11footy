"""The forecast log Vesta writes beside a dataset, read-only.

The window reads three things from it: the newest publication of a match made before kick-off,
the newest season publications, and their distributions. A line's chance is read from the
distribution by Vesta's own reader of lines (`vesta.forecast_reading.read_line`), the one place
where a line is defined, so nothing here sums, rounds or derives a probability. Every line a
quantity declares is read, each with Vesta's decided mark.
"""

import json
import sqlite3
from pathlib import Path

from desk.database import Database
from vesta.forecast_reading import read_line

RESULT_KEY = "M1.result"
# Vesta's trigger of a forecast made after kick-off from the facts known before the round began
# (log schema 12), and the code of the caveat it writes with it
AFTER_KICKOFF = "after_kickoff"
AFTER_KICKOFF_CAVEAT = "forecast_after_kickoff"
NEXT_MATCHES = "next_matches="
MATCHDAY = "matchday="


def parse_window(window_key: str) -> dict | None:
    """The window a forecast covers, as Vesta writes it: `''` is
    one match, or the whole season for a season quantity (None here); `<from>/<to>` a calendar
    window, both dates included; `next_matches=<k>` the subject's next k matches; `matchday=<k>`
    a round of a league."""
    if not window_key:
        return None
    empty = {"from_date": None, "to_date": None, "next_matches": None, "matchday": None}
    if window_key.startswith(NEXT_MATCHES):
        return empty | {"next_matches": int(window_key.removeprefix(NEXT_MATCHES))}
    if window_key.startswith(MATCHDAY):
        return empty | {"matchday": int(window_key.removeprefix(MATCHDAY))}
    start, _, end = window_key.partition("/")
    return empty | {"from_date": start, "to_date": end}


class ForecastLog:
    def __init__(self, path: Path) -> None:
        self.database = Database(path)
        self._quantities: dict[str, sqlite3.Row] = {}

    def is_written(self) -> bool:
        """Whether Vesta has written the log yet: before its first run there is none."""
        return self.database.has_file()

    # Publications
    def find_latest_pre_match_publication(self, match_id: int) -> sqlite3.Row | None:
        """The newest publication of a match made before kick-off: the window shows Vesta's
        forecasts before the match only."""
        return self.database.read_first_row(
            "select * from publications where match_id = ? and stage = 'pre_match'"
            " order by publication_id desc limit 1",
            match_id,
        )

    def find_latest_season_publications(self, season_id: int) -> list[tuple[sqlite3.Row, set[str]]]:
        """Each quantity of the season from the newest morning that published it: Vesta may
        publish a season's quantities in more than one publication, and the newest must not
        hide the others."""
        newest: dict[int, set[str]] = {}
        for row in self.database.read_rows(
            "select d.quantity_key, max(p.publication_id) as publication_id from publications p"
            " join distributions d using (publication_id) where p.season_id = ?"
            " and p.stage = 'season_day' group by d.quantity_key",
            season_id,
        ):
            newest.setdefault(row["publication_id"], set()).add(row["quantity_key"])
        by_id = "select * from publications where publication_id = ?"
        return [
            (self.database.read_first_row(by_id, found), keys)
            for found, keys in sorted(newest.items())
        ]

    # Quantities and distributions
    def read_quantity(self, quantity_key: str) -> sqlite3.Row | None:
        """A quantity's definition, read again when a key is new: a run may register more."""
        if quantity_key not in self._quantities:
            self._quantities = {
                row["quantity_key"]: row
                for row in self.database.read_rows("select * from quantities")
            }
        return self._quantities.get(quantity_key)

    def read_distributions(self, publication: sqlite3.Row, quantity_keys: set[str] | None = None):
        """A publication's distributions, those of `quantity_keys` only when given, in the
        log's order by quantity and subject."""
        has_marks = "decided_lines" in self.database.read_columns("distributions")
        found = []
        for row in self.database.read_rows(
            "select * from distributions where publication_id = ?"
            " order by quantity_key, subject_id, window_key",
            publication["publication_id"],
        ):
            if quantity_keys is not None and row["quantity_key"] not in quantity_keys:
                continue
            found.append({
                "quantity_key": row["quantity_key"],
                "subject_id": row["subject_id"],
                "window_key": row["window_key"],
                "model_version": row["model_version"],
                "values": json.loads(row["distribution_values"]),
                "decided_lines": json.loads(row["decided_lines"] or "[]") if has_marks else [],
            })  # fmt: skip
        return found

    def read_caveats(self, publication_id: int, probability_kind: str) -> list[str]:
        """What Vesta says a publication's chances of a kind leave out, as it wrote them (none
        on a log written before `publication_caveats`)."""
        if not self.database.read_columns("publication_caveats"):
            return []
        return [
            row["description"]
            for row in self.database.read_rows(
                "select description from publication_caveats"
                " where publication_id = ? and probability_kind = ? order by code",
                publication_id,
                probability_kind,
            )
        ]

    # Forecasts
    def build_forecast(self, publication: sqlite3.Row, distribution: dict) -> dict:
        """One distribution with every line its quantity declares, as the screens read it."""
        quantity = self.read_quantity(distribution["quantity_key"])
        kind = quantity["distribution_kind"]
        lines = [
            {
                "line_key": line_key,
                "probability": read_line(kind, distribution["values"], line_key),
                "decided": line_key in distribution["decided_lines"],
            }
            for line_key in json.loads(quantity["lines"])
        ]
        return {
            "publication_id": publication["publication_id"],
            "published_at": publication["published_at"],
            "quantity_key": distribution["quantity_key"],
            "catalogue_id": quantity["catalogue_id"],
            "subject_kind": quantity["subject_kind"],
            "subject_id": distribution["subject_id"],
            "window": parse_window(distribution["window_key"]),
            "model_version": distribution["model_version"],
            "distribution_kind": kind,
            "distribution": distribution["values"],
            "lines": lines,
            "caveats": self.read_caveats(
                publication["publication_id"], quantity["probability_kind"]
            ),
        }

    def read_match_forecasts(self, match_id: int) -> list[dict]:
        """Every forecast of a match's newest publication before kick-off."""
        if not self.is_written():
            return []
        publication = self.find_latest_pre_match_publication(match_id)
        if publication is None:
            return []
        return [
            self.build_forecast(publication, found)
            for found in self.read_distributions(publication)
        ]

    def read_season_forecasts(self, season_id: int) -> list[dict]:
        """A season's forecasts, each quantity from the newest publication that holds it."""
        if not self.is_written():
            return []
        return [
            self.build_forecast(publication, found)
            for publication, keys in self.find_latest_season_publications(season_id)
            for found in self.read_distributions(publication, keys)
        ]

    def read_result_forecast(self, match_id: int) -> dict | None:
        """The match's home, draw and away chances from its newest publication before kick-off,
        with the outcomes Vesta marked decided; None before Vesta's first forecast of it."""
        if not self.is_written():
            return None
        publication = self.find_latest_pre_match_publication(match_id)
        if publication is None:
            return None
        for found in self.read_distributions(publication, {RESULT_KEY}):
            outcomes = dict(found["values"])
            decided = set(found["decided_lines"])
            after = "trigger" in publication.keys() and publication["trigger"] == AFTER_KICKOFF
            return {
                "home": outcomes["home"],
                "draw": outcomes["draw"],
                "away": outcomes["away"],
                "decided_outcomes": [side for side in ("home", "draw", "away")
                                     if f"outcome={side}" in decided],
                "after_kickoff": after,
                "after_kickoff_caveat": (self.read_caveat(publication["publication_id"],
                                                     AFTER_KICKOFF_CAVEAT) if after else None),
            }  # fmt: skip
        return None

    def read_caveat(self, publication_id: int, code: str) -> str | None:
        """One caveat of a publication in Vesta's words, whatever kind of chance it is on."""
        if not self.database.read_columns("publication_caveats"):
            return None
        row = self.database.read_first_row(
            "select description from publication_caveats where publication_id = ? and code = ?",
            publication_id,
            code,
        )
        return row["description"] if row else None
