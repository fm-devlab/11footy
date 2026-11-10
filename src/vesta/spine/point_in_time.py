"""The point-in-time engine: every fact reaches the models at the instant it became known.

Two kinds of event, in the order (instant, kind, match):
- a fact: a finished match becomes known, at its `known_at`;
- a request: the features of a match to forecast are asked for, at its `forecast_time`.
At the same instant a fact comes before a request, so "only what is known at the forecast" is
not a filter but the order of the events: a family hears of a result only once its instant is
reached, and never sees a clock, only what `Fact` and `ForecastRequest` carry. Every request
records the latest instant each family had consumed, and the leakage check of the discipline
proves it is not after the forecast.

The previous engine also saved the state of each stream at every season's start, so that a daily
update replays only what changed; that comes back with the daily job, which is its only user.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from vesta.research.leakage import check_known_by

FACT, REQUEST = 0, 1
EVERY_COMPETITION = "*"


@dataclass(frozen=True)
class Fact:
    """What a family may read of a finished match."""

    match_id: int
    competition: str
    season: int
    home_team_id: int
    away_team_id: int
    home_goals: int
    away_goals: int
    known_at: str
    # our expected goals of each side, known with the result (nan when the match has no shots)
    home_expected_goals: float = float("nan")
    away_expected_goals: float = float("nan")
    # who played: (side 0 home or 1 away, player id, minutes, started, is a goalkeeper)
    players: tuple[tuple[int, int, float, bool, bool], ...] = ()


@dataclass(frozen=True)
class ForecastRequest:
    """What a family may read of a match to forecast: nothing of its outcome."""

    match_id: int
    competition: str
    season: int
    home_team_id: int
    away_team_id: int
    forecast_time: str


class Family(ABC):
    """A family of features: a state fed by facts, read by requests.

    `scope` is "competition" (one state per competition) or "all" (one state for every
    competition). `columns` names every feature the family returns, in order.
    """

    name: ClassVar[str]
    scope: ClassVar[str] = "competition"
    columns: ClassVar[tuple[str, ...]]

    @abstractmethod
    def make_empty_state(self) -> Any: ...

    @abstractmethod
    def learn(self, state: Any, fact: Fact) -> None:
        """A fact became known: update the state."""

    @abstractmethod
    def build_features(self, state: Any, request: ForecastRequest) -> dict[str, float]:
        """The features of a request; must not change the state."""


@dataclass(frozen=True)
class FeatureRow:
    request: ForecastRequest
    features: dict[str, float]
    latest_known_at: dict[str, str | None]  # per family: the latest fact it had consumed
    parts: dict[str, dict] | None = None  # per family its terms, for the requests asked


def replay(
    families: Sequence[Family],
    facts: Sequence[Fact],
    requests: Sequence[ForecastRequest],
    parts_for: frozenset[int] = frozenset(),
) -> list[FeatureRow]:
    """Every request's features, in the order of the events (facts and requests by instant,
    a fact first at the same instant, then by match); for the matches of `parts_for` also each
    family's terms (its `parts`, read by the explanation)."""
    names = [family.name for family in families]
    if len(set(names)) != len(names):
        raise ValueError(f"two families share a name: {names}")
    events = [(fact.known_at, FACT, fact.match_id, fact) for fact in facts]
    events += [(request.forecast_time, REQUEST, request.match_id, request) for request in requests]
    events.sort(key=lambda event: event[:3])
    states: dict[tuple[str, str], Any] = {}
    latest: dict[tuple[str, str], str] = {}
    rows = []
    for _, kind, _, event in events:
        if kind == FACT:
            for family in families:
                stream = _choose_stream(family, event.competition)
                if stream not in states:
                    states[stream] = family.make_empty_state()
                family.learn(states[stream], event)
                latest[stream] = event.known_at
            continue
        features: dict[str, float] = {}
        consumed: dict[str, str | None] = {}
        for family in families:
            stream = _choose_stream(family, event.competition)
            if stream not in states:
                states[stream] = family.make_empty_state()
            values = family.build_features(states[stream], event)
            if tuple(values) != family.columns:
                raise ValueError(f"{family.name} returned {tuple(values)}, not {family.columns}")
            features.update(values)
            consumed[family.name] = latest.get(stream)
        check_known_by(
            [(f"the latest fact of {name}", at) for name, at in consumed.items() if at],
            event.forecast_time,
        )
        parts = None
        if event.match_id in parts_for:
            parts = {
                family.name: family.build_log_terms(
                    states[_choose_stream(family, event.competition)], event
                )
                for family in families
                if hasattr(family, "build_log_terms")
            }
        rows.append(FeatureRow(event, features, consumed, parts))
    return rows


def _choose_stream(family: Family, competition: str) -> tuple[str, str]:
    return family.name, competition if family.scope == "competition" else EVERY_COMPETITION
