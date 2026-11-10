"""The registry of hypotheses: every change of a model is written down before it is tried.

`config/vesta/hypotheses.toml` holds one entry per hypothesis (its fields are described at the
top of the file). The registry refuses a duplicate id, a grid larger than the discipline allows
and a metric that does not belong to the entry's kind; the runner refuses an entry that is not in
the committed file, so no hypothesis is written after its result is known.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from vesta.distributions import KINDS
from vesta.research.repository import list_uncommitted_paths, read_committed_text
from vesta.research.scores import METRICS_BY_KIND

REGISTRY_PATH = "config/vesta/hypotheses.toml"
FAMILY_MEMBERS = (
    "spine",
    "goals in time",
    "team counts",
    "live chain",
    "lineups",
    "player counts",
    "season",
    "nations",
    "cups",
)


class RegistryError(ValueError):
    """An entry of the registry that breaks a rule of the discipline."""


@dataclass(frozen=True)
class Hypothesis:
    id: str
    family_member: str
    title: str
    change: str
    catalogue_ids: tuple[str, ...]
    distribution_kind: str
    metric: str
    expected_gain: float
    registered_on: str
    grid: tuple[dict, ...] = ()
    data: tuple[str, ...] = ()
    alarm_cleared_by: str | None = None


def parse_registry(text: str, grid_max: int) -> list[Hypothesis]:
    """The entries of a registry's text, in order, every rule checked."""
    entries = [_parse_entry(raw, grid_max) for raw in tomllib.loads(text).get("hypothesis", [])]
    seen: set[str] = set()
    for entry in entries:
        if entry.id in seen:
            raise RegistryError(f"hypothesis {entry.id} is registered twice")
        seen.add(entry.id)
    return entries


def read_committed_hypothesis(root: Path, hypothesis_id: str, grid_max: int) -> Hypothesis:
    """The entry `hypothesis_id` as committed, refused while the registry has uncommitted changes
    (an entry edited after a run would not be the one that was registered)."""
    if list_uncommitted_paths(root, REGISTRY_PATH):
        raise RegistryError(f"{REGISTRY_PATH} has uncommitted changes: commit the entry first")
    text = read_committed_text(root, REGISTRY_PATH)
    if text is None:
        raise RegistryError(f"{REGISTRY_PATH} is not committed")
    for entry in parse_registry(text, grid_max):
        if entry.id == hypothesis_id:
            return entry
    raise RegistryError(f"hypothesis {hypothesis_id} is not in the committed {REGISTRY_PATH}")


def _parse_entry(raw: dict, grid_max: int) -> Hypothesis:
    required = (
        "id",
        "family_member",
        "title",
        "change",
        "catalogue_ids",
        "distribution_kind",
        "expected_gain",
        "registered_on",
    )
    missing = [name for name in required if name not in raw]
    if missing:
        raise RegistryError(f"hypothesis {raw.get('id', '?')} lacks {missing}")
    known = {*required, "metric", "grid", "data", "alarm_cleared_by"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise RegistryError(f"hypothesis {raw['id']}: unknown fields {unknown}")
    kind = raw["distribution_kind"]
    if kind not in KINDS:
        raise RegistryError(f"hypothesis {raw['id']}: unknown distribution kind {kind!r}")
    metric = raw.get("metric", METRICS_BY_KIND[kind][0])
    if metric not in METRICS_BY_KIND[kind]:
        raise RegistryError(
            f"hypothesis {raw['id']}: {metric!r} is not a metric of {kind}: "
            f"{METRICS_BY_KIND[kind]}"
        )
    if raw["family_member"] not in FAMILY_MEMBERS:
        member = raw["family_member"]
        raise RegistryError(f"hypothesis {raw['id']}: unknown family member {member!r}")
    grid = tuple(raw.get("grid", ()))
    if len(grid) > grid_max:
        raise RegistryError(
            f"hypothesis {raw['id']}: a grid of {len(grid)} configurations, the limit is {grid_max}"
        )
    return Hypothesis(
        id=raw["id"],
        family_member=raw["family_member"],
        title=raw["title"],
        change=raw["change"],
        catalogue_ids=tuple(raw["catalogue_ids"]),
        distribution_kind=kind,
        metric=metric,
        expected_gain=float(raw["expected_gain"]),
        registered_on=str(raw["registered_on"]),
        grid=grid,
        data=tuple(raw.get("data", ())),
        alarm_cleared_by=raw.get("alarm_cleared_by"),
    )
