"""Putting a frozen model version into the forecast log, before its first publication.

A version is published only from its git tag after its verdicts and the maintainers' approval
(`src/vesta/research/README.md` §5). A release refuses unless the code and configuration that run
are those of the tag, then writes, in the log, the version once with its tag and verdict, the
definitions of its quantities (a key already in the log must be defined exactly as before: a key
never changes meaning) and the map of what it publishes in which competitions
(`published_quantities`), which the readers read to know what each competition has.

A release is made of parts, each a family member with the keys whose verdicts passed and the
competitions those verdicts cover: the spine's keys can pass in the leagues and in the UEFA cups
by different verdicts. Every key is pre-match (a plan's `live` flag, written before 8 November
2026, is read and ignored). The plan of a release is a file in `config/vesta/releases/`, frozen
with the tag:

    python -m vesta.releases <forecast-log.db> <football-world.db> config/vesta/releases/2.0.0.toml
        [--from <instant>]

run from the tag's checkout once the maintainers have approved it. A user's run registers the
release in its own log without git (`vesta.release_gate`).
"""

import argparse
import json
import sqlite3
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from vesta.forecast_log.store import ForecastLogError, open_log
from vesta.forecast_log.writing import (
    register_model_version,
    register_published_quantity,
    register_quantity,
    supersede,
)
from vesta.jobs.fixtures import format_now_utc
from vesta.quantities import QuantityDefinition, load_quantities
from vesta.research.freeze import find_freeze_problems, make_tag

NATIONS = "nations"  # in a plan: every competition whose matches are between national teams


@dataclass(frozen=True)
class ReleasePart:
    family_member: str
    competition_ids: tuple[int, ...]
    published_keys: frozenset[str] | None = None  # None: every key of the member


def register_release_plan(
    connection: sqlite3.Connection,
    root: Path,
    version: str,
    verdict_id: str,
    what_changed: str,
    published_from: str,
    parts: list[ReleasePart],
    definitions: dict[str, QuantityDefinition],
    check_freeze: bool = True,
) -> list[tuple[str, int]]:
    """Register version `version` (the tag's number, `2.0.0`) once, every quantity of its parts'
    members, and each part's published keys in its competitions; returns the (key, competition)
    pairs now published. A key outside its part's member is refused (the hide rule keeps a key
    whose verdict has not passed defined and unpublished), and so is a key published twice in a
    competition. The versions published before it are superseded from `published_from`, all
    their keys with them. `check_freeze` False: the user's run, where the release manifest, not
    git, says whether the code is the release's (`vesta.release_gate`)."""
    problems = find_freeze_problems(root, version) if check_freeze else []
    if problems:
        raise ForecastLogError(f"version {version} is not the frozen code: {'; '.join(problems)}")
    model_version = f"vesta-{version}"
    chosen: dict[tuple[str, int], QuantityDefinition] = {}
    members: list[str] = []
    for part in parts:
        own = [item for item in definitions.values() if item.family_member == part.family_member]
        if not own:
            raise ForecastLogError(f"no quantity of {part.family_member!r} in the quantities file")
        if part.published_keys is not None:
            strangers = part.published_keys - {item.quantity.quantity_key for item in own}
            if strangers:
                raise ForecastLogError(
                    f"not quantities of {part.family_member!r}: {sorted(strangers)}")
        if part.family_member not in members:
            members.append(part.family_member)
            for item in own:
                _register_or_confirm(connection, item)
        for item in own:
            key = item.quantity.quantity_key
            if part.published_keys is not None and key not in part.published_keys:
                continue
            for competition_id in part.competition_ids:
                if (key, competition_id) in chosen:
                    raise ForecastLogError(
                        f"{key} is published twice in competition {competition_id}")
                chosen[(key, competition_id)] = item
    register_model_version(connection, model_version, ", ".join(members), published_from,
                           what_changed, freeze_tag=make_tag(version) if check_freeze else None,
                           verdict_id=verdict_id)  # fmt: skip
    for key, competition_id in chosen:
        register_published_quantity(connection, key, competition_id, model_version,
                                    published_from, False, False)  # fmt: skip
    supersede(connection, model_version, published_from)
    return list(chosen)


def read_plan(path: Path, world: sqlite3.Connection,
              skip_missing: bool = False) -> tuple[str, str, str, list[ReleasePart]]:  # fmt: skip
    """(version, verdict id, what changed, parts) of a plan file; its competitions are codes, or
    `nations`, resolved to the world's ids. `skip_missing`: a code the dataset does not hold is
    left out (a user's set of files), instead of refused."""
    plan = tomllib.loads(path.read_text(encoding="utf-8"))
    ids = dict(world.execute("SELECT code, competition_id FROM competitions").fetchall())
    parts = []
    for part in plan["part"]:
        competitions: list[int] = []
        for code in part["competitions"]:
            if code == NATIONS:
                competitions += read_nation_competitions(world)
            elif code in ids:
                competitions.append(ids[code])
            elif skip_missing:
                continue
            else:
                raise ForecastLogError(f"no competition {code!r} in the world")
        keys = part.get("keys")
        parts.append(ReleasePart(part["family_member"], tuple(dict.fromkeys(competitions)),
                                 None if keys is None else frozenset(keys)))  # fmt: skip
    return plan["version"], plan["verdict_id"], plan["what_changed"], parts


def read_nation_competitions(world: sqlite3.Connection) -> list[int]:
    """The competitions whose matches are between national teams (the nations model's)."""
    return [row[0] for row in world.execute(
        """
        SELECT DISTINCT s.competition_id FROM matches m
        JOIN seasons s ON s.season_id = m.season_id
        JOIN teams h ON h.team_id = m.home_team_id JOIN teams a ON a.team_id = m.away_team_id
        WHERE h.kind = 'nation' AND a.kind = 'nation' ORDER BY s.competition_id
        """)]  # fmt: skip


def _register_or_confirm(connection: sqlite3.Connection, item: QuantityDefinition) -> None:
    quantity = item.quantity
    stored = connection.execute(
        "SELECT catalogue_id, probability_kind, subject_kind, distribution_kind,"
        " conditional_on_playing, lines FROM quantities WHERE quantity_key = ?",
        (quantity.quantity_key,),
    ).fetchone()
    if stored is None:
        register_quantity(connection, quantity)
        return
    wanted = (
        quantity.catalogue_id,
        quantity.probability_kind,
        quantity.subject_kind,
        quantity.distribution_kind,
        int(quantity.conditional_on_playing),
        json.dumps(list(quantity.lines)),
    )
    if tuple(stored) != wanted:
        raise ForecastLogError(
            f"quantity {quantity.quantity_key} is in the log with another definition: a key "
            "never changes meaning, give the new definition a new key"
        )


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(description="register a frozen version in the forecast log")
    parser.add_argument("log", type=Path)
    parser.add_argument("world", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--from", dest="published_from", default=None)
    options = parser.parse_args(arguments)
    world = sqlite3.connect(f"file:{options.world.as_posix()}?mode=ro", uri=True)
    version, verdict_id, what_changed, parts = read_plan(options.plan, world)
    world.close()
    log = open_log(options.log)
    pairs = register_release_plan(log, Path(__file__).resolve().parents[2], version, verdict_id,
                                  what_changed, options.published_from or format_now_utc(), parts,
                                  load_quantities())  # fmt: skip
    log.close()
    keys = len({key for key, _ in pairs})
    print(f"vesta-{version}: {keys} keys published, {len(pairs)} (key, competition) pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
