"""When a fact of a match becomes known, and when the match is forecast.

Every
match has two kick-off bounds: equal when the world trusts its clock (`kickoff_precision =
minute`), the earliest and latest times of its day when only the day is trusted (`date`). A fact
known after the match hangs on the latest bound, a forecast made before it on the earliest, so
whatever the real kick-off, nothing is used too early. Instants are ISO 8601 UTC text
(`2026-10-24T19:45:00Z`), which orders as the instants do.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

INSTANT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass(frozen=True)
class TimeRules:
    """`[time]` of `config/vesta/spine.toml`."""

    result_known_after: timedelta
    date_only_earliest: str
    date_only_latest: str

    @classmethod
    def from_config(cls, spine: dict) -> "TimeRules":
        section = spine["time"]
        return cls(
            result_known_after=timedelta(minutes=int(section["result_known_after_minutes"])),
            date_only_earliest=section["date_only_earliest_utc"],
            date_only_latest=section["date_only_latest_utc"],
        )


def find_kickoff_bounds(
    kickoff_utc: str, kickoff_precision: str, local_date: str, rules: TimeRules
) -> tuple[str, str]:
    """The earliest and the latest instant the match can have kicked off."""
    if kickoff_precision == "minute":
        return kickoff_utc, kickoff_utc
    if kickoff_precision == "date":
        return (
            f"{local_date}T{rules.date_only_earliest}:00Z",
            f"{local_date}T{rules.date_only_latest}:00Z",
        )
    raise ValueError(
        f"no timing rule for a kick-off of precision {kickoff_precision!r} (at {kickoff_utc}): "
        "write one before the spine reads such a match"
    )


def find_result_known_at(
    kickoff_utc: str, kickoff_precision: str, local_date: str, rules: TimeRules
) -> str:
    """The instant the result of the match is known: the latest kick-off plus the delay."""
    _, latest = find_kickoff_bounds(kickoff_utc, kickoff_precision, local_date, rules)
    return _add_delay(latest, rules.result_known_after)


def forecast_time(
    kickoff_utc: str, kickoff_precision: str, local_date: str, rules: TimeRules
) -> str:
    """The instant the spine's forecast of record is made: the earliest kick-off."""
    earliest, _ = find_kickoff_bounds(kickoff_utc, kickoff_precision, local_date, rules)
    return earliest


def _add_delay(instant: str, delay: timedelta) -> str:
    moment = datetime.strptime(instant, INSTANT_FORMAT).replace(tzinfo=UTC)
    return (moment + delay).strftime(INSTANT_FORMAT)
