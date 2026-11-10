"""No information from after the forecast: the check of every input, and the alarm.

Two guards:
- the check: every input a forecast reads carries the instant it became known, and that
  instant must not come after the forecast's (a result known at 19:45 may feed a forecast made
  at 19:45); a single exception stops the run;
- the alarm: a candidate that looks better than any honest change could make it (a gain beyond
  the rule of its kind, an accuracy beyond what football allows) has almost surely seen the
  future; it stops the work until a research record explains it and the hypothesis names that
  record (`alarm_cleared_by`). There is no switch to turn it off.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


class LeakageError(RuntimeError):
    """An input known only after the forecast it feeds."""


class LeakageAlarm(RuntimeError):
    """A candidate too good to be honest; the work stops until it is understood."""


@dataclass(frozen=True)
class AlarmFinding:
    condition: str
    value: float
    threshold: float

    def describe(self) -> str:
        return f"{self.condition} {self.value:.4f} beyond the threshold {self.threshold:.4f}"


def check_known_by(inputs: Iterable[tuple[str, str]], forecast_time_utc: str) -> None:
    """Raise `LeakageError` naming every input (what, known at) known only after the forecast.

    Instants are compared as ISO 8601 UTC text of one form (`2026-10-23T18:45:00Z`), which
    orders as the instants do.
    """
    late = [(what, known_at) for what, known_at in inputs if known_at > forecast_time_utc]
    if late:
        listed = "; ".join(f"{what} known at {known_at}" for what, known_at in late[:10])
        raise LeakageError(
            f"{len(late)} inputs are known only after the forecast at {forecast_time_utc}: {listed}"
        )


def find_alarms(
    distribution_kind: str,
    rules: dict,
    gain: float,
    accuracy: float | None = None,
) -> list[AlarmFinding]:
    """The conditions of the kind's rule (`config/vesta/research.toml` [alarm.<kind>]) that the
    candidate trips. `gain` is the champion's deciding metric minus the candidate's (positive is
    better)."""
    rule = rules.get(distribution_kind)
    if rule is None:
        raise LeakageAlarm(
            f"no alarm rule for the kind {distribution_kind!r}: write [alarm.{distribution_kind}] "
            "in config/vesta/research.toml, with its reason, before its first verdict"
        )
    findings = []
    if gain > rule["max_gain"]:
        findings.append(AlarmFinding("gain over the published model", gain, rule["max_gain"]))
    if accuracy is not None and "max_accuracy" in rule and accuracy > rule["max_accuracy"]:
        findings.append(AlarmFinding("accuracy", accuracy, rule["max_accuracy"]))
    return findings


def enforce(findings: list[AlarmFinding], cleared_by: str | None, root: Path) -> None:
    """Pass when nothing tripped, or when the hypothesis names an existing research record that
    explains the alarm; otherwise raise `LeakageAlarm`."""
    if not findings:
        return
    described = "; ".join(finding.describe() for finding in findings)
    if cleared_by is None:
        raise LeakageAlarm(
            f"leakage alarm: {described}. The work stops until a research record "
            "explains it and the hypothesis names it in alarm_cleared_by"
        )
    if not (root / cleared_by).is_file():
        raise LeakageAlarm(f"leakage alarm: {described}; the record {cleared_by} does not exist")
