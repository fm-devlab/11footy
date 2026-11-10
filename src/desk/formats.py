"""How a number, a line and a time are written in the window.

Percentages follow one rule: whole numbers; outcomes that exclude each other add up to 100 by the
largest remainder (`vesta.forecast_reading.round_to_whole_percentages`); then a 0 is "<1%" and a
100 is ">99%", unless the line is decided, when "0%" and "100%" are true. Times are the PC's own:
its time zone, and on Windows its own short date and time formats (the user's settings, 24 or 12
hours alike).
"""

import ctypes
import sys
from datetime import UTC, datetime

from vesta.forecast_reading import round_to_whole_percentages


def format_chance(probability: float, decided: bool = False, whole: int | None = None) -> str:
    """One probability as the window writes it; `whole` when a set has already been rounded."""
    value = round(100 * probability) if whole is None else whole
    if not decided:
        if value <= 0:
            return "<1%"
        if value >= 100:
            return ">99%"
    return f"{value}%"


def format_chances(lines: list[dict]) -> list[str]:
    """Outcomes that exclude each other (each with `probability` and `decided`), adding up to
    100 as shown."""
    wholes = round_to_whole_percentages([line["probability"] for line in lines])
    return [
        format_chance(line["probability"], line["decided"], whole)
        for line, whole in zip(lines, wholes, strict=True)
    ]


def find_exclusive_lines(line_keys: list[str]) -> list[int]:
    """The positions of the lines that are a set excluding each other and covering every case:
    the outcomes of an ordered or categorical answer (its other lines, "home or draw", are shown
    alone), a binary's yes and no; none otherwise."""
    outcomes = [index for index, key in enumerate(line_keys) if key.startswith("outcome=")]
    if outcomes:
        return outcomes
    if sorted(line_keys) == ["no", "yes"]:
        return [0, 1]
    return []


def name_side(word: str, home: str, away: str) -> str:
    return {"home": home, "away": away}.get(word, word.replace("_", " "))


def describe_line(line_key: str, home: str = "home", away: str = "away") -> str:
    """A line key of the reader (`vesta.forecast_reading`) in plain words."""
    operation, _, arguments = line_key.partition("=")
    parts = arguments.split(",") if arguments else []
    if operation == "outcome":
        word = name_side(arguments, home, away)
        return word if word in (home, away) else word.capitalize()
    if operation == "any":
        said = " or ".join(name_side(part, home, away) for part in parts)
        return said[:1].upper() + said[1:]
    if operation in ("margin", "margin_at_least"):
        side, goals = name_side(parts[0], home, away), parts[1]
        return f"{side} by {goals}" + (" or more" if operation == "margin_at_least" else "")
    fixed = {
        "yes": "Yes",
        "no": "No",
        "odd": "Odd",
        "even": "Even",
        "never": "Never",
        "home_win": f"{home} win",
        "draw": "Draw",
        "away_win": f"{away} win",
        "home_scores": f"{home} scores",
        "away_scores": f"{away} scores",
        "both_score": "Both score",
    }
    if operation in fixed and not arguments:
        return fixed[operation]
    worded = {
        "over": "Over {0}",
        "under": "Under {0}",
        "exactly": "Exactly {0}",
        "at_least": "{0} or more",
        "at_most": "{0} or fewer",
        "between": "{0} to {1}",
        "score": "Score {0}",
        "total_over": "Total over {0}",
        "total_under": "Total under {0}",
        "home_over": home + " over {0}",
        "away_over": away + " over {0}",
        "by": "By minute {0}",
        "band": "Minutes {0}",
        "place": "Place {0}",
        "top": "Top {0}",
        "bottom": "Bottom {0}",
    }
    counted = count_words(operation, parts[0]) if len(parts) == 1 else None
    if counted is not None:
        prefix = {"total_over": "Total ", "total_under": "Total ", "home_over": home + " ",
                  "away_over": away + " "}.get(operation, "")  # fmt: skip
        return prefix + counted
    if operation in worded and parts:
        return worded[operation].format(*parts)
    return line_key


def count_words(operation: str, threshold: str) -> str | None:
    """A count over or under a half ("over=2.5") as the whole numbers it means ("3 or more"),
    the plain words of the window."""
    if operation not in ("over", "under", "total_over", "total_under", "home_over",
                         "away_over"):  # fmt: skip
        return None
    whole, _, half = threshold.partition(".")
    if half != "5" or not whole.lstrip("-").isdigit():
        return None
    if operation.endswith("under"):
        return "None" if int(whole) == 0 else f"{int(whole)} or fewer"
    return f"{int(whole) + 1} or more"


def describe_window(window: dict | None) -> str:
    """A forecast's window (`desk.log.window`) in words; empty for one match or the
    whole season."""
    if not window:
        return ""
    if window["next_matches"] is not None:
        return f"next {window['next_matches']} matches"
    if window["matchday"] is not None:
        return f"matchday {window['matchday']}"
    return f"{window['from_date']} to {window['to_date']}"


def parse_instant(text: str) -> datetime:
    """An instant the dataset or the log writes in UTC (`2026-11-08T14:00:00Z`); one written
    without its zone is UTC too."""
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


class _SystemTime(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint16)
        for name in (
            "year", "month", "weekday", "day", "hour", "minute", "second", "millisecond"
        )
    ]  # fmt: skip


def _format_on_windows(kind: str, moment: datetime) -> str | None:
    """The user's own short date (with the weekday) or time without seconds, from Windows."""
    if sys.platform != "win32":
        return None
    kernel = ctypes.windll.kernel32
    system_time = _SystemTime(
        moment.year, moment.month, (moment.weekday() + 1) % 7, moment.day,
        moment.hour, moment.minute, 0, 0,
    )  # fmt: skip
    buffer = ctypes.create_unicode_buffer(128)
    if kind == "date":
        written = kernel.GetDateFormatEx(
            None,
            0,
            ctypes.byref(system_time),
            "ddd' '" + _read_short_date_pattern(),
            buffer,
            128,
            None,
        )
    else:
        no_seconds = 0x2  # TIME_NOSECONDS
        written = kernel.GetTimeFormatEx(None, no_seconds, ctypes.byref(system_time), None,
                                         buffer, 128)  # fmt: skip
    return buffer.value if written else None


def _read_short_date_pattern() -> str:
    buffer = ctypes.create_unicode_buffer(80)
    short_date = 0x1F  # LOCALE_SSHORTDATE
    ctypes.windll.kernel32.GetLocaleInfoEx(None, short_date, buffer, 80)
    return buffer.value or "dd/MM/yyyy"


def format_local_date(text: str) -> str:
    moment = parse_instant(text).astimezone()
    # elsewhere an English date no reader can take in another order: "Thu 12 Nov 2026"
    return _format_on_windows("date", moment) or format_english_date(moment)


def format_english_date(moment: datetime) -> str:
    """A date in English whatever the PC's locale: "Thu 12 Nov 2026"."""
    weekday = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[moment.weekday()]
    month = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
             "Dec")[moment.month - 1]  # fmt: skip
    return f"{weekday} {moment.day} {month} {moment.year}"


def format_local_time(text: str) -> str:
    moment = parse_instant(text).astimezone()
    return _format_on_windows("time", moment) or moment.strftime("%H:%M")


def format_local_moment(text: str) -> str:
    return f"{format_local_date(text)} {format_local_time(text)}"
