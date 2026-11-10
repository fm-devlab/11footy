"""Coverage in plain words: what each file is, how it was checked, and what Vesta can do with each
competition. Vesta's own words stay one click away (the line under the table); a reason this
does not know is shown as Vesta wrote it."""

from desk.opening import PLAYERS_PART, WHOLE_PART, CheckedFile, Coverage

# Vesta's models by the name it gives them (`vesta.capabilities`), in a reader's words
MODEL_WORDS = {
    "champion": "full model",
    "B0": "basic model",
    "cups Elo": "cups model",
    "nations Elo": "nations model",
}
FAILED = "did not pass Vesta's tests"  # a model served though it failed (`vesta.capabilities`)
# said on its competition's round and on each of its matches, beside the generic caveat
FAILED_NOTE = (
    "This model did not pass Vesta's tests: its chances are shown, marked not validated, and are"
    " less reliable than the others."
)
VALIDATION_WORDS = {"not validated on this dataset": "untested here",
                    "did not pass Vesta's tests": "failed Vesta's tests"}  # fmt: skip
# Vesta's reasons are its own plain words (`vesta.capabilities`); these say them shorter
REASON_WORDS = (
    (
        "the nations model",
        "The nations model, which did not pass Vesta's tests: its chances are"
        " marked not validated.",
    ),
    (
        "Vesta does not have this cup's rules",
        "Not forecast: Vesta does not have this cup's rules yet.",
    ),
)


def describe_file(item: CheckedFile, coverage: list[Coverage]) -> str:
    """What a file holds."""
    if item.part == WHOLE_PART:
        return "Everything: every competition and the players"
    if item.part == PLAYERS_PART or (not item.codes and item.players):
        return "The players' names"
    names = sorted({entry.name for entry in coverage if entry.code in item.codes})
    if not names:
        return item.part
    shown = ", ".join(names[:3])
    return shown + (f" and {len(names) - 3} more" if len(names) > 3 else "")


def describe_digest(item: CheckedFile, checked: bool) -> str:
    """Whether the file is the one SHA256SUMS lists."""
    if item.listed:
        return "Matches SHA256SUMS"
    if not checked:
        return "Checked on first opening"
    return "Not in a SHA256SUMS"


def describe_proof(level: str) -> str:
    """Whether the file is 11footy's, by Vesta's rule (`vesta.dataset_proof`)."""
    return {
        "signed": "Yes, signed",
        "listed": "Listed, not signed",
        "none": "Not proved",
    }.get(level, level)


def describe_forecast(item: Coverage) -> str:
    """Whether Vesta forecasts the competition, and with what."""
    if not item.served:
        return "No"
    said = "Yes · " + MODEL_WORDS.get(item.model or "", item.model or "a model")
    if item.validation:
        said += f" · {VALIDATION_WORDS.get(item.validation, item.validation)}"
    return said


def describe_reason(item: Coverage) -> str:
    """Why, in plain words."""
    if item.missing:
        return f"Needs the files of {', '.join(item.missing)} too."
    for start, words in REASON_WORDS:
        if item.reason.startswith(start):
            return words
    return item.reason[:1].upper() + item.reason[1:]
