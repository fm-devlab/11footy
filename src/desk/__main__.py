"""Open the window: `python -m desk`, or `11footy` once `uv sync` has installed the project.

A failure at start is never silent, though the launchers run the window without a console:
the error and its traceback are written to `start-failure.log` in the user's folder for
applications (`desk.state`). Where Tk works, a message box says so and names the file; where it
does not, the same words go to the error stream, which the macOS and Linux launchers show.
"""

import contextlib
import sys
import traceback
from pathlib import Path

FAILURE_LOG = "start-failure.log"


def main() -> None:
    try:
        start()
    except Exception as error:  # noqa: BLE001 (every failure at start is told, then it ends)
        report_start_failure(error)
        sys.exit(1)


def start() -> None:
    if sys.platform == "win32":
        # sharp text at any scaling of the screen: Tk then sizes its fonts in points itself
        import ctypes

        with contextlib.suppress(AttributeError, OSError):
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
    from desk.state import read_last_chosen
    from desk.window import Window

    Window(read_last_chosen()).mainloop()


def report_start_failure(error: BaseException, log: Path | None = None) -> str:
    """Write the failure down, then tell the person where: in a message box when Tk works, on
    the error stream otherwise. Returns what was said."""
    details = "".join(traceback.format_exception(error))
    where = write_start_failure(details, log)
    said = f"11footy could not start: {error}"
    said += f"\n\nThe details are in {where}" if where else f"\n\n{details}"
    if not show_in_a_box(said) and sys.stderr is not None:
        print(said, file=sys.stderr)
    return said


def write_start_failure(details: str, log: Path | None = None) -> Path | None:
    """The log written, or None when even the user's folder cannot be written."""
    try:
        from desk.state import find_state_file

        log = log or find_state_file().with_name(FAILURE_LOG)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(details, encoding="utf-8")
    except Exception:  # noqa: BLE001 (the message then carries the details itself)
        return None
    return log


def show_in_a_box(said: str) -> bool:
    """True when a message box showed it; False when Tk itself does not work."""
    try:
        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("11footy could not start", said, parent=root)
        root.destroy()
    except Exception:  # noqa: BLE001 (no display, no Tk: the caller prints it)
        return False
    return True


if __name__ == "__main__":
    main()
