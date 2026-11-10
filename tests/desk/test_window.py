"""The window, opened once: a second Tk in the same process often cannot find its tk.tcl on
Windows, and the application only ever makes one."""

import gc
import subprocess
import sys
import tkinter

import pytest
from small_world import GOALLESS, TO_COME

from desk import reading
from desk.opening import open_paths
from desk.reading import SeasonView
from desk.state import read_state
from desk.window import NOT_MADE_BEFORE_KICKOFF, Window, describe_estimate
from desk.words import FAILED, FAILED_NOTE


@pytest.fixture
def unseen(monkeypatch):
    """Every Tk window of the test is transparent from birth and has no taskbar button: it is
    mapped, so its layout can be checked, but nothing ever shows on the screen."""
    made = tkinter.Tk.__init__

    def hide(self, *args, **kwargs):
        made(self, *args, **kwargs)
        self.attributes("-alpha", 0.0)
        if sys.platform == "win32":
            self.attributes("-toolwindow", True)

    monkeypatch.setattr(tkinter.Tk, "__init__", hide)


def test_the_window_starts_empty_then_shows_a_dataset(ours, theirs, tmp_path, unseen,
                                                       monkeypatch):
    try:
        window = Window(None, state_file=tmp_path / "state.json")
    except tkinter.TclError as error:  # no display
        pytest.skip(str(error))
    try:
        window.update()
        assert float(window.attributes("-alpha")) == 0.0, "the test never shows a window"
        assert window.start_panel.winfo_ismapped()
        assert str(window.compute_button["state"]) == "disabled"

        # our dataset's Coverage and status line count only what the Matches list shows
        monkeypatch.setattr(reading, "read_competition_shown", lambda: {"PL": "always"})
        window._show_opened(open_paths([ours], check=False))
        window.update()
        assert window.screens.winfo_ismapped()
        listed = [row.values["competition"] for group in window.coverage_list.items
                  for row in group.children]  # fmt: skip
        assert listed == ["Premier League"]
        assert "of 1 competitions can be forecast" in window.vesta_lines.get("1.0", "end")
        assert window.title().endswith("11footy")
        # the Matches screen opens on the current round and moves along the season
        names = [competition["name"] for competition in window.competitions]
        window.competition_box.current(names.index("Premier League"))
        window.show_matches()
        window.update()
        assert window.round_title["text"] == "Matchday 2"
        assert str(window.next_button["state"]) == "disabled", "the season's last round"
        window.match_list.body.focus_set()
        window.match_list.body.event_generate("<Left>")
        window.update()
        assert window.round_title["text"] == "Matchday 1"
        assert window.round_note["text"] == "1 of 2 · played"
        assert str(window.previous_button["state"]) == "disabled"
        window.move_round(-1)
        assert window.round_title["text"] == "Matchday 1", "nothing before the first round"
        window.next_button.invoke()
        window.update()
        assert window.round_title["text"] == "Matchday 2"
        window.show_match(TO_COME)
        window.update()
        assert window.match_tabs.tabs()
        assert FAILED_NOTE not in window.match_made["text"]
        # a competition served by a model that failed its tests says so, on the round and the match
        for entry in window.opened.coverage:
            if entry.name == "Premier League":
                entry.validation = FAILED
        window.show_round()
        assert window.matches_note["text"].startswith(FAILED_NOTE)
        window.show_match(TO_COME)
        assert window.match_made["text"].endswith(FAILED_NOTE)
        assert window.screens.select() == str(window.match_screen)
        window.show_match(GOALLESS)
        window.update()
        assert window.match_detail["text"].endswith("Finished 0-0")
        assert window.match_made["text"] == NOT_MADE_BEFORE_KICKOFF, "finished: nothing to compute"
        window.event_generate("<Escape>")
        window.update()
        assert window.screens.select() == str(window.matches_screen), "Esc goes back"
        window.show_season()
        window.update()
        assert window.coverage_list.items, "the coverage lists the competitions"
        # a season without a table nor forecasts says so once, without pointing below
        window._fill_season(SeasonView("2026", None, [], [], []), "Cup")
        assert window.season_table.empty == "No table for Cup in this dataset."
        assert window.season_others.empty == "No season forecasts."
        assert window.files_list.items[0].values["ours"].startswith("Not proved")

        # the switch to dark is remembered, and so is the window's size
        window.dark.set(True)
        window.switch_look()
        window.update()
        assert window.look.name == "dark"
        assert window.match_list.body["background"] == window.look.colours.panel
        window.geometry("900x600+30+40")
        window.update()
        window.close()
        kept = read_state(tmp_path / "state.json")
        assert kept["theme"] == "dark" and kept["geometry"].startswith("900x600")
        window = Window(None, state_file=tmp_path / "state.json")
        window.update()
        assert window.look.name == "dark"
        assert window.winfo_width() == 900

        window._show_opened(open_paths([theirs], check=False))
        window.update()
        assert window.dataset_label["text"].startswith("someone")
        window.update()
        assert window.title().endswith("someone")
        assert window.competition_box.get() == "XL"
        assert "no blind verdict" in window.notes["text"]
    finally:
        window.destroy()
        # Tk's objects must die in this thread: collected later from another test's thread,
        # they crash the process.
        del window
        gc.collect()


def test_the_window_imports_nothing_of_the_maintainers():
    """The maintainer's data pipeline stays out of the window people download."""
    code = (
        "import sys, desk.window, desk.__main__;"
        " print(sorted(name for name in ('data',) if name in sys.modules))"
    )
    found = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           check=True).stdout.strip()  # fmt: skip
    assert found == "[]"


def test_the_estimate_is_the_last_run_else_vestas_measure_of_the_whole_file():
    assert describe_estimate(7.4, whole=True) == "The last run on this PC took 7 minutes."
    assert "10 to 30 minutes" in describe_estimate(None, whole=True)
    assert "3 to 8 minutes" in describe_estimate(None, whole=True)
    assert "2 to 5 minutes" in describe_estimate(None, whole=False)


def test_the_current_round_says_played_once_every_match_is_over():
    from desk.reading import RoundView
    from desk.window import describe_current

    def view(*matches):
        return RoundView("Final", [{"status": s, "finished": f} for s, f in matches])

    assert describe_current(view(("Finished 1-1 (4-3 pens)", True))) == "played"
    assert describe_current(view(("Finished 1-0", True), ("Postponed", False))) == "played"
    assert describe_current(view(("Finished 1-0", True), ("To play", False))) == "in progress"
    assert describe_current(view(("To play", False), ("Postponed", False))) == "next"
