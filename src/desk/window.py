"""The window: one dataset at a time, Vesta's forecasts on it computed on this PC, four screens.

The header, always on top: the dataset open, "Open files..." (the whole file `11footy.db.xz`,
chosen split files, or anyone's dataset in the Vesta data format), "Open a folder..." (a folder
of split files, every one of them), "Compute forecasts", which runs Vesta's one command, and the
switch between light and dark. Beneath it a line for the progress, the dataset's notes, then the
screens: Matches (every round of a competition's season, opened on the current one, each match with
its status and its last forecast before kick-off), Match (the whole forecast of the match picked,
in tabs), Season tables (the table as the dataset has it beside the chances of its final places)
and Coverage (each file and how it was checked, each competition and what Vesta can do with it). At
the bottom, what Vesta said, folded to its last line. Before the first dataset, a panel says what
to open.

Keys: Ctrl+O opens files; in a table the arrows move, Enter opens a match or a section; in the
Matches screen left and right move between the rounds; Esc goes back from a match to the round. The
window keeps its size and place, and the look chosen.
"""

import queue
import re
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from desk import formats, state
from desk.opening import WHOLE_PART, Opened, Refused, open_paths
from desk.reading import LATER, PAST, MatchView, Reader, Rounds, RoundView, Row, SeasonView
from desk.run import (
    AlreadyRunning,
    Progress,
    Run,
    compute_command,
    is_command_ready,
    read_last_minutes,
)
from desk.table import CHANCE, CHANCE_UNDER, Column, Item, Table
from desk.theme import DARK, LIGHT, Look, choose_theme
from desk.words import (
    FAILED,
    FAILED_NOTE,
    describe_digest,
    describe_file,
    describe_forecast,
    describe_proof,
    describe_reason,
)

TITLE = "11footy"
FIRST_START = (
    "11footy's dataset is released on GitHub, refreshed after each round: one file per league,"
    " per European cup, the national teams and the players, and the whole dataset in one file"
    " (11footy.db.xz), each with SHA256SUMS. The national teams' forecasts come from a model"
    " that has not passed Vesta's tests yet: the window marks them as not validated. Download"
    " the whole file, or the files you want together with players.db.xz, and SHA256SUMS, into"
    " one folder. A dataset of your own in"
    " the Vesta data format opens the same way: the format is described in"
    " docs/VESTA_DATA_FORMAT.md, in this program's folder.\n\n"
    "Then open it here, and press Compute forecasts once it is open: Vesta computes the"
    " forecasts on this PC."
)
NOT_VALIDATED = (
    "Vesta's forecasts on a dataset other than 11footy's carry its caveat: no blind verdict of"
    " Vesta covers this dataset."
)
NO_COMMAND = "Vesta's command for computing forecasts is not in this version yet."
# Vesta's full run on the whole dataset, measured 9 November 2026 (champion and season included).
# a finished match without a forecast: computing now cannot add one made before its kickoff
NOT_MADE_BEFORE_KICKOFF = "No forecast was made before kickoff."
# a round after the next one: Vesta forecasts the next round only
COMES_LATER = "The forecast comes when this is the next matchday."
PLACE_WORDS = {PAST: "played", LATER: "to come"}
WHOLE_FIRST_RUN = (
    "On the whole dataset the first run takes about 10 to 30 minutes, depending on the PC and"
    " what else it is doing (Vesta builds a cache beside the dataset), the next ones about 3 to 8"
    " minutes."
)
PART_FIRST_RUN = (
    "One league with players.db takes about 2 to 5 minutes, depending on the PC; more files"
    " take longer."
)
NO_MATCH = "No match chosen yet. On Matches, double-click a match, or choose it and press Enter."
COVERAGE_HINT = "Choose a competition to read Vesta's own words about it here."
COVERAGE_INTRO = (
    "Each file of the dataset and how it was checked; each competition, and whether Vesta"
    " forecasts it."
)
FILE_TYPES = [("Dataset files", "*.xz *.db"), ("Any file", "*.*")]
OPEN_LINES = 26  # a section of a match opens by itself when it has this many lines or fewer
GEOMETRY = re.compile(r"(\d+)x(\d+)([+-]-?\d+)([+-]-?\d+)")


class Window(tk.Tk):
    def __init__(self, chosen: list[Path] | None = None, state_file: Path | None = None) -> None:
        super().__init__()
        self.state_file = state_file
        remembered = state.read_state(state_file)
        self.look = Look(self, choose_theme(remembered.get("theme")))
        self.title(TITLE)
        self._place(remembered.get("geometry"), bool(remembered.get("zoomed")))
        self.minsize(self.look.convert_points(560), self.look.convert_points(400))
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.reader: Reader | None = None
        self.opened: Opened | None = None
        self.competitions: list[dict] = []
        self.rounds: Rounds | None = None
        self.round_index = 0
        self.places: dict[int, str] = {}  # each shown match's round: past, current or later
        self.lines: queue.Queue[str] = queue.Queue()
        self.events: queue.Queue[tuple] = queue.Queue()
        self.run: Run | None = None
        self.opening = False
        self.messages_open = False

        self._build_header()
        self._build_messages()
        self.body = ttk.Frame(
            self, padding=(self.measure_gap(2), self.measure_gap(1), self.measure_gap(2), 0)
        )
        self.body.pack(fill="both", expand=True)
        self._build_start()
        self.screens = ttk.Notebook(self.body)
        self.matches_screen = ttk.Frame(self.screens, padding=(0, self.measure_gap(1.5), 0, 0))
        self.match_screen = ttk.Frame(self.screens, padding=(0, self.measure_gap(1.5), 0, 0))
        self.season_screen = ttk.Frame(self.screens, padding=(0, self.measure_gap(1.5), 0, 0))
        self.coverage_screen = ttk.Frame(self.screens, padding=(0, self.measure_gap(1.5), 0, 0))
        self.screens.add(self.matches_screen, text="Matches")
        self.screens.add(self.match_screen, text="Match")
        self.screens.add(self.season_screen, text="Season tables")
        self.screens.add(self.coverage_screen, text="Coverage")
        self.screens.bind("<<NotebookTabChanged>>", lambda _: self._focus_screen())
        self._build_matches()
        self._build_match()
        self._build_season()
        self._build_coverage()
        self.bind("<Control-o>", lambda _: self.choose_files())
        self.bind("<Escape>", lambda _: self.back_to_round())
        self.look.follow(self._recolour)
        self._recolour()

        self._show_start()
        if chosen:
            # the dataset opened last, its digests checked when it was first opened
            self.open_dataset(chosen, check=False)
        self.after(150, self._drain)

    def measure_gap(self, units: float) -> int:
        """The window's spacing: units of 6 points, in this screen's pixels."""
        return self.look.convert_points(6 * units)

    # Size and place
    def _place(self, geometry: str | None, zoomed: bool) -> None:
        """Where the window was last time, when it still fits this screen; else a size that
        suits it, at most the screen less its margins."""
        screen_width, screen_height = self.winfo_screenwidth(), self.winfo_screenheight()
        found = GEOMETRY.fullmatch(geometry or "")
        if found:
            width, height = int(found[1]), int(found[2])
            x, y = int(found[3].replace("+-", "-")), int(found[4].replace("+-", "-"))
            fits = (
                width <= screen_width
                and height <= screen_height
                and -width // 2 < x < screen_width - 100
                and 0 <= y < screen_height - 100
            )
            if fits:
                self.geometry(f"{width}x{height}+{x}+{y}")
                if zoomed:
                    self._zoom()
                return
        width = min(screen_width - self.look.convert_points(30), self.look.convert_points(1020))
        height = min(screen_height - self.look.convert_points(70), self.look.convert_points(680))
        self.geometry(
            f"{width}x{height}+{(screen_width - width) // 2}+{self.look.convert_points(10)}"
        )
        if zoomed:
            self._zoom()

    def _zoom(self) -> None:
        try:
            if sys.platform.startswith("linux"):
                self.attributes("-zoomed", True)
            else:
                self.state("zoomed")
        except tk.TclError:
            pass

    def _is_zoomed(self) -> bool:
        try:
            if sys.platform.startswith("linux"):
                return bool(self.attributes("-zoomed"))
            return self.state() == "zoomed"
        except tk.TclError:
            return False

    def close(self) -> None:
        """Remember the size, place and look, stop Vesta if it is computing, then close."""
        if self.run is not None and self.run.is_running():
            if not messagebox.askokcancel(
                "Vesta is computing", "Stop Vesta's computing and close the window?"
            ):
                return
            self.run.stop()
        zoomed = self._is_zoomed()
        values: dict = {"theme": self.look.name, "zoomed": zoomed}
        if not zoomed and self.state() == "normal":
            values["geometry"] = self.wm_geometry()
        try:
            state.keep(values, self.state_file)
        except OSError:
            pass  # the window opens at its usual size next time
        self.destroy()

    # The header, the progress line, the notes and Vesta's messages
    def _build_header(self) -> None:
        header = ttk.Frame(
            self,
            padding=(
                self.measure_gap(2),
                self.measure_gap(1.2),
                self.measure_gap(2),
                self.measure_gap(1),
            ),
        )
        header.pack(side="top", fill="x")
        ttk.Label(header, text=TITLE, font=self.look.fonts["title"]).pack(side="left")
        self.dataset_label = ttk.Label(header, style="Muted.TLabel")
        self.dataset_label.pack(side="left", padx=(self.measure_gap(2), 0))
        self.dark = tk.BooleanVar(value=self.look.name == DARK)
        ttk.Checkbutton(
            header,
            text="Dark",
            style="Switch.TCheckbutton",
            variable=self.dark,
            command=self.switch_look,
        ).pack(side="right", padx=(self.measure_gap(2), 0))
        self.compute_button = ttk.Button(
            header,
            text="Compute forecasts",
            style="Accent.TButton",
            command=self.compute,
            state="disabled",
        )
        self.compute_button.pack(side="right", padx=(self.measure_gap(1), 0))
        self.open_folder_button = ttk.Button(
            header, text="Open a folder…", command=self.choose_folder
        )
        self.open_folder_button.pack(side="right", padx=(self.measure_gap(1), 0))
        self.open_files_button = ttk.Button(header, text="Open files…", command=self.choose_files)
        self.open_files_button.pack(side="right")
        self.header_rule = ttk.Frame(self, style="Rule.TFrame", height=1)
        self.header_rule.pack(side="top", fill="x")

        strip = ttk.Frame(
            self, padding=(self.measure_gap(2), self.measure_gap(0.8), self.measure_gap(2), 0)
        )
        self.progress = ttk.Progressbar(
            strip, length=self.look.convert_points(160), mode="determinate"
        )
        self.stop_button = ttk.Button(strip, text="Stop", command=self.stop)
        self.status = ttk.Label(strip, style="Muted.TLabel")
        self.status.pack(side="left")
        self.strip = strip

        self.notes_frame = ttk.Frame(
            self, style="Note.TFrame", padding=(self.measure_gap(1.5), self.measure_gap(1))
        )
        self.notes = ttk.Label(self.notes_frame, style="Note.TLabel", justify="left")
        self.notes.pack(anchor="w", fill="x")
        self.notes.bind(
            "<Configure>", lambda event: self.notes.config(wraplength=max(200, event.width))
        )

    def _show_busy(self, busy: bool, stoppable: bool = False) -> None:
        self.busy = busy
        self.progress.pack_forget()
        self.stop_button.pack_forget()
        self.status.pack_forget()
        if busy:
            self.progress.pack(side="left", padx=(0, self.measure_gap(1.5)))
        self.status.pack(side="left")
        if stoppable:
            self.stop_button.pack(side="left", padx=(self.measure_gap(1.5), 0))
        self._place_strip()

    def set_status(self, text: str) -> None:
        """The line under the header: the progress, or the last run's time; none when empty."""
        self.status.config(text=text)
        self._place_strip()

    def _place_strip(self) -> None:
        if getattr(self, "busy", False) or self.status["text"]:
            self.strip.pack(side="top", fill="x", after=self.header_rule)
        else:
            self.strip.pack_forget()

    def _build_messages(self) -> None:
        frame = ttk.Frame(
            self,
            padding=(
                self.measure_gap(2),
                self.measure_gap(0.5),
                self.measure_gap(2),
                self.measure_gap(1),
            ),
        )
        frame.pack(side="bottom", fill="x")
        ttk.Frame(self, style="Rule.TFrame", height=1).pack(side="bottom", fill="x")
        top = ttk.Frame(frame)
        top.pack(fill="x")
        ttk.Label(top, text="What Vesta said", font=self.look.fonts["strong"]).pack(side="left")
        self.last_line = ttk.Label(top, style="Muted.TLabel")
        self.last_line.pack(side="left", fill="x", expand=True, padx=(self.measure_gap(1.5), 0))
        self.messages_button = ttk.Button(top, text="Show all", command=self.toggle_messages)
        self.messages_button.pack(side="right")
        self.vesta_lines = tk.Text(
            frame,
            height=7,
            wrap="word",
            state="disabled",
            borderwidth=0,
            highlightthickness=0,
            padx=self.measure_gap(1),
            pady=self.measure_gap(0.5),
            font=self.look.fonts["caption"],
        )
        self.messages_frame = frame

    def toggle_messages(self) -> None:
        self.messages_open = not self.messages_open
        if self.messages_open:
            self.vesta_lines.pack(fill="x", pady=(self.measure_gap(0.5), 0))
            self.vesta_lines.see("end")
        else:
            self.vesta_lines.pack_forget()
        self.messages_button.config(text="Hide" if self.messages_open else "Show all")

    def say(self, line: str) -> None:
        """A line in what Vesta said; the last one stays in sight."""
        self.vesta_lines.config(state="normal")
        self.vesta_lines.insert("end", line + "\n")
        self.vesta_lines.see("end")
        self.vesta_lines.config(state="disabled")
        last = line.strip().splitlines()[-1] if line.strip() else ""
        self.last_line.config(text=last)

    def switch_look(self) -> None:
        self.look.apply(DARK if self.dark.get() else LIGHT)
        try:
            state.keep({"theme": self.look.name}, self.state_file)
        except OSError:
            pass

    def _recolour(self) -> None:
        colours = self.look.colours
        self.vesta_lines.configure(
            background=colours.panel,
            foreground=colours.text,
            insertbackground=colours.text,
            selectbackground=colours.chosen,
            selectforeground=colours.text,
        )
        if hasattr(self, "result_strip"):
            self._draw_result_strip()

    # Before the first dataset
    def _build_start(self) -> None:
        self.start_panel = ttk.Frame(self.body, padding=(self.measure_gap(1), self.measure_gap(3)))
        ttk.Label(
            self.start_panel, text="No dataset open yet", font=self.look.fonts["subtitle"]
        ).pack(anchor="w")
        explained = ttk.Label(
            self.start_panel,
            text=FIRST_START,
            justify="left",
            wraplength=self.look.convert_points(520),
        )
        explained.pack(anchor="w", pady=(self.measure_gap(1.5), self.measure_gap(2)))
        buttons = ttk.Frame(self.start_panel)
        buttons.pack(anchor="w")
        ttk.Button(
            buttons, text="Open files…", style="Accent.TButton", command=self.choose_files
        ).pack(side="left")
        ttk.Button(buttons, text="Open a folder…", command=self.choose_folder).pack(
            side="left", padx=(self.measure_gap(1), 0)
        )

    def _show_start(self) -> None:
        self.screens.pack_forget()
        self.start_panel.pack(fill="both", expand=True)
        self.dataset_label.config(text="No dataset open")
        self._show_notes([])
        self._update_buttons()

    def _show_notes(self, notes: list[str]) -> None:
        self.notes.config(text="\n".join(notes))
        if notes:
            self.notes_frame.pack(
                side="top",
                fill="x",
                padx=self.measure_gap(2),
                pady=(self.measure_gap(1), 0),
                after=self.strip if self.strip.winfo_manager() else self.header_rule,
            )
        else:
            self.notes_frame.pack_forget()

    # Opening
    def choose_files(self) -> None:
        if self.run is not None or self.opening:
            return
        chosen = filedialog.askopenfilenames(title="Open dataset files", filetypes=FILE_TYPES)
        if chosen:
            self.open_dataset([Path(path) for path in chosen])

    def choose_folder(self) -> None:
        if self.run is not None or self.opening:
            return
        chosen = filedialog.askdirectory(title="Open a folder of dataset files", mustexist=True)
        if chosen:
            self.open_dataset([Path(chosen)])

    def open_dataset(self, chosen: list[Path], check: bool = True) -> None:
        """Unpack, check and join in a thread (a whole release takes a minute), then show it."""
        if self.run is not None or self.opening:
            return
        self.opening = True
        self._update_buttons()
        self._show_busy(True)
        self.progress.config(value=0)
        self.set_status("Opening…")

        def tell(doing: str, done: int, total: int) -> None:
            self.events.put(("progress", Progress(done, total, None, doing)))

        def work() -> None:
            try:
                opened = open_paths(chosen, progress=tell, check=check)
            except Refused as refusal:
                self.events.put(("refused", str(refusal)))
            except OSError as error:
                self.events.put(("refused", f"The dataset could not be read: {error}"))
            else:
                self.events.put(("opened", opened))

        threading.Thread(target=work, daemon=True).start()

    def _show_opened(self, opened: Opened) -> None:
        self.opened = opened
        self.use_reader(Reader(opened.data, opened.log))
        notes = list(opened.notes)
        if not opened.ours:
            notes.append(NOT_VALIDATED)
        self._show_notes(notes)
        self._fill_coverage(opened)
        shown = self.list_shown_coverage(opened)
        served = sum(item.served for item in shown)
        self.say(
            f"Opened {len(opened.paths)} file(s): {served} of {len(shown)}"
            " competitions can be forecast (see Coverage)."
        )
        try:
            state.remember(opened.chosen, self.state_file)
        except OSError as error:
            self.say(f"This dataset will not reopen by itself next time: {error}")

    def use_reader(self, reader: Reader) -> None:
        """Every screen over another dataset and its own log."""
        self.reader = reader
        self.competitions = reader.list_competitions()
        names = [competition["name"] for competition in self.competitions]
        self.competition_box.config(values=names)
        self.season_box.config(values=names)
        # the most important competition with matches around now (ours in our order of importance)
        first = next(
            (
                index
                for index, competition in enumerate(self.competitions)
                if reader.is_active(competition["competition_id"])
            ),
            0,
        )
        leagues = [i for i, c in enumerate(self.competitions) if c["kind"] == "league"]
        season = next((i for i in leagues if i >= first), leagues[0] if leagues else 0)
        for box, index in ((self.competition_box, first), (self.season_box, season)):
            box.set("")
            if self.competitions:
                box.current(index)
        self.start_panel.pack_forget()
        self.screens.pack(fill="both", expand=True)
        self._clear_match()
        self.screens.select(self.matches_screen)
        built = formats.format_local_moment(reader.built_at) if reader.built_at else "date unknown"
        files = self.opened.paths if self.opened else []
        where = files[0].name if len(files) == 1 else f"{len(files)} files"
        self.dataset_label.config(text=f"{reader.name} · built {built} · {where}")
        self.title(f"{TITLE}: {reader.name}")
        self._update_buttons()
        self.show_matches()
        self.show_season()

    # Computing
    def compute(self) -> None:
        if self.opened is None or self.run is not None or self.opening:
            return
        minutes = read_last_minutes(self.opened.log)
        whole = any(file.part == "all" for file in self.opened.files)
        said = describe_estimate(minutes, whole)
        question = (
            "Vesta computes the forecasts of this dataset's coming matches on this PC"
            f" and writes them beside the dataset. {said}\n\nStart?"
        )
        if not messagebox.askokcancel("Compute forecasts", question):
            return
        run = Run(
            self.opened.log,
            compute_command(self.opened),
            self.lines.put,
            progress=lambda found: self.events.put(("progress", found)),
        )
        try:
            run.start()
        except AlreadyRunning as refusal:
            messagebox.showinfo("Vesta not started", str(refusal))
            return
        self.run = run
        self.progress.config(value=0)
        self._show_busy(True, stoppable=True)
        self.set_status("Vesta is starting…")
        self._update_buttons()

    def stop(self) -> None:
        if self.run is not None:
            self.run.stop()
            self.set_status("Stopping…")

    def _update_buttons(self) -> None:
        busy = self.run is not None or self.opening
        ready = self.opened is not None and is_command_ready()
        self.compute_button.config(state="normal" if ready and not busy else "disabled")
        for button in (self.open_files_button, self.open_folder_button):
            button.config(state="disabled" if busy else "normal")
        if busy:
            return
        self._show_busy(False)
        if self.opened is None:
            self.set_status("")
            return
        if not is_command_ready():
            self.set_status(NO_COMMAND)
            return
        minutes = read_last_minutes(self.opened.log)
        self.set_status(
            text=""
            if minutes is None
            else f"The last run on this PC took {describe_minutes(minutes)}"
        )

    def _drain(self) -> None:
        """The run's and the opening's news, moved from their threads into the window."""
        lines = []
        while not self.lines.empty():
            lines.append(self.lines.get_nowait())
        if lines:
            self.say("\n".join(lines))
        while not self.events.empty():
            kind, payload = self.events.get_nowait()
            if kind == "progress":
                self._show_progress(payload)
            elif kind == "refused":
                self.opening = False
                self._update_buttons()
                self.say(f"Not opened: {payload}")
                messagebox.showwarning("Dataset not opened", payload)
            elif kind == "opened":
                self.opening = False
                self._show_opened(payload)
        run = self.run
        if run is not None and not run.is_running():
            self.run = None
            self._update_buttons()
            if not run.succeeded:
                self.set_status("Vesta stopped before the end: see what Vesta said below")
            self.show_matches()
            self.show_season()
        self.after(150, self._drain)

    def _show_progress(self, found: Progress) -> None:
        if found.total > 0:
            self.progress.config(value=100 * found.done / found.total)
        text = found.doing
        if found.seconds_left is not None:
            text += f" · about {max(1, round(found.seconds_left / 60))} min left"
        self.set_status(text)

    def _focus_screen(self) -> None:
        chosen = self.screens.select()
        table = {
            str(self.matches_screen): self.match_list,
            str(self.season_screen): self.season_table,
            str(self.coverage_screen): self.coverage_list,
        }.get(chosen)
        if table is not None:
            table.body.focus_set()

    def note(self, label: ttk.Label, frame: ttk.Frame, text: str) -> None:
        """A note above a screen's table, or none."""
        label.config(text=text)
        if text:
            frame.pack(
                fill="x", pady=(0, self.measure_gap(1)), before=frame.master.winfo_children()[-1]
            )
        else:
            frame.pack_forget()

    def make_note_box(self, parent: ttk.Frame) -> tuple[ttk.Frame, ttk.Label]:
        frame = ttk.Frame(
            parent, style="Note.TFrame", padding=(self.measure_gap(1.5), self.measure_gap(0.8))
        )
        label = ttk.Label(frame, style="Note.TLabel", justify="left")
        label.pack(anchor="w", fill="x")
        label.bind("<Configure>", lambda event: label.config(wraplength=max(200, event.width)))
        return frame, label

    # Matches
    def _build_matches(self) -> None:
        top = ttk.Frame(self.matches_screen)
        top.pack(fill="x", pady=(0, self.measure_gap(1.2)))
        self.competition_box = ttk.Combobox(top, state="readonly", width=30)
        self.competition_box.pack(side="left")
        self.competition_box.bind("<<ComboboxSelected>>", lambda _: self.show_matches())
        self.previous_button = ttk.Button(
            top, text="‹ Previous", command=lambda: self.move_round(-1)
        )
        self.previous_button.pack(side="left", padx=(self.measure_gap(2), 0))
        self.round_title = ttk.Label(top, style="Big.TLabel")
        self.round_title.pack(side="left", padx=(self.measure_gap(1), 0))
        self.next_button = ttk.Button(top, text="Next ›", command=lambda: self.move_round(1))
        self.next_button.pack(side="left", padx=(self.measure_gap(1), 0))
        self.round_note = ttk.Label(top, style="Muted.TLabel")
        self.round_note.pack(side="left", padx=(self.measure_gap(1.5), 0))
        self.matches_note_frame, self.matches_note = self.make_note_box(self.matches_screen)
        self.match_list = Table(
            self.matches_screen,
            self.look,
            [
                Column("kickoff", "Kick-off", 20, muted=True),
                Column("home", "Home", 14, stretch=True),
                Column("away", "Away", 14, stretch=True),
                Column("status", "Status", 20, muted=True),
                Column("home_chance", "Home wins", 16, kind=CHANCE, group="result"),
                Column("draw", "Draw", 16, kind=CHANCE, group="result"),
                Column("away_chance", "Away wins", 16, kind=CHANCE, group="result"),
                Column("made", "", 11, muted=True),
            ],
            on_activate=lambda item: self.show_match(item.key),
        )
        self.match_list.pack(fill="both", expand=True)
        # the arrows move between the rounds, as the buttons do
        self.match_list.body.bind("<Left>", lambda _: self.move_round(-1))
        self.match_list.body.bind("<Right>", lambda _: self.move_round(1))

    def find_chosen_competition(self, box: ttk.Combobox) -> dict | None:
        index = box.current()
        return self.competitions[index] if index >= 0 and self.competitions else None

    def _has_failed(self, name: str) -> bool:
        """Whether Vesta forecasts the competition of that name with a model that failed its
        tests (the national teams)."""
        coverage = self.opened.coverage if self.opened else []
        return any(item.name == name and item.validation == FAILED for item in coverage)

    def _is_served(self, name: str) -> bool | None:
        """Whether Vesta forecasts the competition of that name; None when not known."""
        if self.opened is None:
            return None
        found = [item.served for item in self.opened.coverage if item.name == name]
        return found[0] if found else None

    def show_matches(self) -> None:
        """The chosen competition's rounds, opened on the current one."""
        competition = self.find_chosen_competition(self.competition_box)
        self.rounds = None
        if competition is not None and self.reader is not None:
            self.rounds = self.reader.read_rounds(competition["competition_id"])
        self.round_index = (self.rounds.current if self.rounds else None) or 0
        self.show_round()

    def move_round(self, step: int) -> None:
        """The round before (-1) or after (+1) the one shown, within the season."""
        if self.rounds is None or not self.rounds.rounds:
            return
        index = self.round_index + step
        if 0 <= index < len(self.rounds.rounds):
            self.round_index = index
            self.show_round()

    def show_round(self) -> None:
        competition = self.find_chosen_competition(self.competition_box)
        if competition is None or self.reader is None:
            self.match_list.set_items([], "No competition in this dataset.")
            self._set_round_buttons()
            return
        name = competition["name"]
        if self.rounds is None or not self.rounds.rounds:
            self.round_title.config(text="No matches")
            self.round_note.config(text="")
            self.places = {}
            self.match_list.set_items([], f"{name} has no matches in the dataset.")
            self.note(self.matches_note, self.matches_note_frame, "")
            self._set_round_buttons()
            return
        found = self.reader.read_round(self.rounds, self.round_index)
        self._set_round_buttons()
        self.round_title.config(text=found.label)
        place = PLACE_WORDS.get(found.place) or describe_current(found)
        self.round_note.config(text=f"{found.index + 1} of {found.count} · {place}")
        self.places = {match["match_id"]: found.place for match in found.matches}
        self._fill_round(found, name)

    def _set_round_buttons(self) -> None:
        count = len(self.rounds.rounds) if self.rounds else 0
        before = "normal" if count and self.round_index > 0 else "disabled"
        after = "normal" if count and self.round_index < count - 1 else "disabled"
        self.previous_button.config(state=before)
        self.next_button.config(state=after)

    def _fill_round(self, found: RoundView, name: str) -> None:
        items, after_note, forecast = [], None, 0
        for match in found.matches:
            chances = self.reader.list_result_chances(match)
            forecast += bool(chances)
            result = match["result_forecast"] or {}
            after = bool(result.get("after_kickoff"))
            if after:
                after_note = result.get("after_kickoff_caveat") or after_note
            items.append(
                Item(
                    {
                        "kickoff": formats.format_local_moment(match["kickoff_utc"]),
                        "home": match["home_team"]["name"],
                        "away": match["away_team"]["name"],
                        "status": match["status"],
                        "home_chance": chances[0] if chances else "–",
                        "draw": chances[1] if chances else "–",
                        "away_chance": chances[2] if chances else "–",
                        "made": "after kick-off" if after else "",
                    },
                    key=match["match_id"],
                )
            )
        self.match_list.set_items(items, "No matches in this round.")
        note = ""
        if found.place == LATER and not forecast:
            note = COMES_LATER
        elif found.matches and not forecast:
            if self._is_served(name) is False:
                note = f"Vesta does not forecast {name} with this dataset: see Coverage for why."
            elif all(match["finished"] for match in found.matches):
                note = NOT_MADE_BEFORE_KICKOFF
            else:
                note = (
                    "No forecasts yet for this round. Press Compute forecasts to compute them"
                    " on this PC."
                )
        elif after_note:
            note = f"After kick-off: {after_note}"
        if self._has_failed(name):
            note = FAILED_NOTE + (f"\n{note}" if note else "")
        self.note(self.matches_note, self.matches_note_frame, note)

    def back_to_round(self) -> None:
        if self.reader is not None and self.screens.select() == str(self.match_screen):
            self.screens.select(self.matches_screen)
            self.match_list.body.focus_set()

    # Match
    def _build_match(self) -> None:
        self.match_empty = ttk.Label(self.match_screen, text=NO_MATCH, style="Muted.TLabel")
        self.match_content = ttk.Frame(self.match_screen)
        top = ttk.Frame(self.match_content)
        top.pack(fill="x")
        self.match_title = ttk.Label(top, font=self.look.fonts["title"])
        self.match_title.pack(side="left")
        ttk.Button(top, text="Back to the round (Esc)", command=self.back_to_round).pack(
            side="right"
        )
        self.match_detail = ttk.Label(self.match_content, style="Muted.TLabel")
        self.match_detail.pack(anchor="w", pady=(self.measure_gap(0.3), 0))
        self.match_made = ttk.Label(self.match_content, style="Muted.TLabel")
        self.match_made.pack(anchor="w")
        self.result_strip = tk.Canvas(
            self.match_content,
            height=self.look.convert_points(34),
            highlightthickness=0,
            borderwidth=0,
        )
        self.result_strip.pack(fill="x", pady=(self.measure_gap(1.2), self.measure_gap(0.5)))
        self.result_strip.bind("<Configure>", lambda _: self._draw_result_strip())
        self.result_shares: list[tuple[str, float, str]] = []
        self.match_caveats = ttk.Label(self.match_content, style="Muted.TLabel", justify="left")
        self.match_caveats.pack(side="bottom", anchor="w", fill="x", pady=(self.measure_gap(1), 0))
        self.match_caveats.bind(
            "<Configure>", lambda event: self.match_caveats.config(wraplength=max(200, event.width))
        )
        self.match_tabs = ttk.Notebook(self.match_content)
        self.match_tabs.pack(fill="both", expand=True, pady=(self.measure_gap(0.5), 0))
        self.match_tables: list[Table] = []

    def _clear_match(self) -> None:
        self.match_content.pack_forget()
        self.match_empty.pack(anchor="nw", pady=self.measure_gap(1))

    def show_match(self, match_id: int) -> None:
        if self.reader is None:
            return
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            view = self.reader.read_match(match_id)
        finally:
            self.config(cursor="")
        self._fill_match(view, view.summary["status"])
        self.screens.select(self.match_screen)
        if self.match_tables:
            self.match_tables[0].body.focus_set()

    def _fill_match(self, view: MatchView, status: str) -> None:
        self.match_empty.pack_forget()
        self.match_content.pack(fill="both", expand=True)
        match = view.summary
        home, away = match["home_team"]["name"], match["away_team"]["name"]
        self.match_title.config(text=f"{home} – {away}")
        detail = [match["competition"], formats.format_local_moment(match["kickoff_utc"])]
        self.match_detail.config(text=" · ".join(detail + ([status] if status else [])))
        result = match["result_forecast"]
        if view.published_at:
            when = "after kick-off" if result and result.get("after_kickoff") else "before kick-off"
            made = formats.format_local_moment(view.published_at)
            failed = f"\n{FAILED_NOTE}" if self._has_failed(match["competition"]) else ""
            self.match_made.config(
                text=f"Forecast made {made}, {when} · model {view.model_version}{failed}"
            )
        elif match["finished"]:
            self.match_made.config(text=NOT_MADE_BEFORE_KICKOFF)
        elif self.places.get(match["match_id"]) == LATER:
            self.match_made.config(text=COMES_LATER)
        elif self._is_served(match["competition"]) is False:
            self.match_made.config(
                text=f"Vesta does not forecast {match['competition']} with this"
                " dataset: see Coverage for why."
            )
        else:
            self.match_made.config(text="No forecast for this match yet: press Compute forecasts.")
        chances = self.reader.list_result_chances(match) if self.reader and result else []
        self.result_shares = (
            [
                (home, result["home"], chances[0]),
                ("Draw", result["draw"], chances[1]),
                (away, result["away"], chances[2]),
            ]
            if chances
            else []
        )
        self._draw_result_strip()
        self.match_caveats.config(text="\n".join(view.caveats))
        for tab in self.match_tabs.tabs():
            self.match_tabs.forget(tab)
        self.match_tables = []
        # only the tabs with something in them; the first one when nothing is computed
        shown = {name: rows for name, rows in view.tabs.items() if rows}
        for name, rows in shown.items() or list(view.tabs.items())[:1]:
            table = Table(
                self.match_tabs,
                self.look,
                [
                    Column("line", "", 52),
                    Column("chance", "Chance", 26, kind=CHANCE),
                    Column("rest", "", 0, stretch=True),  # the free width after the chances
                ],
                tree=True,
                empty="Nothing computed for this match.",
            )
            table.set_items(make_items(rows))
            self.match_tabs.add(table, text=name)
            self.match_tables.append(table)

    def _draw_result_strip(self) -> None:
        canvas, colours, fonts = self.result_strip, self.look.colours, self.look.fonts
        canvas.configure(background=colours.bg)
        canvas.delete("all")
        width = canvas.winfo_width()
        if not self.result_shares or width < 50:
            canvas.configure(height=1)
            return
        canvas.configure(height=self.look.convert_points(34))
        best = max(share for _, share, _ in self.result_shares)
        gap, x = self.look.convert_points(2), 0
        thick = self.look.convert_points(8)
        text_y = fonts["strong"].metrics("linespace") // 2 + 1
        for name, share, shown in self.result_shares:
            size = round(width * share)
            fill = colours.bar if share >= best else colours.bar_low
            canvas.create_rectangle(
                x,
                self.look.convert_points(22),
                x + max(1, size - gap),
                self.look.convert_points(22) + thick,
                fill=fill,
                width=0,
            )
            canvas.create_text(
                x,
                text_y,
                text=f"{name}  {shown}",
                anchor="w",
                font=fonts["strong"],
                fill=colours.text,
            )
            x += size

    # Season tables
    def _build_season(self) -> None:
        top = ttk.Frame(self.season_screen)
        top.pack(fill="x", pady=(0, self.measure_gap(1.2)))
        self.season_box = ttk.Combobox(top, state="readonly", width=30)
        self.season_box.pack(side="left")
        self.season_box.bind("<<ComboboxSelected>>", lambda _: self.show_season())
        self.season_title = ttk.Label(top, style="Big.TLabel")
        self.season_title.pack(side="left", padx=(self.measure_gap(2), 0))
        self.season_detail = ttk.Label(top, style="Muted.TLabel")
        self.season_detail.pack(side="left", padx=(self.measure_gap(1.5), 0))
        self.season_note_frame, self.season_note = self.make_note_box(self.season_screen)
        panes = ttk.PanedWindow(self.season_screen, orient="vertical")
        panes.pack(fill="both", expand=True)
        self.season_table = Table(panes, self.look, [])
        self.season_others = Table(
            panes,
            self.look,
            [
                Column("line", "Other season forecasts", 56),
                Column("chance", "Chance", 26, kind=CHANCE),
                Column("rest", "", 0, stretch=True),
            ],
            tree=True,
            empty="No other season forecasts.",
        )
        # the table first: the other forecasts below it start small
        self.season_table.body.configure(height=self.look.convert_points(400))
        self.season_others.body.configure(height=self.look.convert_points(70))
        panes.add(self.season_table, weight=1)
        panes.add(self.season_others, weight=0)

    def show_season(self) -> None:
        competition = self.find_chosen_competition(self.season_box)
        if competition is None or self.reader is None:
            return
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            view = self.reader.read_season(competition["competition_id"])
        finally:
            self.config(cursor="")
        self._fill_season(view, competition["name"])

    def _fill_season(self, view: SeasonView, name: str) -> None:
        if view.season_name is None:
            self.season_title.config(text="No season")
            self.season_detail.config(text="")
        else:
            self.season_title.config(text=f"Season {view.season_name}")
            self.season_detail.config(
                text=f"chances computed {formats.format_local_moment(view.published_at)}"
                if view.published_at
                else ""
            )
        columns = []
        for index, heading in enumerate(view.columns):
            if heading == "Team":
                columns.append(Column(f"c{index}", heading, 16, stretch=True))
            elif heading == "Group":
                columns.append(Column(f"c{index}", heading, 18, muted=True))
            elif heading in ("Place", "Played", "Points"):
                columns.append(Column(f"c{index}", heading, 5, align="e",
                                      muted=heading != "Points", strong=heading == "Points"))
            else:
                columns.append(Column(f"c{index}", heading, 7, kind=CHANCE_UNDER))
        self.season_table.columns = columns or [Column("c0", "", 10, stretch=True)]
        items = [Item({f"c{index}": value for index, value in enumerate(row)}) for row in view.rows]
        if view.season_name is None:
            empty = f"No season of {name} in the dataset."
        elif view.others:
            empty = f"No table of {name} in the dataset: its season forecasts are listed below."
        else:
            empty = f"No table for {name} in this dataset."
        self.season_table.set_items(items, empty)
        self.season_others.set_items(
            make_items(view.others, open_lines=0),
            "No other season forecasts." if view.rows else "No season forecasts.",
        )
        notes = []
        if view.partial:
            notes.append(
                f"This table holds {view.teams_in_table} of the {view.teams_playing} teams"
                f" playing this season: the dataset has the standings of part of {name} only."
            )
        note = ""
        if view.season_name is not None and not view.published_at:
            if self._is_served(name) is False:
                note = f"Vesta does not forecast {name} with this dataset: see Coverage for why."
            else:
                # Vesta's run on this PC (`vesta.run`) forecasts matches only: its season
                # chances come from logs that hold them (9 November 2026: none of its own yet)
                note = (
                    f"No season chances for {name} in the forecast log: Vesta's run on this PC"
                    " computes the matches' forecasts. The table is the dataset's."
                )
        notes.append(note)
        self.note(self.season_note, self.season_note_frame, " ".join(filter(None, notes)))

    # Coverage
    def _build_coverage(self) -> None:
        ttk.Label(self.coverage_screen, text=COVERAGE_INTRO, style="Muted.TLabel").pack(
            anchor="w", pady=(0, self.measure_gap(1))
        )
        self.coverage_why = ttk.Label(
            self.coverage_screen, text=COVERAGE_HINT, justify="left", style="Muted.TLabel"
        )
        self.coverage_why.pack(side="bottom", anchor="w", fill="x", pady=(self.measure_gap(1), 0))
        self.coverage_why.bind(
            "<Configure>", lambda event: self.coverage_why.config(wraplength=max(200, event.width))
        )
        self.files_list = Table(
            self.coverage_screen,
            self.look,
            [
                Column("file", "File", 34),
                Column("holds", "Holds", 26, stretch=True),
                Column("digest", "SHA256SUMS", 22),
                Column("ours", "11footy's", 18),
            ],
        )
        self.files_list.pack(fill="x")
        self.coverage_list = Table(
            self.coverage_screen,
            self.look,
            [
                Column("competition", "Competition", 40),
                Column("forecast", "Forecast by Vesta", 36),
                Column("why", "Why", 30, stretch=True),
            ],
            tree=True,
            on_choose=self._explain_coverage,
        )
        self.coverage_list.pack(fill="both", expand=True, pady=(self.measure_gap(2), 0))

    def _explain_coverage(self, item: Item | None) -> None:
        entry = item.key if item is not None else None
        if entry is None:
            self.coverage_why.config(text=COVERAGE_HINT)
            return
        source = "the whole file" if entry.part == WHOLE_PART else f"the file {entry.part}"
        said = f"{entry.name}, from {source}. Vesta: {entry.reason}"
        self.coverage_why.config(text=f"{said}. {entry.families}" if entry.families else said)

    def list_shown_coverage(self, opened: Opened) -> list:
        """The competitions the Coverage screen lists and counts: for our dataset those the
        Matches list shows (`Reader.list_competitions`); for someone else's every one."""
        if not opened.ours:
            return opened.coverage
        shown = {competition["code"] for competition in self.competitions}
        return [item for item in opened.coverage if item.code in shown]

    def _fill_coverage(self, opened: Opened) -> None:
        self.files_list.set_items(
            [
                Item(
                    {
                        "file": item.path.name,
                        "holds": describe_file(item, opened.coverage),
                        "digest": describe_digest(item, opened.digests_checked),
                        "ours": describe_proof(item.proof.level),
                    }
                )
                for item in opened.files
            ]
        )
        shown = self.list_shown_coverage(opened)
        served = [item for item in shown if item.served]
        others = [item for item in shown if not item.served]

        def make_rows(entries: list) -> list[Item]:
            return [
                Item(
                    {
                        "competition": entry.name,
                        "forecast": describe_forecast(entry),
                        "why": describe_reason(entry),
                    },
                    key=entry,
                    muted=not entry.served,
                )
                for entry in sorted(entries, key=lambda entry: entry.name)
            ]

        groups = []
        if served:
            groups.append(
                Item(
                    {"competition": f"Forecast by Vesta ({len(served)})"},
                    make_rows(served),
                    open=True,
                    heading=True,
                )
            )
        if others:
            groups.append(
                Item(
                    {"competition": f"Not forecast ({len(others)})"},
                    make_rows(others),
                    open=len(others) <= OPEN_LINES or not served,
                    heading=True,
                )
            )
        self.coverage_list.set_items(groups, "No competition in this dataset.")
        self._explain_coverage(None)
        # the files' list as tall as its rows, up to six
        rows = min(6, max(1, len(opened.files)))
        self.files_list.body.configure(height=rows * self.files_list.row_height)


def describe_current(found: RoundView) -> str:
    """The current round in a word: "played" once every match of it is finished, called off or
    played without its result yet (the season's last round), "in progress" once a match of it
    kicked off, else "next"."""
    called_off = ("Postponed", "Cancelled", "Abandoned", "Suspended")
    over = [
        match["finished"] or match["status"] in (*called_off, "Played, no result yet")
        for match in found.matches
    ]
    if found.matches and all(over):
        return PLACE_WORDS[PAST]
    started = (match["status"] not in ("To play", *called_off) for match in found.matches)
    return "in progress" if any(started) else "next"


def describe_estimate(minutes: float | None, whole: bool) -> str:
    """What the window says of the coming run's length: the last run's on this PC, else Vesta's
    measure for the whole dataset, else for one league."""
    if minutes is not None:
        return f"The last run on this PC took {describe_minutes(minutes)}."
    if whole:
        return WHOLE_FIRST_RUN
    return PART_FIRST_RUN


def describe_minutes(minutes: float) -> str:
    """A run's length in words: "less than a minute", "1 minute", "3 minutes"."""
    if minutes < 0.5:
        return "less than a minute"
    whole = round(minutes)
    return f"{whole} minute" + ("" if whole == 1 else "s")


def make_items(rows: list[Row], open_lines: int = OPEN_LINES) -> list[Item]:
    """A screen's rows as a table's: a section opens by itself when its lines are few."""

    def convert(row: Row, depth: int) -> Item:
        children = [convert(child, depth + 1) for child in row.children]
        leaves = bool(children) and not any(child.children for child in children)
        return Item(
            {"line": row.text or row.line, "chance": row.chance},
            children,
            open=leaves and depth == 0 and len(children) <= open_lines,
            heading=bool(children) and depth == 0,
        )

    return [convert(row, 0) for row in rows]
