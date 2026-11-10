"""The window's look: the Sun Valley theme (sv-ttk), light or dark, its fonts and the colours of
what the window draws itself (the tables with their bars).

Light unless the person chose otherwise (kept in state.json) or the system says dark where it
can be read: Windows' "apps use light theme", macOS's dark appearance, GNOME's colour scheme.
Fonts are set in points, so they follow the screen's scaling (sv-ttk's own are in pixels).
"""

import subprocess
import sys
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass
from tkinter import font, ttk

import sv_ttk

LIGHT, DARK = "light", "dark"


@dataclass(frozen=True)
class Colours:
    bg: str  # the window
    panel: str  # tables and the header
    alt: str  # every other row
    hover: str
    chosen: str  # the chosen row
    line: str  # rules and borders
    text: str
    muted: str
    bar: str  # the largest chance of a row
    bar_low: str
    track: str  # behind a bar
    note: str  # a note's background


PALETTES = {
    LIGHT: Colours(
        bg="#fafafa",
        panel="#ffffff",
        alt="#f5f5f5",
        hover="#ececec",
        chosen="#dbe8f7",
        line="#e3e3e3",
        text="#1b1b1b",
        muted="#6e6e6e",
        bar="#005fb8",
        bar_low="#9cc3e6",
        track="#e8e8e8",
        note="#f0f4fa",
    ),
    DARK: Colours(
        bg="#1c1c1c",
        panel="#232323",
        alt="#2a2a2a",
        hover="#333333",
        chosen="#1f3a52",
        line="#3a3a3a",
        text="#f3f3f3",
        muted="#a9a9a9",
        bar="#57c8ff",
        bar_low="#2f6a88",
        track="#363636",
        note="#22303b",
    ),
}

# The window's fonts, in points: (sv-ttk's font it redefines, or a new one), size, weight
FONTS = {
    "body": ("SunValleyBodyFont", 10, "normal"),
    "strong": ("SunValleyBodyStrongFont", 10, "bold"),
    "caption": ("SunValleyCaptionFont", 9, "normal"),
    "large": ("SunValleyBodyLargeFont", 12, "normal"),
    "subtitle": ("SunValleySubtitleFont", 13, "bold"),
    "title": ("SunValleyTitleFont", 17, "bold"),
    "heading": ("DeskHeadingFont", 9, "bold"),
}


def read_system_theme() -> str | None:
    """Whether the system asks apps for dark, where it can be read; None where it cannot."""
    try:
        if sys.platform == "win32":
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows"
                r"\CurrentVersion\Themes\Personalize",
            ) as key:
                light, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return LIGHT if light else DARK
        if sys.platform == "darwin":
            said = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            return DARK if said.stdout.strip().lower() == "dark" else LIGHT
        said = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if said.returncode == 0:
            return DARK if "dark" in said.stdout else LIGHT
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return None


def choose_theme(remembered: str | None) -> str:
    """The person's own choice first, then the system's, then light."""
    if remembered in PALETTES:
        return remembered
    return read_system_theme() or LIGHT


class Look:
    """The theme of one window: apply it, switch it, and the fonts and colours it draws with."""

    def __init__(self, root: tk.Tk, name: str) -> None:
        self.root = root
        self.name = name
        self.fonts: dict[str, font.Font] = {}
        self.listeners: list[Callable[[], None]] = []
        self.apply(name)

    @property
    def colours(self) -> Colours:
        return PALETTES[self.name]

    def apply(self, name: str) -> None:
        self.name = name
        sv_ttk.set_theme(name, self.root)
        family = font.nametofont("SunValleyBodyFont").actual("family")
        for key, (named, size, weight) in FONTS.items():
            try:
                found = font.nametofont(named)
                found.configure(family=family, size=size, weight=weight)
            except tk.TclError:
                found = font.Font(self.root, name=named, family=family, size=size, weight=weight)
            self.fonts[key] = found
        for default_font in ("TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont"):
            font.nametofont(default_font).configure(family=family, size=10)
        self._style()
        self.root.configure(background=self.colours.bg)
        for listener in self.listeners:
            listener()

    def follow(self, listener: Callable[[], None]) -> None:
        """Call `listener` each time the look changes: what draws itself redraws."""
        self.listeners.append(listener)

    def _style(self) -> None:
        colours = self.colours
        style = ttk.Style(self.root)
        style.configure("TNotebook.Tab", padding=(14, 5))
        style.configure("Header.TFrame", background=colours.panel)
        style.configure("Header.TLabel", background=colours.panel, foreground=colours.text)
        style.configure("HeaderMuted.TLabel", background=colours.panel, foreground=colours.muted)
        style.configure(
            "Title.TLabel",
            background=colours.panel,
            foreground=colours.text,
            font=self.fonts["title"],
        )
        style.configure("Header.Switch.TCheckbutton", background=colours.panel)
        style.configure("Muted.TLabel", foreground=colours.muted)
        style.configure("Big.TLabel", font=self.fonts["subtitle"])
        style.configure("Strong.TLabel", font=self.fonts["strong"])
        style.configure("Note.TFrame", background=colours.note)
        style.configure("Note.TLabel", background=colours.note, foreground=colours.text)
        style.configure("Rule.TFrame", background=colours.line)

    def convert_points(self, points: float) -> int:
        """A length given in points, in this screen's pixels."""
        return round(points * float(self.root.tk.call("tk", "scaling")))
