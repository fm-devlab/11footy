"""The window's tables, drawn on a canvas: a fixed header, shaded rows, chances as percentages
with a bar, sections that open and close, a chosen row, and the keyboard (arrows, Page Up and
Down, Home, End, Enter, Right to open, Left to close).

A Treeview cannot draw a bar; this draws what the screens need and nothing more. Widths are in
digits of the body font, so they follow the screen's scaling; text too long for its column ends
in an ellipsis instead of running under the next one.
"""

import sys
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass, field
from tkinter import ttk

from desk.theme import Look

ELLIPSIS = "…"
OPEN, CLOSED = "▾", "▸"
CHANCE = "chance"  # the percentage, its bar beside it
CHANCE_UNDER = "chance_under"  # the percentage, a thin bar beneath it (narrow columns)
TEXT = "text"


@dataclass
class Column:
    key: str
    heading: str
    chars: int  # width in digits of the body font
    align: str = "w"
    kind: str = TEXT
    stretch: bool = False  # takes the width left over
    muted: bool = False
    strong: bool = False
    group: str | None = None  # chance columns of one row compared: the largest bar is stronger


@dataclass
class Item:
    values: dict[str, str]
    children: list["Item"] = field(default_factory=list)
    key: object = None  # what the screen knows the row by
    open: bool = False
    muted: bool = False
    heading: bool = False  # a section's row, in the strong font


def read_percent(text: str) -> float | None:
    """A chance as shown ("43%", "<1%", ">99%") as a share of the bar; None for no chance."""
    text = text.strip()
    if not text.endswith("%"):
        return None
    try:
        number = float(text.strip("%<> "))
    except ValueError:
        return None
    if text.startswith("<"):
        number = min(number, 0.5)
    elif text.startswith(">"):
        number = max(number, 99.5)
    return max(0.0, min(1.0, number / 100))


class Table(ttk.Frame):
    def __init__(
        self,
        parent: tk.Misc,
        look: Look,
        columns: list[Column],
        on_activate: Callable[[Item], None] | None = None,
        on_choose: Callable[[Item | None], None] | None = None,
        empty: str = "",
        tree: bool = False,
    ) -> None:
        super().__init__(parent)
        self.look = look
        self.columns = columns
        self.on_activate = on_activate
        self.on_choose = on_choose
        self.empty = empty
        self.tree = tree
        self.items: list[Item] = []
        self.shown: list[tuple[Item, int]] = []  # the rows on screen, with their depth
        self.chosen: int | None = None
        self.widths: list[int] = []
        self.header = tk.Canvas(self, highlightthickness=0, borderwidth=0, takefocus=0)
        self.body = tk.Canvas(self, highlightthickness=0, borderwidth=0, takefocus=1)
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.body.yview)
        self.body.configure(yscrollcommand=self.scroll.set)
        self.header.grid(row=0, column=0, sticky="ew")
        self.body.grid(row=1, column=0, sticky="nsew")
        self.scroll.grid(row=1, column=1, sticky="ns")
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)
        self._last_width = 0
        self.body.bind("<Configure>", self._resized)
        self.body.bind("<Motion>", self._hover)
        self.body.bind("<Leave>", lambda _: self.body.itemconfigure("hover", state="hidden"))
        self.body.bind("<Button-1>", self._click)
        self.body.bind("<Double-Button-1>", lambda event: self._activate(self._row_at(event.y)))
        self.body.bind("<Return>", lambda _: self._activate(self.chosen))
        self.body.bind("<KP_Enter>", lambda _: self._activate(self.chosen))
        self.body.bind("<space>", lambda _: self._toggle(self.chosen))
        self.body.bind("<Up>", lambda _: self._move(-1))
        self.body.bind("<Down>", lambda _: self._move(1))
        self.body.bind("<Prior>", lambda _: self._move(-self._page()))
        self.body.bind("<Next>", lambda _: self._move(self._page()))
        self.body.bind("<Home>", lambda _: self._move(-len(self.shown)))
        self.body.bind("<End>", lambda _: self._move(len(self.shown)))
        self.body.bind("<Right>", lambda _: self._open_chosen(True))
        self.body.bind("<Left>", lambda _: self._open_chosen(False))
        self.body.bind("<FocusIn>", lambda _: self._focus_in())
        self.body.bind("<FocusOut>", lambda _: self._place_chosen())
        # the wheel scrolls the table under the pointer, whichever widget has the focus
        self.bind("<Enter>", lambda _: self._take_wheel(True))
        self.bind("<Leave>", lambda _: self._take_wheel(False))
        look.follow(self.redraw)
        self.bind("<Destroy>", self._destroyed)
        self.redraw()

    def _destroyed(self, event: tk.Event) -> None:
        if event.widget is self and self.redraw in self.look.listeners:
            self.look.listeners.remove(self.redraw)

    def _take_wheel(self, taking: bool) -> None:
        if sys.platform.startswith("linux"):
            for sequence, step in (("<Button-4>", -1), ("<Button-5>", 1)):
                if taking:
                    self.bind_all(sequence, lambda _, step=step: self._wheel(step))
                else:
                    self.unbind_all(sequence)
        elif taking:
            self.bind_all("<MouseWheel>", self._wheel_event)
        else:
            self.unbind_all("<MouseWheel>")

    # What it shows
    def set_items(self, items: list[Item], empty: str | None = None) -> None:
        self.items = items
        if empty is not None:
            self.empty = empty
        self.chosen = None
        self._flatten()
        self.body.yview_moveto(0)
        self.redraw()

    def find_chosen_item(self) -> Item | None:
        return self.shown[self.chosen][0] if self.chosen is not None else None

    def choose_key(self, key: object) -> None:
        """Choose the row the screen knows by `key`, if shown."""
        for index, (item, _) in enumerate(self.shown):
            if item.key == key:
                self._choose(index)
                return

    def measure_natural_height(self, rows: int) -> int:
        """The height that shows the header and this many rows."""
        header = self.look.fonts["heading"].metrics("linespace") + self.look.convert_points(9)
        return header + rows * self.row_height + 2

    def _flatten(self) -> None:
        shown: list[tuple[Item, int]] = []

        def walk(items: list[Item], depth: int) -> None:
            for item in items:
                shown.append((item, depth))
                if item.children and item.open:
                    walk(item.children, depth + 1)

        walk(self.items, 0)
        self.shown = shown

    # Sizes
    @property
    def row_height(self) -> int:
        return self.look.fonts["body"].metrics("linespace") + self.look.convert_points(7)

    @property
    def pad(self) -> int:
        return self.look.convert_points(6)

    def _layout(self, width: int) -> None:
        digit = self.look.fonts["body"].measure("0")
        heading = self.look.fonts["heading"]
        wanted = [
            max(column.chars * digit, heading.measure(column.heading)) + 2 * self.pad
            for column in self.columns
        ]  # a heading is never cut
        spare = width - sum(wanted)
        stretching = [i for i, column in enumerate(self.columns) if column.stretch]
        if stretching:
            share = spare // len(stretching)
            for index in stretching:
                wanted[index] = max(
                    digit * min(8, self.columns[index].chars), wanted[index] + share
                )
        self.widths = wanted

    # Drawing
    def redraw(self) -> None:
        colours = self.look.colours
        self.header.configure(background=colours.panel)
        self.body.configure(background=colours.panel, yscrollincrement=self.row_height)
        width = max(self.body.winfo_width(), 50)
        self._layout(width)
        self._draw_header(width)
        self._draw_body(width)

    def _draw_header(self, width: int) -> None:
        colours, fonts = self.look.colours, self.look.fonts
        height = fonts["heading"].metrics("linespace") + self.look.convert_points(9)
        self.header.configure(height=height)
        self.header.delete("all")
        x = 0
        for column, size in zip(self.columns, self.widths, strict=True):
            text = self._fit(column.heading, size - 2 * self.pad, fonts["heading"])
            right = column.align == "e" or column.kind == CHANCE_UNDER
            self.header.create_text(
                x + size - self.pad if right else x + self.pad,
                height // 2,
                text=text,
                anchor="e" if right else "w",
                font=fonts["heading"],
                fill=colours.muted,
            )
            x += size
        self.header.create_line(0, height - 1, width, height - 1, fill=colours.line)

    def _draw_body(self, width: int) -> None:
        colours, fonts = self.look.colours, self.look.fonts
        canvas, height = self.body, self.row_height
        canvas.delete("all")
        if not self.shown:
            canvas.create_text(
                self.pad * 2,
                height,
                text=self.empty,
                anchor="nw",
                font=fonts["body"],
                fill=colours.muted,
                width=max(200, width - self.pad * 4),
            )
            canvas.configure(scrollregion=(0, 0, width, height * 3))
            self._show_scroll(False)
            return
        for index in range(1, len(self.shown), 2):
            canvas.create_rectangle(
                0,
                index * height,
                width,
                (index + 1) * height,
                fill=colours.alt,
                width=0,
                tags="shade",
            )
        canvas.create_rectangle(
            0, 0, width, height, fill=colours.hover, width=0, tags="hover", state="hidden"
        )
        canvas.create_rectangle(
            0, 0, width, height, fill=colours.chosen, width=0, tags="chosen", state="hidden"
        )
        for index, (item, depth) in enumerate(self.shown):
            self._draw_row(item, depth, index * height, height)
        canvas.configure(scrollregion=(0, 0, width, len(self.shown) * height))
        self._show_scroll(len(self.shown) * height > max(self.body.winfo_height(), 1))
        self._place_chosen()

    def _show_scroll(self, needed: bool) -> None:
        """The scroll bar only when the rows do not fit."""
        if needed and not self.scroll.winfo_ismapped():
            self.scroll.grid()
        elif not needed and self.scroll.winfo_ismapped():
            self.scroll.grid_remove()

    def _draw_row(self, item: Item, depth: int, top: int, height: int) -> None:
        colours, fonts, canvas, pad = self.look.colours, self.look.fonts, self.body, self.pad
        middle = top + height // 2
        compared: dict[str, float] = {}
        for column in self.columns:
            share = read_percent(item.values.get(column.key, ""))
            if column.group and share is not None:
                compared[column.group] = max(compared.get(column.group, 0.0), share)
        x = 0
        for position, (column, size) in enumerate(zip(self.columns, self.widths, strict=True)):
            text = item.values.get(column.key, "")
            left = x + pad
            if position == 0 and self.tree:
                left += depth * fonts["body"].measure("0000")
                if item.children:
                    canvas.create_text(
                        left,
                        middle,
                        text=OPEN if item.open else CLOSED,
                        anchor="w",
                        font=fonts["body"],
                        fill=colours.muted,
                    )
                left += fonts["body"].measure("00")
            if column.kind in (CHANCE, CHANCE_UNDER) and text:
                self._draw_chance(column, text, x, size, middle, compared)
            elif text:
                chosen_font = fonts["strong"] if (column.strong or item.heading) else fonts["body"]
                muted = column.muted or item.muted
                room = x + size - pad - left
                shown = self._fit(text, room, chosen_font)
                anchor_x = x + size - pad if column.align == "e" else left
                canvas.create_text(
                    anchor_x,
                    middle,
                    text=shown,
                    anchor=column.align,
                    font=chosen_font,
                    fill=colours.muted if muted else colours.text,
                )
            x += size

    def _draw_chance(
        self, column: Column, text: str, x: int, size: int, middle: int, compared: dict[str, float]
    ) -> None:
        colours, fonts, canvas, pad = self.look.colours, self.look.fonts, self.body, self.pad
        share = read_percent(text)
        number_font = fonts["strong"]
        if share is None:
            right = column.kind == CHANCE_UNDER
            canvas.create_text(
                x + size - pad if right else x + pad,
                middle,
                text=text,
                anchor="e" if right else "w",
                font=fonts["body"],
                fill=colours.muted,
            )
            return
        top = column.group is not None and share >= compared.get(column.group, 2.0) - 1e-9
        fill = colours.bar if top else colours.bar_low
        thick = self.look.convert_points(4)
        if column.kind == CHANCE:
            number_right = x + pad + number_font.measure(">99%")
            canvas.create_text(
                number_right, middle, text=text, anchor="e", font=number_font, fill=colours.text
            )
            start, end = number_right + pad, x + size - pad
            if end - start > 8:
                canvas.create_rectangle(
                    start,
                    middle - thick // 2,
                    end,
                    middle + thick - thick // 2,
                    fill=colours.track,
                    width=0,
                )
                canvas.create_rectangle(
                    start,
                    middle - thick // 2,
                    start + max(2, round((end - start) * share)),
                    middle + thick - thick // 2,
                    fill=fill,
                    width=0,
                )
            return
        line = self.look.fonts["body"].metrics("linespace")
        canvas.create_text(
            x + size - pad,
            middle - self.look.convert_points(1.5),
            text=text,
            anchor="e",
            font=number_font,
            fill=colours.text,
        )
        start, end = x + pad * 2, x + size - pad
        base = middle + line // 2
        thin = max(2, thick // 2)
        canvas.create_rectangle(start, base, end, base + thin, fill=colours.track, width=0)
        canvas.create_rectangle(
            start,
            base,
            start + max(1, round((end - start) * share)),
            base + thin,
            fill=fill,
            width=0,
        )

    def _fit(self, text: str, room: int, chosen_font) -> str:
        """The text, cut to its room with an ellipsis when it is too long."""
        if room <= 0:
            return ""
        if chosen_font.measure(text) <= room:
            return text
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if chosen_font.measure(text[:middle].rstrip() + ELLIPSIS) <= room:
                low = middle
            else:
                high = middle - 1
        return text[:low].rstrip() + ELLIPSIS if low else ""

    def _place_chosen(self) -> None:
        canvas, height = self.body, self.row_height
        if self.chosen is None or not self.shown:
            canvas.itemconfigure("chosen", state="hidden")
            return
        top = self.chosen * height
        canvas.coords("chosen", 0, top, max(self.body.winfo_width(), 50), top + height)
        canvas.itemconfigure("chosen", state="normal")

    # Events
    def _resized(self, event: tk.Event) -> None:
        if event.width != self._last_width:
            self._last_width = event.width
            self.redraw()
        else:
            self._show_scroll(len(self.shown) * self.row_height > event.height)

    def _row_at(self, y: int) -> int | None:
        index = int(self.body.canvasy(y) // self.row_height)
        return index if 0 <= index < len(self.shown) else None

    def _hover(self, event: tk.Event) -> None:
        index = self._row_at(event.y)
        if index is None:
            self.body.itemconfigure("hover", state="hidden")
            return
        top = index * self.row_height
        self.body.coords("hover", 0, top, self.body.winfo_width(), top + self.row_height)
        self.body.itemconfigure("hover", state="normal")

    def _click(self, event: tk.Event) -> None:
        self.body.focus_set()
        index = self._row_at(event.y)
        if index is None:
            return
        item = self.shown[index][0]
        self._choose(index)
        if item.children:
            self._toggle(index)

    def _choose(self, index: int | None) -> None:
        if index is None or not self.shown:
            return
        self.chosen = max(0, min(index, len(self.shown) - 1))
        self._place_chosen()
        self._see(self.chosen)
        if self.on_choose:
            self.on_choose(self.find_chosen_item())

    def _see(self, index: int) -> None:
        total = len(self.shown) * self.row_height
        if total <= 0 or self.body.winfo_height() < self.row_height:
            return  # not on screen yet: nothing to scroll to
        top, bottom = (self.body.canvasy(0), self.body.canvasy(self.body.winfo_height()))
        row_top = index * self.row_height
        if row_top < top:
            self.body.yview_moveto(row_top / total)
        elif row_top + self.row_height > bottom:
            self.body.yview_moveto((row_top + self.row_height - self.body.winfo_height()) / total)

    def _move(self, step: int) -> str:
        if self.shown:
            self._choose((self.chosen if self.chosen is not None else -1 if step > 0 else 0) + step)
        return "break"

    def _page(self) -> int:
        return max(1, self.body.winfo_height() // self.row_height - 1)

    def _focus_in(self) -> None:
        if self.chosen is None and self.shown:
            self._choose(0)
        else:
            self._place_chosen()

    def _toggle(self, index: int | None) -> None:
        if index is None or index >= len(self.shown):
            return
        item = self.shown[index][0]
        if not item.children:
            return
        item.open = not item.open
        self._flatten()
        self.redraw()
        self.chosen = index
        self._place_chosen()

    def _open_chosen(self, opening: bool) -> str:
        item = self.find_chosen_item()
        if item is not None and item.children and item.open != opening:
            self._toggle(self.chosen)
        elif not opening and item is not None and self.chosen is not None:
            depth = self.shown[self.chosen][1]
            for index in range(self.chosen - 1, -1, -1):  # Left on a line: to its section
                if self.shown[index][1] < depth:
                    self._choose(index)
                    break
        return "break"

    def _activate(self, index: int | None) -> str:
        if index is None or index >= len(self.shown):
            return "break"
        item = self.shown[index][0]
        if item.children:
            self._toggle(index)
        elif self.on_activate:
            self.on_activate(item)
        return "break"

    def _wheel_event(self, event: tk.Event) -> str:
        step = -event.delta if sys.platform == "darwin" else -event.delta // 120
        return self._wheel(step * 3 if sys.platform != "darwin" else step)

    def _wheel(self, step: int) -> str:
        if len(self.shown) * self.row_height > self.body.winfo_height():
            self.body.yview_scroll(step, "units")
        return "break"
