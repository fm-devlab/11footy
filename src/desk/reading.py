"""What each screen shows: one dataset in the Vesta data format (`docs/VESTA_DATA_FORMAT.md`) and
the forecast log Vesta writes beside it on this PC (`desk.log`).

Nothing here computes a probability: every chance is a line Vesta's reader reads from a
published distribution, written as `desk.formats` says. The screens get plain
rows: a label, the line in words and the chance as shown.

A dataset holds the format's required columns and may hold more. Ours (`11footy`) carries
competition, team and player names, squads and standings; someone else's may carry a
competition's code and a team's name only. So each optional table or column is read when it is
there, and the screens fall back to the code, "player N" or no table when it is not.
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from desk import formats, rounds
from desk.database import Database
from desk.log import ForecastLog
from desk.order import rank_by_importance, read_competition_order, read_competition_shown
from vesta.data_format import OUR_DATASET
from vesta.forecast_reading import LineError, read_line

# The tabs of a match, each catalogue item in one; an item not named
# here goes to "Other", so nothing Vesta publishes is left out.
TABS = {
    "Result": ["M1", "M6", "M14"],
    "Goals": ["M2", "M3", "M4", "M5", "M12", "M13", "M22", "M23", "M26"],
    "Shots": ["M20"],
    "Corners": ["M15", "M16"],
    "Cards": ["M17", "M18"],
    "Fouls": ["M21"],
    "Offsides": ["M19"],
    "Timing": ["M7", "M8", "M9", "M10", "M11"],
    "Lineups": ["L"],
    "Players": ["P"],
    "Other": [],
}
PLAYER_TABS = ("Lineups", "Players")
# The matches read around now to find the current round: wider than any round.
# A competition with a match this close to now is under way: the window opens on the first one.
ACTIVE_REACH = timedelta(days=14)
# a competition is listed when it has a match in the last four years before the dataset was built
RECENT_REACH = timedelta(days=4 * 365 + 1)
# a tournament of ours played every few years is listed from 90 days before a match to 30 after
BEFORE_MATCHES = timedelta(days=90)
AFTER_MATCHES = timedelta(days=30)
FINAL_POSITION = "T1.final_position"
# the standings' columns that name a team's part of a season: its league (Nations League), group
GROUPING = ("league_name", "group_name")
# a table holding fewer than this share of the season's teams covers part of it only
WHOLE_TABLE_SHARE = 0.75


def find_log(dataset: Path) -> Path:
    """A dataset's own forecast log, beside it: Vesta writes it on this PC alone."""
    return dataset.with_name(dataset.name + ".forecast-log.db")


def find_tab(catalogue_id: str) -> str:
    for tab, ids in TABS.items():
        if catalogue_id in ids or catalogue_id.rstrip("0123456789") in ids:
            return tab
    return "Other"


def rank_by_catalogue(item: dict) -> tuple:
    """A tab's quantities in the catalogue's order (M1, M2, ... M10), then by key."""
    catalogue_id = item["catalogue_id"]
    letters = catalogue_id.rstrip("0123456789")
    return letters, int(catalogue_id[len(letters) :] or 0), item["quantity_key"]


def format_utc(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def name_cut(line_key: str) -> str | None:
    """A season line as a column heading ("top=4" as "Top 4"); None for a line that is not a
    cut of the table."""
    kind, _, value = line_key.partition("=")
    if kind == "place" and value == "1":
        return "Champion"
    if kind in ("top", "bottom") and value.isdigit():
        return f"{kind.capitalize()} {value}"
    return None


def rank_cut(line_key: str) -> tuple:
    """The champion first, then the top cuts, then the bottom ones, each by its place."""
    kind, _, value = line_key.partition("=")
    return kind != "place", kind == "bottom", int(value)


@dataclass
class Row:
    """One row of a screen's table; its children open beneath it."""

    text: str
    line: str = ""
    chance: str = ""
    children: list["Row"] = field(default_factory=list)


# Where a round stands beside the current one (`desk.rounds`)
PAST, CURRENT, LATER = "past", "current", "later"


@dataclass
class Rounds:
    """Every round of a competition's season, in start order, and the current one's place."""

    rounds: list[rounds.Round]
    current: int | None


@dataclass
class RoundView:
    label: str  # "Matchday 10", "League phase, matchday 4", or its dates
    matches: list[dict]
    place: str = CURRENT  # PAST, CURRENT or LATER
    index: int = 0
    count: int = 0


@dataclass
class MatchView:
    summary: dict
    published_at: str | None
    model_version: str | None
    caveats: list[str]
    tabs: dict[str, list[Row]]


@dataclass
class SeasonView:
    season_name: str | None
    published_at: str | None
    columns: list[str]
    rows: list[list[str]]
    others: list[Row]
    teams_in_table: int = 0
    teams_playing: int = 0  # the teams of the season's matches

    @property
    def partial(self) -> bool:
        """The table holds clearly fewer teams than the season's matches involve (a dataset with
        the standings of one group of a qualifying competition)."""
        return bool(self.rows) and self.teams_in_table < WHOLE_TABLE_SHARE * self.teams_playing


class Reader:
    """A dataset and its forecast log, both read-only. `data` is one dataset file, or the copy
    Vesta joins a set of files into (`desk.opening`)."""

    def __init__(self, data: Path, log: Path) -> None:
        self.database = Database(data)
        self.log = ForecastLog(log)
        header = self.database.read_first_row("select * from format_version limit 1")
        self.name = header["dataset"] if header else data.stem
        self.built_at = header["built_at"] if header else None
        self.ours = self.name == OUR_DATASET

    # Competitions and matches
    def list_competitions(self, now: datetime | None = None) -> list[dict]:
        """The dataset's competitions still played: those with a match in the four years before
        the dataset was built (`RECENT_REACH`; before now when the file does not say when),
        named (by code when they have no names). Ours only as `config/competition-order.csv`
        shows them (`desk.order`), in its order of importance; someone else's all, by name."""
        now = now or datetime.now(UTC)
        named = "name" in self.database.read_columns("competitions")
        order = read_competition_order() if self.ours else {}
        shown = read_competition_shown() if self.ours else {}
        found = [
            {
                "competition_id": row["competition_id"],
                "code": row["code"],
                "name": row["name"] if named and row["name"] else row["code"],
                "kind": row["kind"],
                "display_order": order.get(row["code"]),
            }
            for row in self.database.read_rows(
                "select * from competitions where competition_id in"
                " (select s.competition_id from seasons s join matches m using (season_id)"
                " where m.kickoff_utc >= ?)",
                format_utc(self._read_built_moment() - RECENT_REACH),
            )
            if not self.ours or self._is_shown(row, shown.get(row["code"], "no"), now)
        ]
        return sorted(found, key=rank_by_importance)

    def _is_shown(self, competition: sqlite3.Row, rule: str, now: datetime) -> bool:
        if rule == "always":
            return True
        if rule == "around_matches":
            return bool(self._find_matches_around(competition["competition_id"], now,
                                                  BEFORE_MATCHES, AFTER_MATCHES))
        return False

    def _read_built_moment(self) -> datetime:
        try:
            return datetime.fromisoformat(self.built_at) if self.built_at else datetime.now(UTC)
        except ValueError:
            return datetime.now(UTC)

    def read_rounds(self, competition_id: int, now: datetime | None = None) -> Rounds:
        """Every round of the competition's season nearest to now (the one under way, else the
        one just over or about to start) and the current one's place (`desk.rounds`)."""
        now = now or datetime.now(UTC)
        competition = self.database.read_first_row(
            "select kind from competitions where competition_id = ?", competition_id
        )
        kind = (competition["kind"] if competition else None) or ""
        gap = rounds.WINDOW_GAP if "nations" in kind or kind == "friendly" else rounds.DATE_GAP
        nearest = self.database.read_first_row(
            "select season_id from matches where season_id in"
            " (select season_id from seasons where competition_id = ?)"
            " order by abs(julianday(kickoff_utc) - julianday(?)), match_id limit 1",
            competition_id,
            format_utc(now),
        )
        if nearest is None:
            return Rounds([], None)
        rows = self.database.read_rows(
            "select * from matches where season_id = ? order by kickoff_utc, match_id",
            nearest["season_id"],
        )
        grouped = rounds.group_rounds(rows, gap, formats.format_local_date)
        return Rounds(grouped, rounds.find_current_index(grouped, now))

    def read_round(self, found: Rounds, index: int, now: datetime | None = None) -> RoundView:
        """One round of `found`, whole: every match with its status and its last forecast
        before kick-off."""
        now = now or datetime.now(UTC)
        chosen = found.rounds[index]
        matches = [self._summarise(row) | {"status": rounds.describe_status(row, now)}
                   for row in chosen.matches]  # fmt: skip
        current = found.current if found.current is not None else index
        place = PAST if index < current else LATER if index > current else CURRENT
        return RoundView(chosen.label, matches, place, index, len(found.rounds))

    def read_current_round(self, competition_id: int, now: datetime | None = None) -> RoundView:
        """The competition's current round (`read_rounds`); an empty view without rounds."""
        found = self.read_rounds(competition_id, now)
        if found.current is None:
            return RoundView("", [], CURRENT, 0, 0)
        return self.read_round(found, found.current, now)

    def is_active(self, competition_id: int, now: datetime | None = None) -> bool:
        return bool(
            self._find_matches_around(competition_id, now or datetime.now(UTC), ACTIVE_REACH)
        )

    def _find_matches_around(self, competition_id: int, now: datetime, reach: timedelta,
                             reach_back: timedelta | None = None) -> list:
        """The competition's matches from `reach_back` (else `reach`) before now to `reach`
        after."""
        return self.database.read_rows(
            "select * from matches where kickoff_utc >= ? and kickoff_utc < ? and season_id in"
            " (select season_id from seasons where competition_id = ?)"
            " order by kickoff_utc, match_id",
            format_utc(now - (reach if reach_back is None else reach_back)),
            format_utc(now + reach),
            competition_id,
        )

    def read_team_name(self, team_id: int) -> str:
        row = self.database.read_first_row("select name from teams where team_id = ?", team_id)
        return row["name"] if row else f"team {team_id}"

    def read_player_name(self, player_id: int) -> str:
        """A player's name when the dataset has names; the log may still name him by an id a
        merge of two records retired (`player_merges`)."""
        if "name" not in self.database.read_columns("players"):
            return f"player {player_id}"
        if self.database.read_columns("player_merges"):
            merged = self.database.read_first_row(
                "select kept_id from player_merges where retired_id = ?", player_id
            )
            player_id = merged["kept_id"] if merged else player_id
        row = self.database.read_first_row(
            "select name from players where player_id = ?", player_id
        )
        return row["name"] if row and row["name"] else f"player {player_id}"

    def _summarise(self, row: sqlite3.Row) -> dict:
        competition = self.database.read_first_row(
            "select c.* from seasons s join competitions c using (competition_id)"
            " where s.season_id = ?",
            row["season_id"],
        )
        named = "name" in self.database.read_columns("competitions") and competition["name"]
        home, away = row["home_team_id"], row["away_team_id"]
        return {
            "match_id": row["match_id"],
            "competition": competition["name"] if named else competition["code"],
            "kickoff_utc": row["kickoff_utc"],
            "home_team": {"team_id": home, "name": self.read_team_name(home)},
            "away_team": {"team_id": away, "name": self.read_team_name(away)},
            "result_forecast": self.log.read_result_forecast(row["match_id"]),
            "finished": rounds.is_finished(row),
        }

    @staticmethod
    def list_result_chances(match: dict) -> list[str]:
        """Home, draw and away as shown, or nothing before Vesta's first forecast."""
        forecast = match["result_forecast"]
        if forecast is None:
            return []
        lines = [
            {"probability": forecast[side], "decided": side in forecast["decided_outcomes"]}
            for side in ("home", "draw", "away")
        ]
        return formats.format_chances(lines)

    # One match
    def read_match(self, match_id: int, now: datetime | None = None) -> MatchView:
        """The whole forecast of one match: every quantity of its newest publication before
        kick-off, in tabs; a player's quantities gathered under him; its status, with the result
        once finished."""
        row = self.database.read_first_row("select * from matches where match_id = ?", match_id)
        found = self._summarise(row) | {
            "status": rounds.describe_status(row, now or datetime.now(UTC))
        }
        items = self.log.read_match_forecasts(match_id)
        home, away = found["home_team"]["name"], found["away_team"]["name"]
        tabs: dict[str, list[Row]] = {tab: [] for tab in TABS}
        players: dict[str, dict[int, list[Row]]] = {tab: {} for tab in PLAYER_TABS}
        for group in self._group_by_quantity(items):
            tab = find_tab(group[0]["catalogue_id"])
            name = self._name_quantity(group[0])
            for item in group:
                lines = self._make_lines(item, home, away)
                if tab in PLAYER_TABS and "player" in item["subject_kind"]:
                    players[tab].setdefault(item["subject_id"], []).append(
                        Row(name, children=lines)
                    )
                    continue
                subject = self._name_subject(item)
                tabs[tab].append(Row(name if subject is None else f"{name}: {subject}",
                                     children=lines))  # fmt: skip
        for tab in PLAYER_TABS:
            tabs[tab] += self._make_player_rows(players[tab], found)
        newest = max(items, key=lambda item: item["publication_id"], default=None)
        return MatchView(
            summary=found,
            published_at=newest["published_at"] if newest else None,
            model_version=newest["model_version"] if newest else None,
            caveats=sorted({caveat for item in items for caveat in item["caveats"]}),
            tabs={tab: rows for tab, rows in tabs.items() if rows or tab != "Other"},
        )

    @staticmethod
    def _group_by_quantity(items: list[dict]) -> list[list[dict]]:
        """The forecasts grouped by quantity and window, in the catalogue's order."""
        groups: dict[str, list[dict]] = {}
        for item in items:
            groups.setdefault(f"{item['quantity_key']} {item['window']}", []).append(item)
        return sorted(groups.values(), key=lambda group: rank_by_catalogue(group[0]))

    def _name_quantity(self, item: dict) -> str:
        """The quantity's name as Vesta registered it, with its window unless the name says it
        already ("its next 3 matches", "next ten matches")."""
        quantity = self.log.read_quantity(item["quantity_key"])
        name = quantity["name"] if quantity is not None else item["quantity_key"]
        name = name[:1].upper() + name[1:]
        said = formats.describe_window(item["window"])
        named = said.startswith("next ") and " next " in f" {name} "
        if said and said not in name and not named:
            name += f" ({said})"
        return name

    def _name_subject(self, item: dict) -> str | None:
        kind, subject_id = item["subject_kind"], item["subject_id"]
        if "player" in kind:
            return self.read_player_name(subject_id)
        if "team" in kind:
            return self.read_team_name(subject_id)
        return None

    @staticmethod
    def _make_lines(item: dict, home: str, away: str) -> list[Row]:
        """A forecast's lines in words, the outcomes that exclude each other rounded together
        so that they add up to 100."""
        lines = item["lines"]
        chances = [formats.format_chance(line["probability"], line["decided"]) for line in lines]
        together = formats.find_exclusive_lines([line["line_key"] for line in lines])
        for index, chance in zip(together, formats.format_chances([lines[i] for i in together]),
                                 strict=True):  # fmt: skip
            chances[index] = chance
        return [
            Row("", formats.describe_line(line["line_key"], home, away), chance)
            for line, chance in zip(lines, chances, strict=True)
        ]

    def _find_team_of_player(self, player_id: int, on: str) -> int | None:
        if not self.database.read_columns("squad_memberships"):
            return None
        row = self.database.read_first_row(
            "select team_id from squad_memberships where player_id = ? and from_date <= ?"
            " and (to_date is null or to_date >= ?) order by from_date desc limit 1",
            player_id,
            on,
            on,
        )
        return row["team_id"] if row else None

    def _make_player_rows(self, players: dict[int, list[Row]], match: dict) -> list[Row]:
        """One row per player, named with the team when the dataset says it, home side first."""
        on = match["kickoff_utc"][:10]
        sides = {match["home_team"]["team_id"]: 0, match["away_team"]["team_id"]: 1}
        teams = {match[side]["team_id"]: match[side]["name"] for side in ("home_team",
                                                                          "away_team")}  # fmt: skip
        rows = []
        for player_id, quantities in players.items():
            name = self.read_player_name(player_id)
            team_id = self._find_team_of_player(player_id, on)
            team = teams.get(team_id, "")
            label = f"{name} ({team})" if team else name
            rows.append((sides.get(team_id, 2), name, Row(label, children=quantities)))
        return [row for _, _, row in sorted(rows, key=lambda entry: entry[:2])]

    # Season tables
    def read_season(self, competition_id: int, now: datetime | None = None) -> SeasonView:
        """The table of the season the Matches screen shows (`find_season`) as the dataset has
        it, beside Vesta's chances of its final places: the champion, then the cuts Vesta
        declared, none assumed. A season played in groups names each team's group."""
        found = self.find_season(competition_id, now or datetime.now(UTC))
        if found is None:
            return SeasonView(None, None, [], [], [])
        season_id = found["season_id"]
        items = self.log.read_season_forecasts(season_id)
        positions = {item["subject_id"]: item for item in items
                     if item["quantity_key"] == FINAL_POSITION}  # fmt: skip
        declared = {line["line_key"] for item in positions.values() for line in item["lines"]
                    if name_cut(line["line_key"])}  # fmt: skip
        cuts = sorted(declared | ({"place=1"} if positions else set()), key=rank_cut)
        standings = self._read_standings(season_id)
        groups = [self._name_group(line) for line in standings]
        teams_in_table = len({line["team_id"] for line in standings})
        teams_playing = self._count_season_teams(season_id)
        partial = teams_in_table < WHOLE_TABLE_SHARE * teams_playing
        grouped = len(set(groups)) > 1 or (partial and any(groups))
        columns = [*(["Group"] if grouped else []), "Place", "Team", "Played", "Points",
                   *(name_cut(key) for key in cuts)]  # fmt: skip
        rows = [
            [*([group] if grouped else []), str(line["position"]),
             self.read_team_name(line["team_id"]), str(line["played"]), str(line["points"]),
             *(self._describe_season_line(positions.get(line["team_id"]), key) for key in cuts)]
            for line, group in zip(standings, groups, strict=True)
        ]  # fmt: skip
        # The other season forecasts, one heading per quantity and window, one row per team or
        # player beneath it.
        others: list[Row] = []
        for group in self._group_by_quantity(items):
            if group[0]["quantity_key"] == FINAL_POSITION and rows:
                continue
            heading = Row(self._name_quantity(group[0]))
            for item in group:
                lines = self._make_lines(item, "home", "away")
                subject = self._name_subject(item)
                if subject is None:
                    heading.children += lines
                else:
                    heading.children.append(Row(subject, children=lines))
            if all(child.children for child in heading.children):
                heading.children.sort(key=lambda child: child.text)
            others.append(heading)
        newest = max(items, key=lambda item: item["publication_id"], default=None)
        named = "name" in self.database.read_columns("seasons") and found["name"]
        return SeasonView(found["name"] if named else str(found["season"]),
                          newest["published_at"] if newest else None,
                          columns if rows else [], rows, others, teams_in_table,
                          teams_playing)  # fmt: skip

    def find_season(self, competition_id: int, now: datetime) -> sqlite3.Row | None:
        """The season the Matches screen shows (`read_rounds`): that of the match nearest to now.
        A competition without a match: its newest season with a table, else its newest."""
        nearest = self.database.read_first_row(
            "select s.* from matches m join seasons s using (season_id)"
            " where s.competition_id = ?"
            " order by abs(julianday(m.kickoff_utc) - julianday(?)), m.match_id limit 1",
            competition_id,
            format_utc(now),
        )
        if nearest is not None:
            return nearest
        if "after_matchday" in self.database.read_columns("standings"):
            tabled = self.database.read_first_row(
                "select * from seasons where competition_id = ? and season_id in"
                " (select season_id from standings) order by season desc limit 1",
                competition_id,
            )
            if tabled is not None:
                return tabled
        return self.database.read_first_row(
            "select * from seasons where competition_id = ? order by season desc limit 1",
            competition_id,
        )

    def _read_standings(self, season_id: int) -> list[sqlite3.Row]:
        """The season's newest standings, when the dataset carries them (ours does): each
        group's (and league's) after its own last matchday, group by group."""
        columns = self.database.read_columns("standings")
        if "after_matchday" not in columns:
            return []
        same = "".join(f" and coalesce(o.{name}, '') = coalesce(s.{name}, '')"
                       for name in GROUPING if name in columns)  # fmt: skip
        order = "".join(f"coalesce(s.{name}, ''), " for name in GROUPING if name in columns)
        return self.database.read_rows(
            "select * from standings s where s.season_id = ? and s.after_matchday = (select"
            f" max(o.after_matchday) from standings o where o.season_id = s.season_id{same})"
            f" order by {order}s.position",
            season_id,
        )

    @staticmethod
    def _name_group(line: sqlite3.Row) -> str:
        """A team's part of the season in words ("League A, Group A1", "Group C"); empty when
        the dataset does not say."""
        keys = line.keys()
        return ", ".join(line[name] for name in GROUPING if name in keys and line[name])

    def _count_season_teams(self, season_id: int) -> int:
        row = self.database.read_first_row(
            "select count(*) as teams from (select home_team_id from matches where season_id = ?"
            " union select away_team_id from matches where season_id = ?)",
            season_id,
            season_id,
        )
        return row["teams"] if row else 0

    @staticmethod
    def _describe_season_line(forecast: dict | None, line_key: str) -> str:
        """A team's chance of a cut: the line as Vesta declared it, or else (the champion) read
        from the published distribution by Vesta's reader."""
        if forecast is None:
            return "–"
        for line in forecast["lines"]:
            if line["line_key"] == line_key:
                return formats.format_chance(line["probability"], line["decided"])
        try:
            probability = read_line(forecast["distribution_kind"], forecast["distribution"],
                                    line_key)  # fmt: skip
        except LineError:
            # A cut the published table cannot reach (a group smaller than the line).
            return "–"
        return formats.format_chance(probability)
