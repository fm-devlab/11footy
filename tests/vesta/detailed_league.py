"""A league with every block the families read, for the tests of `vesta.run`: seeded seasons of
matches with their half-time scores, goal events, shots with their xg, the teams' and the
players' match statistics and the lineups, so the champion, the team counts, goals in time, the
lineups and the players all run on it.

`write_detailed_league` builds one competition `code` of `clubs` clubs, a double round robin a
season: 2020 to 2024 played, 2025 open with its first `rounds_played` rounds played. Every club
keeps one squad (`SQUAD` positions); its eleven starts with a seeded rotation of one player.
"""

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from vesta.data_format import VERSION, create_statements

SEASONS = ((1, 2020), (2, 2021), (3, 2022), (4, 2023), (5, 2024), (6, 2025))  # the lineups'
# priors read the five seasons before the one forecast
FORMATION = "4-4-2"
# the squad of every club: position code, group; the first eleven the usual starters
SQUAD = (("GK", "goalkeeper"), ("RB", "defender"), ("CB", "defender"), ("CB", "defender"),
         ("LB", "defender"), ("RM", "midfielder"), ("CM", "midfielder"), ("CM", "midfielder"),
         ("LM", "midfielder"), ("ST", "forward"), ("ST", "forward"), ("GK", "goalkeeper"),
         ("CB", "defender"), ("CM", "midfielder"), ("ST", "forward"), ("RM", "midfielder"))
BENCH = 3  # the substitutes who come on in every match


def _make_schedule(clubs: int) -> list[list[tuple[int, int]]]:
    """A double round robin by the circle method: every pairing once a half."""
    teams = list(range(1, clubs + 1))
    halves = []
    for _ in range(clubs - 1):
        halves.append([(teams[index], teams[-1 - index]) for index in range(clubs // 2)])
        teams = [teams[0], teams[-1], *teams[1:-1]]
    return halves + [[(away, home) for home, away in matches] for matches in halves]


def _write_side(connection, generator, match_id: int, team: int, ids: dict) -> list[int]:
    """One side's lineup, shots and statistics; returns the minutes of its goals."""
    squad = [team * 100 + index for index in range(len(SQUAD))]
    eleven = list(range(11))
    rested = generator.randrange(1, 11)  # one usual starter rests, his reserve starts
    reserve = next(index for index in range(11, len(SQUAD))
                   if SQUAD[index][1] == SQUAD[rested][1])  # fmt: skip
    eleven[rested] = reserve
    bench = [index for index in range(len(SQUAD)) if index not in eleven][:BENCH + 1]
    played = {}
    for slot, index in enumerate(eleven, start=1):
        minutes = 90 if slot <= 8 else 70
        played[index] = minutes
        connection.execute(
            "INSERT INTO match_players VALUES (?, ?, ?, 1, ?, ?, ?, ?)",
            (match_id, team, squad[index], minutes, SQUAD[slot - 1][1], SQUAD[slot - 1][0],
             slot))  # fmt: skip
    for index in bench:
        minutes = 20 if index != bench[-1] else 0
        played[index] = minutes
        connection.execute("INSERT INTO match_players VALUES (?, ?, ?, 0, ?, NULL, NULL, NULL)",
                           (match_id, team, squad[index], minutes))  # fmt: skip
    connection.execute("INSERT INTO match_team_sheets VALUES (?, ?, ?)",
                       (match_id, team, FORMATION))  # fmt: skip
    goals, on_target, scorers = [], 0, {}
    shooters = [index for index in played if SQUAD[index][1] != "goalkeeper" and played[index]]
    for _ in range(generator.randrange(6, 18)):
        ids["shot"] += 1
        shooter = generator.choice(shooters)
        xg = round(generator.uniform(0.02, 0.35), 3)
        minute = generator.randrange(1, 91)
        result = ("goal" if generator.random() < xg
                  else generator.choice(("saved", "blocked", "off_target")))  # fmt: skip
        on_target += result in ("goal", "saved")
        assist = squad[generator.choice(shooters)] if result == "goal" else None
        connection.execute(
            "INSERT INTO shots VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 95.0, 34.0, ?, 'open_play',"
            " 'right_foot', 0, 'penalty_area', 'tracked', ?)",
            (ids["shot"], match_id, team, squad[shooter], assist, 1 if minute <= 45 else 2,
             minute, result, xg))  # fmt: skip
        scorers[shooter] = scorers.get(shooter, 0) + 1
        if result == "goal":
            ids["event"] += 1
            connection.execute(
                "INSERT INTO match_events VALUES (?, ?, ?, ?, ?, NULL, 'goal', ?, NULL, ?, ?, ?,"
                " 'play_by_play')",
                (ids["event"], match_id, ids["event"], 1 if minute <= 45 else 2, minute, team,
                 squad[shooter], assist, ids["shot"]))  # fmt: skip
            goals.append(minute)
    strength = 0.6 + 0.1 * (team % 9)  # the counts follow the club, as the models assume

    def count(mean: float) -> int:
        return max(0, round(generator.gauss(mean * strength, 1.5)))

    connection.execute(
        "INSERT INTO match_team_statistics VALUES (?, ?, 'report', ?, ?, ?, ?, ?, ?, 0)",
        (match_id, team, sum(scorers.values()), on_target, count(5.0), count(11.0), count(2.0),
         count(2.0)))  # fmt: skip
    for index, minutes in played.items():
        connection.execute(
            "INSERT INTO player_match_statistics VALUES (?, ?, ?, 0, ?, 0, ?, ?, ?, 0, ?, ?)",
            (match_id, team, squad[index], scorers.get(index, 0) if minutes else 0,
             generator.randrange(0, 3) if minutes else 0, generator.randrange(0, 2),
             generator.randrange(0, 2), int(generator.random() < 0.1) if minutes else 0,
             generator.randrange(0, 5) if SQUAD[index][1] == "goalkeeper" and minutes else None))
    return goals


def write_detailed_league(path: Path, code: str = "PL", clubs: int = 14,
                          rounds_played: int = 4, dataset: str = "someone") -> Path:  # fmt: skip
    """The league as a dataset in the format, every block present; the players named."""
    connection = sqlite3.connect(path)
    for statement in create_statements("B1", "B2", "B3", "B4", "B5", "B6", "B7"):
        connection.execute(statement)
    connection.execute("ALTER TABLE matches ADD COLUMN matchday INTEGER")
    connection.execute("CREATE TABLE players (player_id INTEGER PRIMARY KEY, name TEXT)")
    connection.execute("INSERT INTO format_version VALUES (?, ?, '2025-10-01T00:00:00Z')",
                       (VERSION, dataset))  # fmt: skip
    connection.execute("INSERT INTO competitions VALUES (1, ?, 'league', 'Europe/London')", (code,))
    for season_id, season in SEASONS:
        connection.execute("INSERT INTO seasons VALUES (?, 1, ?, ?)",
                           (season_id, season, "open" if season == 2025 else "closed"))
    connection.executemany("INSERT INTO teams VALUES (?, 'club', ?)",
                           [(team, f"club {team}") for team in range(1, clubs + 1)])  # fmt: skip
    connection.executemany("INSERT INTO players VALUES (?, ?)",
                           [(team * 100 + index, f"player {team * 100 + index}")
                            for team in range(1, clubs + 1)
                            for index in range(len(SQUAD))])  # fmt: skip
    generator = random.Random(5)
    ids = {"match": 0, "shot": 0, "event": 0}
    for season_id, season in SEASONS:
        for number, matches in enumerate(_make_schedule(clubs)):
            kickoff = f"{date(season, 8, 2) + timedelta(days=7 * number):%Y-%m-%d}T15:00:00Z"
            for home, away in matches:
                ids["match"] += 1
                match_id = ids["match"]
                connection.execute(
                    "INSERT INTO matches (match_id, season_id, kickoff_utc, kickoff_precision,"
                    " local_date, home_team_id, away_team_id, neutral_venue, status,"
                    " extra_time_played, matchday) VALUES (?, ?, ?, 'minute', ?, ?, ?, 0,"
                    " 'scheduled', 0, ?)",
                    (match_id, season_id, kickoff, kickoff[:10], home, away, number + 1))
                if season == 2025 and number >= rounds_played:
                    continue
                home_goals = _write_side(connection, generator, match_id, home, ids)
                away_goals = _write_side(connection, generator, match_id, away, ids)
                connection.execute(
                    "UPDATE matches SET status = 'played', home_goals = ?, away_goals = ?,"
                    " home_goals_on_pitch = ?, away_goals_on_pitch = ?, home_goals_half_time = ?,"
                    " away_goals_half_time = ? WHERE match_id = ?",
                    (len(home_goals), len(away_goals), len(home_goals), len(away_goals),
                     sum(minute <= 45 for minute in home_goals),
                     sum(minute <= 45 for minute in away_goals), match_id))  # fmt: skip
    connection.commit()
    connection.close()
    return path
