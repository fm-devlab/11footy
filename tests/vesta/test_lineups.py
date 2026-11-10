"""Lineups: the chances always describe an eleven, the logistic fit finds its weights, the rows read
only what was known before the match, and the minutes are binned as the plan says."""

import numpy as np

from vesta.lineups.model import (
    TeamMatch,
    World,
    classify_position,
    collect_targets,
    fit_logistic,
    scale_to_eleven,
)
from vesta.lineups.verdict import _find_minute_bin


def test_the_chances_add_up_to_one_goalkeeper_and_ten_others():
    rng = np.random.default_rng(3)
    log_odds = rng.normal(0, 2, 30)
    goalkeepers = np.zeros(30, bool)
    goalkeepers[:3] = True
    chances = scale_to_eleven(log_odds, goalkeepers)
    assert abs(chances[goalkeepers].sum() - 1) < 1e-6
    assert abs(chances[~goalkeepers].sum() - 10) < 1e-6
    assert (np.argsort(chances[~goalkeepers]) == np.argsort(log_odds[~goalkeepers])).all()


def test_a_logistic_fit_finds_the_weight_it_was_drawn_from():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20_000, 1))
    y = (rng.random(20_000) < 1 / (1 + np.exp(-(0.5 + 2.0 * x[:, 0])))).astype(float)
    beta = fit_logistic(x, y, ridge=0.0)
    assert abs(beta[0] - 0.5) < 0.06 and abs(beta[1] - 2.0) < 0.1


def test_the_rows_read_only_the_matches_before_and_the_minutes_bins():
    def match(index, league=True):
        day = f"2020-0{1 + index // 28}-{1 + index % 28:02d}T15:00:00Z"
        return TeamMatch(index, day, "PL" if league else "CL", league, 2020, day,
                         day.replace("15:00", "17:30"), 1_577_000_000 + index * 86_400)

    played = [match(index) for index in range(12)]
    sheets, formations = {}, {}
    for item in played:
        # players 1-11 start every match; player 12 comes on; in match 11, 12 starts for 11
        sheet = {player: (True, 90, "G" if player == 1 else "CM") for player in range(1, 12)}
        sheet[12] = (False, 20, None)
        if item.match_id == 11:
            sheet[11], sheet[12] = (False, 0, None), (True, 90, "CM")
        sheets[(item.match_id, 7)] = sheet
        formations[(item.match_id, 7)] = "4-4-2"
    appearances = {player: [(item.kickoff, 7) for item in played] for player in range(1, 13)}
    rows, teams = collect_targets(
        World({7: played}, sheets, formations, appearances), {"PL"}, {2020}
    )
    last = [row for row in rows if row.match_id == 11]
    assert {row.player_id for row in last} == set(range(1, 13))
    twelve = next(row for row in last if row.player_id == 12)
    assert twelve.state == "came_on" and twelve.features[0] == 0.0  # the match's own sheet unread
    assert twelve.started and not next(row for row in last if row.player_id == 11).started
    assert next(row for row in last if row.player_id == 1).goalkeeper
    assert teams[-1].last_formation == "4-4-2"
    assert [_find_minute_bin(m) for m in (0, 1, 44, 45, 60, 89, 90, 97)] == [0, 1, 1, 2, 3, 3, 4, 4]
    assert classify_position("CD-L") == "defender" and classify_position("CF-R") == "forward"


def test_the_expected_eleven_puts_each_player_in_the_slot_he_most_likely_plays():
    from vesta.lineups.publication import choose_expected_eleven

    codes = ["G", "RB", "CD-R", "CD-L", "LB", "RM", "CM-R", "CM-L", "LM", "CF-R", "CF-L"]
    players = list(range(100, 113))  # 13 players, two left out
    starts = [0.95] * 11 + [0.2, 0.1]
    position_of = {player: {code: 0.9 if index == slot else 0.1 / 10
                            for slot, code in enumerate(codes)}
                   for index, player in enumerate(players)}  # fmt: skip
    slots = choose_expected_eleven(players, starts, position_of, codes)
    assert [slot.formation_slot for slot in slots] == list(range(1, 12))
    assert [slot.player_id for slot in slots] == players[:11]
    assert slots[0].position_code == "G" and abs(slots[0].probability_starts - 0.95) < 1e-12


def test_what_happened_to_a_lineups_key_from_the_sheet():
    from vesta.jobs.check import settle_lineup_outcome
    from vesta.quantities import load_quantities

    definitions = load_quantities()
    sheet = {"players": {7: (True, 97, "CD-L"), 8: (False, 12, None)},
             "formations": {1: "4-2-3-1", 2: "4-2-4-0"}}  # fmt: skip
    assert settle_lineup_outcome("L1.starts", 8, sheet, definitions) == (True, False)
    assert settle_lineup_outcome("L6.plays", 8, sheet, definitions) == (True, True)
    assert settle_lineup_outcome("L6.minutes", 7, sheet, definitions) == (True, 90)
    assert settle_lineup_outcome("L6.minutes", 9, sheet, definitions) == (
        True,
        0,
    )  # not on the sheet
    assert settle_lineup_outcome("L9.position", 7, sheet, definitions) == (True, "CD-L")
    assert settle_lineup_outcome("L9.position", 8, sheet, definitions) == (True, None)  # void
    assert settle_lineup_outcome("L2.formation.team", 2, sheet, definitions) == (True, "other")
    assert settle_lineup_outcome("L2.formation.team", 3, sheet, definitions) == (False, None)


def test_a_player_named_by_both_clubs_of_a_fixture_stays_in_one_pool():
    # Meslier before Arsenal v Leeds (match 58136, 10 October 2026): on Leeds' bench in 2025, on
    # Arsenal's in September 2026, never on the pitch, so no appearance told his team and both
    # pools held him; the morning wrote his rows twice (a UNIQUE failure in the forecast log)
    from vesta.lineups.model import FADE, HISTORY, build_player_rows
    from vesta.lineups.publication import LineupsOfTheDay, TeamLineup

    keeper, home, away = 999, 82, 708
    days = {home: ["2026-08-20", "2026-08-30", "2026-09-09"],
            away: ["2025-11-01", "2025-11-15", "2025-11-29"]}  # fmt: skip
    team_matches, sheets = {}, {}
    for team, dates in days.items():
        team_matches[team] = []
        for index, day in enumerate(dates):
            kickoff = f"{day}T15:00:00Z"
            item = TeamMatch(team * 10 + index, kickoff, "PL", True, 2026, kickoff, kickoff,
                             np.datetime64(kickoff[:19]).astype("datetime64[s]").astype(float))
            team_matches[team].append(item)
            sheet = {team * 100 + slot: (True, 90, "CD-L") for slot in range(11)}
            sheet[keeper] = (False, 0, None)  # named, unused
            sheets[(item.match_id, team)] = sheet
    world = World(team_matches, sheets, {}, {})

    class Stub(LineupsOfTheDay):  # the real rows and their guard; no priors to fit
        def build_team_lineup(self, match_id, competition, season, kickoff_utc, team_id,
                              exclude=frozenset()):
            before = self.world.team_matches[team_id]
            fixture = TeamMatch(match_id, kickoff_utc, competition, True, season, self.now,
                                self.now, np.datetime64(kickoff_utc[:19]).astype(
                                    "datetime64[s]").astype(float))  # fmt: skip
            recent = before[::-1][:HISTORY]
            rows = [
                row
                for row in build_player_rows(
                    self.world,
                    team_id,
                    fixture,
                    before,
                    recent,
                    [FADE**age for age in range(len(recent))],
                    {},
                )
                if row.player_id not in exclude
            ]
            return TeamLineup(team_id, [], "4-4-2", 1.0, [],
                              tuple((row.player_id, row.group, 0.5, 0.5, 45.0) for row in rows))

    lineups = Stub.__new__(Stub)
    lineups.world, lineups.now = world, "2026-10-10T06:00:00Z"
    alone = [
        lineups.build_team_lineup(58136, "PL", 2026, "2026-10-10T11:30:00Z", team) for team in days
    ]
    assert all(keeper in {item[0] for item in side.pool} for side in alone)  # the defect
    both = lineups.build_fixture_lineups(58136, "PL", 2026, "2026-10-10T11:30:00Z", home, away)
    pools = [{item[0] for item in side.pool} for side in both]
    assert keeper in pools[0] and keeper not in pools[1]  # Arsenal named him last
    assert not pools[0] & pools[1]
    assert len(pools[1]) == 11


def test_a_player_banned_by_the_rules_is_left_out_of_the_pool():
    # Thorstvedt before Sassuolo v Fiorentina (73613): a straight red on 27 October, a one-match
    # ban the world's suspensions_by_rule names; the morning of 2 November gave him a 19% start
    import sqlite3

    from vesta.lineups.model import FADE, HISTORY, build_player_rows, read_bans

    connection = sqlite3.connect(":memory:")
    assert read_bans(connection) == ({}, False)  # no table: no source
    connection.execute("CREATE TABLE suspensions_by_rule (player_id INTEGER, competition_id"
                       " INTEGER, served_in_match_ids TEXT)")  # fmt: skip
    connection.execute("CREATE TABLE competitions (competition_id INTEGER, code TEXT)")
    connection.executemany("INSERT INTO competitions VALUES (?, ?)", [(4, "SA"), (9, "L1")])
    assert read_bans(connection) == ({}, False)  # an empty table is no source either
    connection.executemany("INSERT INTO suspensions_by_rule VALUES (?, ?, ?)",
                           [(7, 4, "73613"), (8, 4, "73613,73620"), (9, 4, None),
                            (10, 9, "73700")])  # fmt: skip
    assert read_bans(connection, frozenset()) == ({}, False)  # no league applies bans
    assert read_bans(connection, frozenset({"L1"})) == ({73700: frozenset({10})}, True)
    bans, read = read_bans(connection, frozenset({"SA"}))
    assert read and bans == {73613: frozenset({7, 8}), 73620: frozenset({8})}

    team, dates = 601, ["2026-10-16", "2026-10-24", "2026-10-27"]
    matches, sheets = [], {}
    for index, day in enumerate(dates):
        kickoff = f"{day}T15:00:00Z"
        item = TeamMatch(index, kickoff, "SA", True, 2026, kickoff, kickoff,
                         np.datetime64(kickoff[:19]).astype("datetime64[s]").astype(float))
        matches.append(item)
        sheets[(index, team)] = {player: (True, 90, "CD-L") for player in range(1, 13)}
    fixture = TeamMatch(73613, "2026-11-02T17:30:00Z", "SA", True, 2026, "2026-11-02T06:00:00Z",
                        "2026-11-02T06:00:00Z",
                        np.datetime64("2026-11-02T17:30:00").astype("datetime64[s]").astype(float))
    recent = matches[::-1][:HISTORY]
    weights = [FADE ** age for age in range(len(recent))]
    world = World({team: matches}, sheets, {}, {}, bans, read)
    pool = {
        row.player_id
        for row in build_player_rows(world, team, fixture, matches, recent, weights, {})
    }
    assert pool == set(range(1, 13)) - {7, 8}
    unbanned = World({team: matches}, sheets, {}, {})
    assert {
        row.player_id
        for row in build_player_rows(unbanned, team, fixture, matches, recent, weights, {})
    } == set(range(1, 13))


def test_the_bans_apply_in_the_leagues_whose_check_passed():
    """The ban check stopped on 3 November 2026 (played-through bans, Ligue 1 worse); plan (a)
    passed for four leagues on 3 November 2026. Ligue 1
    applies none until its own plan passes."""
    from vesta.lineups.model import load_ban_competitions

    assert load_ban_competitions() == frozenset({"BL", "LL", "PL", "SA"})
