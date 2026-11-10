from datetime import UTC, datetime

from desk.rounds import (
    DATE_GAP,
    WINDOW_GAP,
    describe_status,
    find_current_index,
    group_rounds,
    is_finished,
    label_round,
)


class Row(dict):
    """A match row as sqlite3.Row answers it: by column name, with its keys."""


def match(match_id, kickoff, matchday=None, stage=None, leg=None, status="scheduled", goals=None):
    home, away = goals if goals else (None, None)
    return Row(match_id=match_id, season_id=1, kickoff_utc=kickoff, matchday=matchday,
               stage=stage, leg=leg, status=status, home_goals=home, away_goals=away)  # fmt: skip


def at(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def label(text: str) -> str:
    return text[:10]


LEAGUE = [
    match(1, "2026-11-07T14:00:00Z", 10, status="played", goals=(2, 1)),
    match(2, "2026-11-08T15:00:00Z", 10),
    match(3, "2026-11-09T20:00:00Z", 10),
    # matchday 11, one match brought forward weeks before the others
    match(4, "2026-10-20T19:00:00Z", 11, status="played", goals=(0, 0)),
    match(5, "2026-11-21T15:00:00Z", 11),
    match(6, "2026-11-22T15:00:00Z", 11),
]


def current(rows, now: str) -> str:
    rounds = group_rounds(rows, DATE_GAP, label)
    return rounds[find_current_index(rounds, at(now))].label


def finished(row, goals=(1, 0)):
    return match(row["match_id"], row["kickoff_utc"], row["matchday"], status="played",
                 goals=goals)  # fmt: skip


def test_mid_matchday_the_round_in_progress_is_current():
    for now in ("2026-11-07T13:00:00", "2026-11-08T16:00:00", "2026-11-20T23:00:00"):
        assert current(LEAGUE, now) == "Matchday 10", now
    rounds = group_rounds(LEAGUE, DATE_GAP, label)
    assert [row["match_id"] for row in rounds[0].matches] == [1, 2, 3], "the round stays whole"


def test_once_every_match_is_finished_the_next_round_is_current():
    over = [finished(LEAGUE[0]), finished(LEAGUE[1]), finished(LEAGUE[2]), *LEAGUE[3:]]
    assert current(over, "2026-11-10T09:00:00") == "Matchday 11", "at once, before its kick-off"
    assert current(over[:3], "2026-11-10T09:00:00") == "Matchday 10", "the season's last stays"


def test_a_postponed_match_stays_in_its_round_and_does_not_hold_it():
    postponed = match(3, "2026-11-09T20:00:00Z", 10, status="postponed")
    rows = [finished(LEAGUE[0]), finished(LEAGUE[1]), postponed, *LEAGUE[3:]]
    assert current(rows, "2026-11-10T09:00:00") == "Matchday 11"
    rounds = group_rounds(rows, DATE_GAP, label)
    assert [row["match_id"] for row in rounds[0].matches] == [1, 2, 3]
    assert describe_status(postponed, at("2026-11-10T09:00:00")) == "Postponed"


def test_a_match_moved_weeks_later_does_not_hold_its_round():
    moved = match(3, "2026-12-20T20:00:00Z", 10)
    rows = [finished(LEAGUE[0]), finished(LEAGUE[1]), moved, *LEAGUE[3:]]
    assert current(rows, "2026-11-10T09:00:00") == "Matchday 11"
    assert 3 in [row["match_id"] for row in group_rounds(rows, DATE_GAP, label)[0].matches]


def test_a_match_moved_early_does_not_start_its_round():
    assert current(LEAGUE, "2026-10-21T12:00:00") == "Matchday 10", "before the season, the first"


def test_once_the_clock_passes_a_later_kickoff_that_round_is_current():
    # a dataset built before matchday 10 was played: its results are not in it yet
    assert current(LEAGUE, "2026-11-21T15:00:00") == "Matchday 11"


def test_cups_go_by_stage_and_leg():
    rows = [
        match(1, "2026-11-04T20:00:00Z", 4, "league phase"),
        match(2, "2027-02-17T20:00:00Z", None, "round of 16", 1),
        match(3, "2027-03-10T20:00:00Z", None, "round of 16", 2),
    ]
    labels = [found.label for found in group_rounds(rows, DATE_GAP, label)]
    assert labels == ["League phase, matchday 4", "Round of 16, first leg",
                      "Round of 16, second leg"]  # fmt: skip
    assert label_round(rows[0]) == "League phase, matchday 4"


def test_without_matchdays_the_dates_make_the_rounds():
    weekend = [match(1, "2026-11-07T14:00:00Z"), match(2, "2026-11-08T18:00:00Z"),
               match(3, "2026-11-14T14:00:00Z")]  # fmt: skip
    assert [r.label for r in group_rounds(weekend, DATE_GAP, label)] == [
        "2026-11-07 to 2026-11-08", "2026-11-14"]  # fmt: skip
    window = [match(1, "2026-11-12T18:00:00Z"), match(2, "2026-11-15T18:00:00Z"),
              match(3, "2026-11-18T18:00:00Z"), match(4, "2027-03-26T18:00:00Z")]  # fmt: skip
    assert len(group_rounds(window, WINDOW_GAP, label)) == 2


def test_each_match_says_its_status():
    now = at("2026-11-08T16:00:00")
    assert describe_status(LEAGUE[0], now) == "Finished 2-1"
    assert describe_status(LEAGUE[1], now) == "In play"
    assert describe_status(LEAGUE[2], now) == "To play"
    assert describe_status(LEAGUE[1], at("2026-11-08T20:00:00")) == "Played, no result yet"
    assert describe_status(match(9, "2026-11-08T15:00:00Z", status="postponed"), now) == "Postponed"


def test_a_cup_without_matchdays_moves_by_stage_and_leg_the_same_way():
    first = match(1, "2027-02-17T20:00:00Z", None, "round of 16", 1)
    second = match(2, "2027-03-10T20:00:00Z", None, "round of 16", 2)
    assert current([first, second], "2027-02-17T21:00:00") == "Round of 16, first leg"
    played = match(1, first["kickoff_utc"], None, "round of 16", 1, "played", (1, 1))
    assert current([played, second], "2027-02-20T09:00:00") == "Round of 16, second leg"


def test_a_postponed_match_stored_at_nil_nil_is_not_finished():
    postponed = match(1, "2026-11-08T15:00:00Z", 13, status="postponed", goals=(0, 0))
    assert not is_finished(postponed)
    assert describe_status(postponed, at("2026-11-10T09:00:00")) == "Postponed"
    assert is_finished(match(2, "2026-11-08T15:00:00Z", 13, status="awarded", goals=(3, 0)))
    assert not is_finished(match(3, "2026-11-08T15:00:00Z", 13, status="abandoned", goals=(1, 0)))


def test_a_match_called_off_and_moved_later_is_to_play():
    now = at("2026-11-10T09:00:00")
    for status in ("suspended", "postponed", "cancelled"):
        moved = match(1, "2026-12-09T20:00:00Z", 8, status=status)
        assert describe_status(moved, now) == "To play", status
    assert describe_status(match(2, "2026-11-01T20:00:00Z", 8, status="suspended"),
                           now) == "Suspended"  # fmt: skip


def test_a_shootout_and_extra_time_are_said():
    final = match(1, "2026-05-30T16:00:00Z", stage="final", status="played", goals=(1, 1))
    final.update(extra_time_played=1, shootout_home_goals=4, shootout_away_goals=3)
    now = at("2026-11-10T09:00:00")
    assert describe_status(final, now) == "Finished 1-1 (4-3 pens)"
    final.update(shootout_home_goals=None, shootout_away_goals=None, home_goals=2)
    assert describe_status(final, now) == "Finished 2-1 a.e.t."
    final.update(extra_time_played=0)
    assert describe_status(final, now) == "Finished 2-1"
    assert describe_status(LEAGUE[0], now) == "Finished 2-1", "a dataset without the columns"


def nations(match_id, kickoff, home, away, stage="group stage"):
    row = match(match_id, kickoff, stage=stage)
    row.update(home_team_id=home, away_team_id=away)
    return row


def test_a_stage_without_matchdays_is_split_into_its_rounds():
    window = [
        # one international window, each team playing twice: two rounds
        nations(1, "2026-09-24T18:00:00Z", 1, 2), nations(2, "2026-09-25T18:00:00Z", 3, 4),
        nations(3, "2026-09-27T18:00:00Z", 1, 3), nations(4, "2026-09-28T18:00:00Z", 2, 4),
        # the next window, a month later
        nations(5, "2026-10-24T18:00:00Z", 1, 4), nations(6, "2026-10-24T18:00:00Z", 2, 3),
    ]  # fmt: skip
    found = group_rounds(window, WINDOW_GAP, label)
    assert [r.label for r in found] == [
        "Group stage, 2026-09-24 to 2026-09-25", "Group stage, 2026-09-27 to 2026-09-28",
        "Group stage, 2026-10-24"]  # fmt: skip
    final = [nations(7, "2026-11-01T18:00:00Z", 1, 2, "final")]
    assert [r.label for r in group_rounds(final, WINDOW_GAP, label)] == ["Final"]


def test_a_league_match_without_matchday_joins_its_matchday():
    rows = [*LEAGUE, match(7, "2026-11-08T20:00:00Z"), match(8, "2026-11-23T20:00:00Z"),
            match(9, "2027-01-10T20:00:00Z")]  # fmt: skip
    found = group_rounds(rows, DATE_GAP, label)
    assert [r.label for r in found] == ["Matchday 10", "Matchday 11", "2027-01-10"]
    assert [row["match_id"] for row in found[0].matches] == [1, 2, 7, 3]
    assert 8 in [row["match_id"] for row in found[1].matches], "the nearest within days"
