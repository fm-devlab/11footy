from desk import formats


def test_a_chance_is_held_between_below_one_and_above_ninety_nine():
    assert formats.format_chance(0.004) == "<1%"
    assert formats.format_chance(0.996) == ">99%"
    assert formats.format_chance(0.5) == "50%"
    assert formats.format_chance(0.0) == "<1%"
    assert formats.format_chance(1.0) == ">99%"


def test_only_a_decided_line_is_zero_or_a_hundred():
    assert formats.format_chance(1.0, decided=True) == "100%"
    assert formats.format_chance(0.0, decided=True) == "0%"


def test_outcomes_that_exclude_each_other_add_up_to_a_hundred():
    lines = [{"probability": p, "decided": False} for p in (0.335, 0.335, 0.33)]
    shown = formats.format_chances(lines)
    assert sum(int(text.rstrip("%")) for text in shown) == 100


def test_the_set_is_the_outcomes_or_yes_and_no():
    assert formats.find_exclusive_lines(["outcome=home", "outcome=draw", "any=home,draw"]) == [0, 1]
    assert formats.find_exclusive_lines(["yes", "no"]) == [0, 1]
    assert formats.find_exclusive_lines(["over=2.5", "under=2.5"]) == []


def test_lines_in_words():
    label = formats.describe_line
    assert label("outcome=home", "Arsenal", "Brentford") == "Arsenal"
    assert label("outcome=draw", "Arsenal", "Brentford") == "Draw"
    assert label("any=draw,away", "Arsenal", "Brentford") == "Draw or Brentford"
    assert label("over=2.5") == "3 or more"
    assert label("under=0.5") == "None"
    assert label("total_over=1.5") == "Total 2 or more"
    assert label("over=2.25") == "Over 2.25"
    assert label("at_least=2") == "2 or more"
    assert label("margin_at_least=home,2", "Arsenal") == "Arsenal by 2 or more"
    assert label("score=2-1") == "Score 2-1"
    assert label("bottom=3") == "Bottom 3"
    assert label("something_new=1") == "something_new=1"


def test_times_are_written_in_the_pcs_own_formats():
    assert "2026" in formats.format_local_date("2026-11-08T14:00:00Z")
    assert any(mark in formats.format_local_time("2026-11-08T14:00:00Z") for mark in ":.")


def test_a_window_in_words():
    empty = {"from_date": None, "to_date": None, "next_matches": None, "matchday": None}
    assert formats.describe_window(None) == ""
    assert formats.describe_window(empty | {"next_matches": 3}) == "next 3 matches"
    assert formats.describe_window(empty | {"matchday": 12}) == "matchday 12"
    window = empty | {"from_date": "2026-11-08", "to_date": "2026-12-07"}
    assert formats.describe_window(window) == "2026-11-08 to 2026-12-07"


def test_away_from_windows_a_date_reads_one_way_only():
    from datetime import datetime

    assert formats.format_english_date(datetime(2026, 11, 12)) == "Thu 12 Nov 2026"
