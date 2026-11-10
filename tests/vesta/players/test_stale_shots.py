from vesta.players.model import Appearance
from vesta.players.verdict import drop_stale_appearances


def appearance(known_at: str) -> Appearance:
    return Appearance(known_at, 2026, 90.0, 0.3, 0, 0.1, 0, 0, 1)


def test_the_check_leaves_out_the_sides_last_k_league_matches():
    """The outage of the stale-shots check: his appearances from his side's k-th last league
    match before the forecast onward are left out, whatever club he played them for."""
    side = ["2026-10-01T17:00Z", "2026-10-08T17:00Z", "2026-10-15T17:00Z", "2026-10-22T17:00Z"]
    known = [appearance(day) for day in ("2026-09-24T17:00Z", "2026-10-01T17:00Z",
                                         "2026-10-08T17:00Z", "2026-10-15T17:00Z",
                                         "2026-10-22T17:00Z")]  # fmt: skip
    when = "2026-10-29T12:00Z"
    assert drop_stale_appearances(known, side, when, 0) == known
    assert drop_stale_appearances(known, side, when, 1) == known[:4]
    assert drop_stale_appearances(known, side, when, 3) == known[:2]
    assert drop_stale_appearances(known, side, when, 5) == []  # fewer matches than k: nothing left
