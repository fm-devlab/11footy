"""The small football world of the tests: its clubs, season and matches, by the ids the
window's and Vesta's tests share.

Four clubs in one Premier League season: a played match with every kind of fact, a goalless
match, a result awarded by a ruling, and a match to come. The world itself is built by the
maintainer's pipeline (`src/data`, not in the public repository); its exported dataset and
release are kept ready in `tests/fixtures/desk/release` (the whole file, the split files and
SHA256SUMS), rebuilt by `python tests/data/make_desk_fixtures.py`.
"""

ARSENAL, VILLA, BOURNEMOUTH, BRENTFORD = 82, 213, 652, 773
SEASON, COMPETITION = 1, 1
PLAYED, GOALLESS, AWARDED, TO_COME = 101, 102, 103, 104
BUILT_AT = "2026-10-20T04:00:00Z"
# The rest of Arsenal's expected eleven, players with no match in the small world.
ARSENAL_SQUAD = (
    (11, "Ben White", "defender"),
    (12, "William Saliba", "defender"),
    (13, "Gabriel Magalhães", "defender"),
    (14, "Riccardo Calafiori", "defender"),
    (15, "Declan Rice", "midfielder"),
    (16, "Martin Ødegaard", "midfielder"),
    (17, "Mikel Merino", "midfielder"),
    (18, "Viktor Gyökeres", "forward"),
    (19, "Gabriel Martinelli", "forward"),
)
