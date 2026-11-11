"""A small forecast log for the tests (the window, Vesta), written by Vesta's own writer
(`vesta.forecast_log.writing`) over the matches of the small world.

The played match has its whole life: the first publication (later withdrawn), the lineup, the
forecasts of record at the real kick-off, two live minutes, the whistle, the outcomes, the
checked lines, and a void for the substitute who did not come on. The match to come has a
morning publication; the season two mornings. What is published where, the calibration, the
scores, the explanations and the expected eleven go through Vesta's writer too.
"""

import json
from pathlib import Path

from small_world import ARSENAL, BOURNEMOUTH, BRENTFORD, PLAYED, SEASON, TO_COME, VILLA

from vesta.forecast_log.store import open_log
from vesta.forecast_log.writing import (
    ExpectedSlot,
    Explanation,
    KeyScore,
    Publication,
    PublishedDistribution,
    Quantity,
    TrackRecordHits,
    TrackRecordScore,
    publish,
    record_calibration,
    record_expected_eleven,
    record_explanation,
    record_forecasts_of_record,
    record_hits,
    record_key_scores,
    record_line_check,
    record_outcome,
    record_scores,
    record_void,
    record_withdrawal,
    register_model_version,
    register_published_quantity,
    register_quantity,
)
from vesta.forecast_reading import read_line
from vesta.quantities import load_quantities
from vesta.research.calibration import Bucket

VERSION = "vesta-2.0.0"
# The scores of each key (kind, key, metric, value, the reference's value on the same forecasts).
KEY_SCORES = (
    ("results", "M1.result", "ranked_probability_score", 0.198, 0.224),
    ("results", "M1.result", "log_loss", 0.973, 1.071),
    ("goals", "M3.goals", "ranked_probability_score", 0.138, 0.150),
)
# The results kind's summary: its one key's skill on its deciding metric (the first of §9).
RESULTS_SKILL = 1 - 0.198 / 0.224
VESTA_ONE = "vesta-1"
VESTA_ONE_UNMATCHED = 40
# Vesta v1's results: M1.result's deciding metric over its 300 counted forecasts (own, reference).
VESTA_ONE_RESULTS = (0.205, 0.212, 300)
# Every version together (model_version 'all', schema 9), as the check job writes it: the key's
# losses pooled over both versions' forecasts, then its skill; never the versions' skills averaged.
ALL_RESULTS_SKILL = 1 - (0.198 * 1 + 0.205 * 300) / (0.224 * 1 + 0.212 * 300)
COMMIT = "abcdef0"
PLAYED_KICKOFF = "2026-08-22T14:00:00Z"
COMPUTED_AT = "2026-09-01T05:00:00Z"
# The live series quantities with the lines Vesta declares (config/vesta/quantities.csv), so
# that every line of live_headlines is written and read (schema 3: M1 6, M4 2, M3 12 of 25).
DECLARED = {
    key: tuple(load_quantities()[key].quantity.lines)
    for key in ("M1.result", "M3.goals", "M4.both_score")
}
QUANTITIES = (
    Quantity(
        "M1.result",
        "M1",
        "results",
        "match",
        "the result",
        "full_time",
        "ordered",
        False,
        "matches.home_goals_on_pitch",
        DECLARED["M1.result"],
    ),
    Quantity(
        "M3.goals",
        "M3",
        "goals",
        "match",
        "goals of the match",
        "full_time",
        "count",
        False,
        "matches.home_goals + away_goals",
        DECLARED["M3.goals"],
    ),
    Quantity(
        "M4.both_score",
        "M4",
        "goals",
        "match",
        "both teams score",
        "full_time",
        "binary",
        False,
        "matches.home_goals and away_goals",
        DECLARED["M4.both_score"],
    ),
    Quantity(
        "P3.goals",
        "P3",
        "scorers",
        "player",
        "goals of a player",
        "full_time",
        "count",
        True,
        "player_match_statistics.goals",
        ("at_least=1",),
    ),
    Quantity(
        "L6.plays",
        "L6",
        "lineups",
        "player",
        "plays",
        "full_time",
        "binary",
        False,
        "match_players.minutes_played > 0",
        ("yes",),
    ),
    Quantity(
        "T1.final_position",
        "T1",
        "seasons",
        "season_team",
        "final position",
        "season",
        "rank",
        False,
        "standings.position",
        ("place=1", "bottom=1"),
    ),
)
PLAYERS = (2, 3, 5)
# The live.db events between the minutes 30 and 70 of the played match.


def result(home: float, draw: float) -> list:
    return [["home", home], ["draw", draw], ["away", 1.0 - home - draw]]


def goals(already: int = 0) -> dict:
    """Goals of the match: the goals already scored and a fixed spread of the rest."""
    body = [0.10, 0.22, 0.27, 0.21, 0.12, 0.08]
    probabilities = [0.0] * already + body[: len(body) - already]
    probabilities[-1] += 1.0 - sum(probabilities)
    return {"probabilities": probabilities, "tail_from": len(body), "tail": 0.0}


def player_goals(chance: float) -> dict:
    return {
        "probabilities": [1.0 - chance, chance * 0.8, chance * 0.2],
        "tail_from": 3,
        "tail": 0.0,
    }


def match_distributions(
    home: float, draw: float, already: int = 0, goals_decided: tuple[str, ...] = ()
) -> list:
    found = [
        PublishedDistribution("M1.result", PLAYED, VERSION, result(home, draw)),
        PublishedDistribution(
            "M3.goals", PLAYED, VERSION, goals(already), decided_lines=goals_decided
        ),
        PublishedDistribution("M4.both_score", PLAYED, VERSION, {"yes": 0.52 + 0.01 * already}),
    ]
    for player_id, chance in zip(PLAYERS, (0.45, 0.20, 0.38), strict=True):
        found.append(PublishedDistribution("P3.goals", player_id, VERSION, player_goals(chance)))
        found.append(
            PublishedDistribution(
                "L6.plays", player_id, VERSION, {"yes": 0.95 if player_id != 3 else 0.40}
            )
        )
    return found


def full_time_distributions() -> list:
    """The whistle at 2-1: the result and the goals are final, so Vesta marks their lines
    decided, at exactly 1 or 0 (contract §6.3b); the players' quantities are not marked."""
    final = {
        "M1.result": PublishedDistribution(
            "M1.result",
            PLAYED,
            VERSION,
            result(1.0, 0.0),
            decided_lines=("outcome=home", "outcome=draw", "outcome=away"),
        ),
        "M3.goals": PublishedDistribution(
            "M3.goals",
            PLAYED,
            VERSION,
            {"probabilities": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], "tail_from": 6, "tail": 0.0},
            decided_lines=("over=1.5", "over=2.5"),
        ),
    }
    return [
        final.get(distribution.quantity_key, distribution)
        for distribution in match_distributions(0.999, 0.0005, 3)
    ]


def settled(kind: str, outcome: object, like: object) -> object:
    """The distribution that puts all its weight on what happened: a line read from it says
    whether the line happened."""
    if kind == "ordered":
        return [[name, 1.0 if name == outcome else 0.0] for name, _ in like]
    if kind == "binary":
        return {"yes": 1.0 if outcome else 0.0}
    size = like["tail_from"]
    return {
        "probabilities": [1.0 if count == outcome else 0.0 for count in range(size)],
        "tail_from": size,
        "tail": 0.0,
    }


def build(path: Path) -> Path:
    connection = open_log(path)
    register_model_version(
        connection,
        VERSION,
        "spine",
        "2026-08-01T00:00:00Z",
        "The match engine moved to the new world.",
        freeze_tag=VERSION,
        verdict_id="V-0001",
    )
    for quantity in QUANTITIES:
        register_quantity(connection, quantity)
    first = publish(
        connection,
        Publication(
            "pre_match",
            "first",
            "2026-08-17T18:00:00Z",
            COMMIT,
            match_id=PLAYED,
            scheduled_kickoff_utc=PLAYED_KICKOFF,
        ),
        match_distributions(0.50, 0.26),
    )
    publish(
        connection,
        Publication(
            "pre_match",
            "lineup",
            "2026-08-22T13:00:00Z",
            COMMIT,
            match_id=PLAYED,
            scheduled_kickoff_utc=PLAYED_KICKOFF,
        ),
        match_distributions(0.55, 0.24),
    )
    record_forecasts_of_record(connection, PLAYED, PLAYED_KICKOFF, PLAYED_KICKOFF)
    publish(
        connection,
        Publication(
            "full_time",
            "final_whistle",
            "2026-08-22T15:55:00Z",
            COMMIT,
            match_id=PLAYED,
            match_minute=90,
            period=2,
            match_state={
                "home_goals": 2,
                "away_goals": 1,
                "home_red_cards": 0,
                "away_red_cards": 0,
            },
        ),
        full_time_distributions(),
    )
    record_void(
        connection,
        "did_not_play",
        "match_players: no minutes",
        COMPUTED_AT,
        match_id=PLAYED,
        quantity_key="P3.goals",
        subject_id=3,
    )
    outcomes = {
        ("M1.result", PLAYED): "home",
        ("M3.goals", PLAYED): 3,
        ("P3.goals", 2): 2,
        ("P3.goals", 5): 1,
        ("L6.plays", 2): True,
        ("L6.plays", 3): False,
        ("L6.plays", 5): True,
    }
    kinds = {quantity.quantity_key: quantity for quantity in QUANTITIES}
    for (quantity_key, subject_id), value in outcomes.items():
        record_outcome(
            connection,
            quantity_key,
            subject_id,
            value,
            "matches",
            "build-1",
            COMPUTED_AT,
            match_id=PLAYED,
        )
    for row in connection.execute(
        "select publication_id, quantity_key, subject_id, line_key from probabilities"
        " where publication_id in (select publication_id from forecasts_of_record)"
    ).fetchall():
        publication_id, quantity_key, subject_id, line_key = row
        quantity = kinds[quantity_key]
        if (quantity_key, subject_id) not in outcomes:
            record_line_check(
                connection,
                publication_id,
                quantity_key,
                subject_id,
                line_key,
                "void",
                None,
                COMPUTED_AT,
            )
            continue
        values = json.loads(
            connection.execute(
                "select distribution_values from distributions where publication_id = ? and"
                " quantity_key = ? and subject_id = ?",
                (publication_id, quantity_key, subject_id),
            ).fetchone()[0]
        )
        happened = (
            read_line(
                quantity.distribution_kind,
                settled(quantity.distribution_kind, outcomes[(quantity_key, subject_id)], values),
                line_key,
            )
            > 0.5
        )
        record_line_check(
            connection,
            publication_id,
            quantity_key,
            subject_id,
            line_key,
            "checked",
            happened,
            COMPUTED_AT,
        )
    record_withdrawal(
        connection,
        first,
        "wrong_datum",
        "The first run read a kick-off an hour early.",
        "1234567",
        "a reviewer",
        "2026-08-20T09:00:00Z",
    )
    upcoming = Publication(
        "pre_match",
        "morning",
        "2026-10-20T06:00:00Z",
        COMMIT,
        match_id=TO_COME,
        scheduled_kickoff_utc="2026-10-25T15:30:00Z",
    )
    publish(
        connection,
        upcoming,
        [
            PublishedDistribution("M1.result", TO_COME, VERSION, result(0.61, 0.22)),
            PublishedDistribution("M3.goals", TO_COME, VERSION, goals()),
        ],
    )
    for published_at, moved_by, leader in (
        ("2026-08-24T06:00:00Z", [PLAYED], ARSENAL),
        ("2026-08-31T06:00:00Z", [], BRENTFORD),
    ):
        order = [leader] + [
            team for team in (ARSENAL, BRENTFORD, BOURNEMOUTH, VILLA) if team != leader
        ]
        publish(
            connection,
            Publication(
                "season_day", "morning", published_at, COMMIT, season_id=SEASON, moved_by=moved_by
            ),
            [
                PublishedDistribution(
                    "T1.final_position",
                    team,
                    VERSION,
                    [0.55 if place == rank else 0.15 for place in range(4)],
                )
                for rank, team in enumerate(order)
            ],
        )
    write_rows(connection)
    connection.close()
    return path


def write_rows(connection) -> None:
    """What is published where, the calibration, the scores, the explanations and the expected
    eleven, through Vesta's writer."""
    for quantity in QUANTITIES:
        register_published_quantity(
            connection,
            quantity.quantity_key,
            1,
            VERSION,
            "2026-08-01T00:00:00Z",
            quantity.subject_kind == "match",
            quantity.quantity_key == "M1.result",
        )
    buckets = [
        Bucket(
            tenth / 10,
            (tenth + 1) / 10,
            tenth / 10 + 0.05,
            tenth / 10 + 0.04,
            20 + tenth,
            max(0.0, tenth / 10 - 0.1),
            min(1.0, tenth / 10 + 0.2),
        )
        for tenth in range(10)
    ]
    for competition in (0, 1):
        record_calibration(connection, "results", VERSION, competition, "all", buckets, COMPUTED_AT)
    # The likeliest outcome per competition season (schema 4): an older round kept, and the
    # latest round's row of every version together and of the version alone.
    record_hits(
        connection,
        [TrackRecordHits("M1.result", "all", 1, SEASON, 1, 1, 0, 0, 0, 1)],
        "2026-08-25T05:00:00Z",
    )
    record_hits(
        connection,
        [
            TrackRecordHits("M1.result", "all", 1, SEASON, 2, 1, 1, 1, 1, 0),
            TrackRecordHits("M1.result", VERSION, 1, SEASON, 2, 1, 1, 1, 1, 0),
        ],
        COMPUTED_AT,
    )
    # Each key scored on its own, and the kind's summary skill, as the check job writes them
    # (Vesta's proposal of 28 October 2026): raw losses of different kinds are never averaged.
    record_key_scores(
        connection,
        [
            KeyScore(kind, key, VERSION, competition, "all", metric, value, baseline,
                     1 - value / baseline, 1)
            for competition in (0, 1)
            for kind, key, metric, value, baseline in KEY_SCORES
        ],
        COMPUTED_AT,
    )  # fmt: skip
    record_scores(
        connection,
        [
            TrackRecordScore(
                "results", VERSION, competition, "all", "skill", RESULTS_SKILL,
                "the league's frequencies", 0.0, 1, 0, 1, 0,
            )
            for competition in (0, 1)
        ],
        COMPUTED_AT,
    )  # fmt: skip
    # Vesta v1's closed record: 40 rows the world does not link, counted beside the scores.
    register_model_version(
        connection, VESTA_ONE, "spine", "2025-08-01T00:00:00Z", "The first Vesta, imported."
    )
    record_scores(
        connection,
        [
            TrackRecordScore(
                "results", VESTA_ONE, 0, "all", "log_loss", 1.01, "the league's frequencies",
                1.07, 300, 5, 0, VESTA_ONE_UNMATCHED,
            ),
            TrackRecordScore(
                "results", VESTA_ONE, 0, "all", "skill",
                1 - VESTA_ONE_RESULTS[0] / VESTA_ONE_RESULTS[1], "the league's frequencies",
                0.0, 300, 5, 0, VESTA_ONE_UNMATCHED,
            ),
            # every version together: the counts summed, the skill of the pooled losses
            TrackRecordScore(
                "results", "all", 0, "all", "skill", ALL_RESULTS_SKILL,
                "the league's frequencies", 0.0, 301, 5, 1, VESTA_ONE_UNMATCHED,
            ),
            TrackRecordScore(
                "results", "all", 1, "all", "skill", RESULTS_SKILL,
                "the league's frequencies", 0.0, 1, 0, 1, 0,
            ),
        ],
        COMPUTED_AT,
    )  # fmt: skip
    lineup = connection.execute(
        "select publication_id from publications where trigger = 'lineup'"
    ).fetchone()[0]
    factors = [
        {"label": "Attack", "weight_points": 7.5, "direction": "for", "note": None},
        {"label": "Home ground", "weight_points": 5.0, "direction": "for", "note": None},
        {"label": "Villa's defence", "weight_points": 2.0, "direction": "against", "note": None},
    ]
    for line_key, sentence in (
        ("", "Arsenal are favourites, mostly on their attack and the home ground."),
        ("outcome=home", "Arsenal win 55 % of the time, mostly on their attack."),
    ):
        record_explanation(
            connection,
            Explanation(
                lineup,
                "M1.result",
                PLAYED,
                VERSION,
                sentence,
                factors,
                [{"step": 1, "label": "Expected goals", "values": {"home": 1.7, "away": 1.1}}],
                [{"change": "Saka does not start", "effect_points": -4.0}],
                line_key=line_key,
            ),
        )
    write_expected_elevens(connection, lineup)
    write_caveats(connection)
    write_changes(connection, lineup)


# Vesta's caveats (forecast log schema 5, agreed with Vesta on 2 November 2026), in the table the
# log's schema creates: on the first publication of the played match, never on the lineup one,
# whose sheets make the absences known.
CAVEATS = (
    ("injuries_not_included", "Injuries are not yet included in this chance."),
    ("suspensions_not_included", "Suspensions are not yet included in this chance."),
)


def write_caveats(connection) -> None:
    first = connection.execute(
        "select publication_id from publications where trigger = 'first' and match_id = ?",
        (PLAYED,),
    ).fetchone()[0]
    connection.executemany(
        "insert into publication_caveats values (?, ?, ?, ?, ?)",
        [
            (first, kind, code, description, VERSION)
            for kind in ("lineups", "scorers", "player_counts")
            for code, description in CAVEATS
        ],
    )
    connection.commit()


def write_changes(connection, lineup: int) -> None:
    """Why the lineup publication moved (forecast log schema 5, Vesta's c6bae6d1), in the table
    the log's schema creates: Arsenal's eleven changed, a new release for the whole match
    (team 0), and Arsenal's official sheet read with one player the world does not know yet."""
    previous = connection.execute(
        "select max(publication_id) from publications where match_id = ? and stage = 'pre_match'"
        " and publication_id < ?",
        (PLAYED, lineup),
    ).fetchone()[0]
    connection.executemany(
        "insert into publication_changes values (?, ?, ?, ?, ?, ?)",
        [
            (
                lineup,
                previous,
                "eleven_changed",
                ARSENAL,
                json.dumps({"in": [19], "out": [4]}),
                VERSION,
            ),
            (
                lineup,
                previous,
                "release_changed",
                0,
                json.dumps({"from": "vesta-2.0.0-rc", "to": VERSION}),
                VERSION,
            ),
            (
                lineup,
                previous,
                "sheet_announced",
                ARSENAL,
                json.dumps({"version": 1, "unknown_players": 1}),
                VERSION,
            ),
        ],
    )
    connection.commit()


FOUR_THREE_THREE = ("G", "RB", "CD-R", "CD-L", "LB", "DM", "CM-R", "CM-L", "RW", "F", "LW")
ARSENAL_ELEVEN = (1, 11, 12, 13, 14, 15, 16, 17, 2, 18, 19)


def write_expected_elevens(connection, publication_id: int) -> None:
    """Arsenal's expected eleven at the lineup publication, one player per slot, through
    Vesta's writer."""
    slots = [
        ExpectedSlot(
            slot,
            player_id,
            position_code,
            0.97 if player_id in (1, 2) else 0.80,
            0.90 if player_id in (1, 2) else 0.70,
        )
        for slot, (position_code, player_id) in enumerate(
            zip(FOUR_THREE_THREE, ARSENAL_ELEVEN, strict=True), start=1
        )
    ]
    record_expected_eleven(connection, publication_id, ARSENAL, "4-3-3", 0.62, slots, VERSION)


