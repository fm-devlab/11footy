"""Writing to the forecast log: publications, the forecasts of record, outcomes, checks,
voids, withdrawals, the map of what is published, the calibration, the scores, the
explanations and the expected elevens (`docs/FORECAST_LOG.md` §4, §7, §8).

Every write is one transaction: a publication and all its distributions land together or not at
all, so no reader ever sees half of one. Every distribution
is checked before it is written: it must sum to one in the shape of its quantity's kind, and
every line its quantity declares must be readable from it (`vesta.forecast_reading`), so no
published distribution can leave a line unanswerable.
"""

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace

from vesta.distributions import COUNT, check_distribution, compute_expected_count, hold_at_edges
from vesta.forecast_log.store import ForecastLogError
from vesta.forecast_reading import read_line
from vesta.research.calibration import Bucket

STAGES_WITH_DISTRIBUTION_ROWS = ("pre_match", "full_time", "season_day", "day")
# the round in numbers of a day spans competitions: its own stage, and nothing else in it
DAY_QUANTITY = re.compile(r"C4\.[a-z_]+\.day")
# a decided line's value read from its distribution may miss 0 or 1 by the rounding of a sum
# of floats; its row is written as exactly 0 or 1
DECIDED_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Publication:
    """One run of Vesta at one instant for one stage (the columns of `publications`)."""

    stage: str
    trigger: str
    published_at: str
    code_commit: str
    match_id: int | None = None
    match_reference: str | None = None
    season_id: int | None = None
    scheduled_kickoff_utc: str | None = None
    # the clock and state of a match in play: written by the live job of the record (until
    # 8 November 2026), kept so the record's rows can be written and read the same way
    match_minute: int | None = None
    added_minute: int | None = None
    period: int | None = None
    match_state: dict | None = None
    events_since_previous: list | None = None
    moved_by: list | None = None
    world_build: str | None = None
    live_observed_at: str | None = None
    tie_break_carried_forward: bool | None = None  # season_day: last season's rule used
    at_break: bool | None = None  # the record's live minutes: the half-time break or in play


@dataclass(frozen=True)
class PublishedDistribution:
    quantity_key: str
    subject_id: int
    model_version: str
    values: object
    window_key: str = ""
    # the lines decided by the facts and the rules, never by a probability that came out 0 or 1
    decided_lines: tuple[str, ...] = ()


# the fixed codes of `publication_caveats` and `publication_changes` (§4), each caveat with the
# sentence the reader sees
CAVEATS = {
    "suspensions_not_included": "Suspensions are not yet included in this chance.",
    "suspensions_from_cards": "Suspensions are derived from the cards the dataset records: a card"
                              " rescinded on appeal or a ban set by a committee may be missed.",
    "injuries_not_included": "Injuries are not yet included in this chance.",
    "shots_stale": "Built on the shots of earlier matches: the dataset does not yet hold the"
                   " located shots of the latest matches.",
    # VDF S1 (schema 10): every publication from a dataset other than ours, and on ours where no
    # released part covers the competition
    "not_validated_on_dataset": "No blind verdict of Vesta covers this dataset.",
    # schema 11: the round snapshot
    "before_previous_match": "Made before one of the teams' previous match: its result is not"
                             " in this chance.",
    # schema 12: a match of the round under way when the dataset was built
    "forecast_after_kickoff": "Made after kick-off from the facts known before this round began.",
}
CHANGES = ("weekly_fit", "eleven_changed", "sheet_announced", "ban_started", "ban_ended",
           "release_changed", "source_stale", "data_correction")  # fmt: skip


@dataclass(frozen=True)
class Caveat:
    """What a publication's chances of one probability kind do not account for."""

    probability_kind: str
    code: str
    model_version: str


@dataclass(frozen=True)
class Change:
    """A fact that changed since the match's previous pre-match publication (team 0: the
    match); `detail` holds ids only."""

    code: str
    team_id: int
    detail: dict
    previous_publication_id: int | None
    model_version: str


@dataclass(frozen=True)
class SideInputs:
    """What a pre-match publication read for one side: the Monday of the weekly fit and the
    eleven of the player ratings."""

    team_id: int
    fit_week: str | None
    eleven: tuple[int, ...]


@dataclass(frozen=True)
class Quantity:
    """One definition of `quantities` (§5)."""

    quantity_key: str
    catalogue_id: str
    probability_kind: str
    subject_kind: str
    name: str
    period: str
    distribution_kind: str
    conditional_on_playing: bool
    outcome_rule: str
    lines: tuple[str, ...]
    breakdown: str = ""
    unit: str | None = None


def register_model_version(
    connection: sqlite3.Connection,
    model_version: str,
    family_member: str,
    published_from: str,
    what_changed: str,
    model_detail: str | None = None,
    freeze_tag: str | None = None,
    verdict_id: str | None = None,
) -> None:
    with _open_transaction(connection):
        connection.execute(
            "INSERT INTO model_versions (model_version, family_member, model_detail, freeze_tag,"
            " verdict_id, published_from, what_changed) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (model_version, family_member, model_detail, freeze_tag, verdict_id, published_from,
             what_changed),  # fmt: skip
        )


def register_quantity(connection: sqlite3.Connection, quantity: Quantity) -> None:
    with _open_transaction(connection):
        connection.execute(
            "INSERT INTO quantities (quantity_key, catalogue_id, probability_kind, subject_kind,"
            " name, period, breakdown, distribution_kind, conditional_on_playing, outcome_rule,"
            " unit, lines) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                quantity.quantity_key,
                quantity.catalogue_id,
                quantity.probability_kind,
                quantity.subject_kind,
                quantity.name,
                quantity.period,
                quantity.breakdown,
                quantity.distribution_kind,
                int(quantity.conditional_on_playing),
                quantity.outcome_rule,
                quantity.unit,
                json.dumps(list(quantity.lines)),
            ),
        )


def publish(
    connection: sqlite3.Connection,
    publication: Publication,
    distributions: list[PublishedDistribution],
    caveats: list[Caveat] | tuple = (),
    changes: list[Change] | tuple = (),
    inputs: list[SideInputs] | tuple = (),
) -> int:
    """Write a pre-match, full-time, season or day publication and its distributions; at the final
    whistle also the rows of its lines (decision A2); with them its caveats, its changes since
    the match's previous pre-match publication and the inputs the next one compares (schema
    version 5). Returns the publication's id."""
    for caveat in caveats:
        if caveat.code not in CAVEATS:
            raise ForecastLogError(f"{caveat.code}: not a caveat of §4")
    for change in changes:
        if change.code not in CHANGES:
            raise ForecastLogError(f"{change.code}: not a change of §4")
    if publication.stage not in STAGES_WITH_DISTRIBUTION_ROWS:
        raise ForecastLogError(f"stage {publication.stage!r} is not written by this code")
    _refuse_empty(distributions)
    for distribution in distributions:
        of_the_day = DAY_QUANTITY.fullmatch(distribution.quantity_key) is not None
        if of_the_day != (publication.stage == "day"):
            raise ForecastLogError(
                f"{distribution.quantity_key}: the stage 'day' carries the C4 quantities of a day,"
                " and they ride in no other stage"
            )
    distributions = [hold_distribution_at_edges(distribution) for distribution in distributions]
    with _open_transaction(connection):
        publication_id = _insert_publication(connection, publication)
        for distribution in distributions:
            kind, lines = _check_against_quantity(connection, distribution)
            connection.execute(
                "INSERT INTO distributions (publication_id, quantity_key, subject_id, window_key,"
                " model_version, distribution_values, expected_value, values_sha256,"
                " decided_lines) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    publication_id,
                    distribution.quantity_key,
                    distribution.subject_id,
                    distribution.window_key,
                    distribution.model_version,
                    *_serialise_values(kind, distribution.values),
                    json.dumps(list(distribution.decided_lines)),
                ),
            )
            if publication.stage == "full_time":
                _insert_lines(connection, publication_id, distribution, kind, lines)
        for caveat in caveats:
            connection.execute(
                "INSERT INTO publication_caveats (publication_id, probability_kind, code,"
                " description, model_version) VALUES (?, ?, ?, ?, ?)",
                (publication_id, caveat.probability_kind, caveat.code, CAVEATS[caveat.code],
                 caveat.model_version))  # fmt: skip
        for change in changes:
            connection.execute(
                "INSERT INTO publication_changes (publication_id, previous_publication_id, code,"
                " team_id, detail, model_version) VALUES (?, ?, ?, ?, ?, ?)",
                (publication_id, change.previous_publication_id, change.code, change.team_id,
                 json.dumps(change.detail, sort_keys=True), change.model_version))  # fmt: skip
        for side in inputs:
            connection.execute(
                "INSERT INTO publication_inputs (publication_id, team_id, fit_week, eleven)"
                " VALUES (?, ?, ?, ?)",
                (publication_id, side.team_id, side.fit_week, json.dumps(sorted(side.eleven))))
    return publication_id


def record_forecasts_of_record(
    connection: sqlite3.Connection, match_id: int, real_kickoff_utc: str, recorded_at: str
) -> int:
    """At the real kick-off: for every quantity and subject of the match, the last pre-match
    publication before it (by its publication time, never one at or after the kick-off) becomes
    the forecast of record, and its lines become rows (decision A2, choice C). Written once per
    match. Returns the number of forecasts of record."""
    with _open_transaction(connection):
        if connection.execute(
            "SELECT 1 FROM forecasts_of_record WHERE match_id = ?", (match_id,)
        ).fetchone():
            raise ForecastLogError(f"match {match_id}: the forecasts of record are written")
        chosen = connection.execute(
            """
            SELECT quantity_key, subject_id, window_key, publication_id FROM (
                SELECT d.quantity_key, d.subject_id, d.window_key, d.publication_id,
                       ROW_NUMBER() OVER (PARTITION BY d.quantity_key, d.subject_id, d.window_key
                                          ORDER BY p.published_at DESC, p.publication_id DESC)
                           AS latest
                FROM distributions d JOIN publications p ON p.publication_id = d.publication_id
                WHERE p.match_id = ? AND p.stage = 'pre_match' AND p.published_at < ?
                  AND p.trigger <> 'after_kickoff')
            WHERE latest = 1
            ORDER BY quantity_key, subject_id, window_key
            """,
            (match_id, real_kickoff_utc),
        ).fetchall()
        for quantity_key, subject_id, window_key, publication_id in chosen:
            connection.execute(
                "INSERT INTO forecasts_of_record (match_id, quantity_key, subject_id,"
                " publication_id, real_kickoff_utc, recorded_at) VALUES (?, ?, ?, ?, ?, ?)",
                (match_id, quantity_key, subject_id, publication_id, real_kickoff_utc,
                 recorded_at),  # fmt: skip
            )
            model_version, values_text, decided_text = connection.execute(
                "SELECT model_version, distribution_values, decided_lines FROM distributions"
                " WHERE publication_id = ? AND quantity_key = ? AND subject_id = ?"
                " AND window_key = ?",
                (publication_id, quantity_key, subject_id, window_key),
            ).fetchone()
            distribution = PublishedDistribution(
                quantity_key, subject_id, model_version, json.loads(values_text), window_key,
                tuple(json.loads(decided_text)),
            )
            kind, lines = _read_quantity_kind_and_lines(connection, quantity_key)
            _insert_lines(connection, publication_id, distribution, kind, lines)
    return len(chosen)


def record_outcome(
    connection: sqlite3.Connection,
    quantity_key: str,
    subject_id: int,
    outcome_value: object,
    source_tables: str,
    world_build: str,
    recorded_at: str,
    match_id: int | None = None,
    season_id: int | None = None,
    window_key: str = "",
    correction_reason: str | None = None,
) -> None:
    """What happened; a later row with its `correction_reason` corrects it, the last counts."""
    with _open_transaction(connection):
        connection.execute(
            "INSERT INTO outcomes (quantity_key, subject_id, window_key, match_id, season_id,"
            " outcome_value, source_tables, world_build, correction_reason, recorded_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (quantity_key, subject_id, window_key, match_id, season_id,
             json.dumps(outcome_value), source_tables, world_build, correction_reason,
             recorded_at),  # fmt: skip
        )


def record_line_check(
    connection: sqlite3.Connection,
    publication_id: int,
    quantity_key: str,
    subject_id: int,
    line_key: str,
    status: str,
    happened: bool | None,
    checked_at: str,
) -> None:
    with _open_transaction(connection):
        connection.execute(
            "INSERT INTO line_checks (publication_id, quantity_key, subject_id, line_key, status,"
            " happened, checked_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (publication_id, quantity_key, subject_id, line_key, status,
             None if happened is None else int(happened), checked_at),  # fmt: skip
        )


def record_void(
    connection: sqlite3.Connection,
    reason: str,
    evidence: str,
    recorded_at: str,
    match_id: int | None = None,
    match_reference: str | None = None,
    quantity_key: str | None = None,
    subject_id: int | None = None,
) -> None:
    with _open_transaction(connection):
        connection.execute(
            "INSERT INTO voids (quantity_key, subject_id, match_id, match_reference, reason,"
            " evidence, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (quantity_key, subject_id, match_id, match_reference, reason, evidence, recorded_at),
        )


def record_withdrawal(
    connection: sqlite3.Connection,
    publication_id: int,
    cause: str,
    description: str,
    fixing_commit: str,
    reviewed_by: str,
    recorded_at: str,
) -> None:
    """A publication found wrong: kept and marked, out of the figures (choice F1, risk K15).
    The cause is documented by the commit that fixes it and the reviewer who checked it."""
    with _open_transaction(connection):
        connection.execute(
            "INSERT INTO withdrawals (publication_id, cause, description, fixing_commit,"
            " reviewed_by, recorded_at) VALUES (?, ?, ?, ?, ?, ?)",
            (publication_id, cause, description, fixing_commit, reviewed_by, recorded_at),
        )



@dataclass(frozen=True)
class TrackRecordScore:
    """One row of `track_record_scores`: a metric of the counted forecasts of record."""

    probability_kind: str
    model_version: str
    competition_id: int  # 0: every competition together
    period: str
    metric: str
    value: float
    baseline: str
    baseline_value: float
    count: int
    void_count: int
    withdrawn_count: int
    unmatched_count: int


@dataclass(frozen=True)
class Explanation:
    """One row of `explanations` (deliverable 2d): the factors of a distribution or of one line."""

    publication_id: int
    quantity_key: str
    subject_id: int
    model_version: str
    sentence: str
    factors: list
    chain: list
    sensitivity: list
    window_key: str = ""
    line_key: str = ""  # '' explains the whole distribution
    data_note: str | None = None


def register_published_quantity(
    connection: sqlite3.Connection,
    quantity_key: str,
    competition_id: int,
    model_version: str,
    published_from: str,
    moves_live: bool,
    live_headline: bool,
) -> None:
    """A quantity starts being published for a competition by a model version (the map a reader
    reads to know what each competition has)."""
    with _open_transaction(connection):
        connection.execute(
            "INSERT INTO published_quantities (quantity_key, competition_id, model_version,"
            " published_from, moves_live, live_headline) VALUES (?, ?, ?, ?, ?, ?)",
            (quantity_key, competition_id, model_version, published_from, int(moves_live),
             int(live_headline)),  # fmt: skip
        )


def supersede(connection: sqlite3.Connection, model_version: str, at: str) -> int:
    """Every other version still open is superseded from `at`, and every quantity it still
    publishes stops being published then (`published_until`): a new release replaces the whole
    previous one, its keys and its competitions; returns the published rows closed."""
    with _open_transaction(connection):
        closed = connection.execute(
            "UPDATE published_quantities SET published_until = ?"
            " WHERE published_until IS NULL AND model_version <> ?", (at, model_version),
        ).rowcount
        connection.execute(
            "UPDATE model_versions SET superseded_from = ?"
            " WHERE superseded_from IS NULL AND model_version <> ?", (at, model_version),
        )
    return closed


def record_calibration(
    connection: sqlite3.Connection,
    probability_kind: str,
    model_version: str,
    competition_id: int,
    period: str,
    buckets: list[Bucket],
    computed_at: str,
) -> None:
    """The calibration of one kind, version, competition (0: all) and period, as computed by the
    check job (`vesta.research.calibration.buckets`); a reader takes the latest `computed_at`."""
    with _open_transaction(connection):
        for bucket in buckets:
            connection.execute(
                "INSERT INTO calibration_buckets (probability_kind, model_version, competition_id,"
                " period, said_from, said_to, said_mean, happened_share, count, chance_band_low,"
                " chance_band_high, computed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (probability_kind, model_version, competition_id, period, bucket.said_from,
                 bucket.said_to, bucket.said_mean, bucket.happened_share, bucket.count,
                 bucket.chance_band_low, bucket.chance_band_high, computed_at),  # fmt: skip
            )


@dataclass(frozen=True)
class TrackRecordHits:
    """One row of `track_record_hits`: a competition season's counted matches whose likeliest
    outcome was a single one, how many of them it happened in, and the matches left out."""

    quantity_key: str
    model_version: str  # 'all': every version of the season together
    competition_id: int
    season_id: int
    matches: int
    hits: int
    ties: int
    void_count: int
    withdrawn_count: int
    pending_count: int


def record_hits(connection: sqlite3.Connection, rows: list[TrackRecordHits],
                computed_at: str) -> None:  # fmt: skip
    """The hits of a round of checks, all with one `computed_at`."""
    with _open_transaction(connection):
        connection.executemany(
            "INSERT INTO track_record_hits (quantity_key, model_version, competition_id,"
            " season_id, matches, hits, ties, void_count, withdrawn_count, pending_count,"
            " computed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(row.quantity_key, row.model_version, row.competition_id, row.season_id,
              row.matches, row.hits, row.ties, row.void_count, row.withdrawn_count,
              row.pending_count, computed_at) for row in rows])  # fmt: skip


def record_scores(
    connection: sqlite3.Connection, scores: list[TrackRecordScore], computed_at: str
) -> None:
    """The scores of a round of checks, all with one `computed_at`."""
    with _open_transaction(connection):
        for score in scores:
            connection.execute(
                "INSERT INTO track_record_scores (probability_kind, model_version, competition_id,"
                " period, metric, value, baseline, baseline_value, count, void_count,"
                " withdrawn_count, unmatched_count, computed_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (score.probability_kind, score.model_version, score.competition_id, score.period,
                 score.metric, score.value, score.baseline, score.baseline_value, score.count,
                 score.void_count, score.withdrawn_count, score.unmatched_count,
                 computed_at),  # fmt: skip
            )


@dataclass(frozen=True)
class KeyScore:
    """One row of `track_record_key_scores`: a metric of one quantity key's counted forecasts of
    record beside the same metric of the naive reference on the same forecasts."""

    probability_kind: str
    quantity_key: str
    model_version: str
    competition_id: int  # 0: every competition together
    period: str
    metric: str
    value: float
    baseline_value: float
    skill: float  # 1 - value / baseline_value: the share of the reference's loss saved
    count: int


def record_key_scores(
    connection: sqlite3.Connection, scores: list[KeyScore], computed_at: str
) -> None:
    """The per-key scores of a round of checks, all with one `computed_at`."""
    with _open_transaction(connection):
        for score in scores:
            connection.execute(
                "INSERT INTO track_record_key_scores (probability_kind, quantity_key,"
                " model_version, competition_id, period, metric, value, baseline_value, skill,"
                " count, computed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (score.probability_kind, score.quantity_key, score.model_version,
                 score.competition_id, score.period, score.metric, score.value,
                 score.baseline_value, score.skill, score.count, computed_at),  # fmt: skip
            )


def record_explanation(connection: sqlite3.Connection, explanation: Explanation) -> None:
    with _open_transaction(connection):
        connection.execute(
            "INSERT INTO explanations (publication_id, quantity_key, subject_id, window_key,"
            " line_key, model_version, sentence, factors, chain, sensitivity, data_note)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (explanation.publication_id, explanation.quantity_key, explanation.subject_id,
             explanation.window_key, explanation.line_key, explanation.model_version,
             explanation.sentence, json.dumps(explanation.factors), json.dumps(explanation.chain),
             json.dumps(explanation.sensitivity), explanation.data_note),  # fmt: skip
        )


ELEVEN = range(1, 12)


@dataclass(frozen=True)
class ExpectedSlot:
    """One slot of an expected eleven: the player the lineups model puts there."""

    formation_slot: int
    player_id: int
    position_code: str
    probability_starts: float
    probability_position: float


def record_expected_eleven(
    connection: sqlite3.Connection,
    publication_id: int,
    team_id: int,
    formation: str,
    probability_formation: float,
    slots: list[ExpectedSlot],
    model_version: str,
) -> None:
    """A team's expected eleven at a pre-match publication, the eleven slots in one transaction:
    one player per slot of the likeliest formation (the lineups model's assignment, 2c)."""
    if sorted(slot.formation_slot for slot in slots) != list(ELEVEN):
        raise ForecastLogError(
            f"team {team_id}: an expected eleven fills the slots 1 to 11 once each, got "
            f"{sorted(slot.formation_slot for slot in slots)}"
        )
    with _open_transaction(connection):
        for slot in slots:
            connection.execute(
                "INSERT INTO expected_elevens (publication_id, team_id, formation_slot, formation,"
                " probability_formation, player_id, position_code, probability_starts,"
                " probability_position, model_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (publication_id, team_id, slot.formation_slot, formation, probability_formation,
                 slot.player_id, slot.position_code, slot.probability_starts,
                 slot.probability_position, model_version),  # fmt: skip
            )





@contextmanager
def _open_transaction(connection: sqlite3.Connection) -> Iterator[None]:
    """One immediate transaction: the writer's lock is taken at the start, so a write never
    fails halfway on a lock, and everything inside commits together or not at all."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    connection.execute("COMMIT")


def _insert_publication(connection: sqlite3.Connection, publication: Publication) -> int:
    cursor = connection.execute(
        "INSERT INTO publications (stage, trigger, published_at, match_id, match_reference,"
        " season_id, scheduled_kickoff_utc, match_minute, added_minute, period, match_state,"
        " events_since_previous, moved_by, world_build, live_observed_at, code_commit,"
        " tie_break_carried_forward, at_break)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            publication.stage,
            publication.trigger,
            publication.published_at,
            publication.match_id,
            publication.match_reference,
            publication.season_id,
            publication.scheduled_kickoff_utc,
            publication.match_minute,
            publication.added_minute,
            publication.period,
            _to_json_or_null(publication.match_state),
            _to_json_or_null(publication.events_since_previous),
            _to_json_or_null(publication.moved_by),
            publication.world_build,
            publication.live_observed_at,
            publication.code_commit,
            None if publication.tie_break_carried_forward is None
            else int(publication.tie_break_carried_forward),
            None if publication.at_break is None else int(publication.at_break),
        ),
    )
    return cursor.lastrowid


def hold_distribution_at_edges(distribution: PublishedDistribution) -> PublishedDistribution:
    """The distribution with its probabilities held at 0 and 1 where a sum of floats strays
    beyond them by the last bits (`vesta.distributions.held_at_edges`)."""
    return replace(distribution, values=hold_at_edges(distribution.values))


def _check_against_quantity(
    connection: sqlite3.Connection, distribution: PublishedDistribution
) -> tuple[str, list[str]]:
    """The kind and lines of the distribution's quantity, after proving the distribution sums to
    one in that kind's shape and answers every line."""
    kind, lines = _read_quantity_kind_and_lines(connection, distribution.quantity_key)
    check_distribution(kind, distribution.values)
    for line_key in lines:
        read_line(kind, distribution.values, line_key)
    for line_key in distribution.decided_lines:
        if line_key not in lines:
            raise ForecastLogError(
                f"{distribution.quantity_key}: the decided line {line_key!r} is not one of"
                " its lines"
            )
        value = read_line(kind, distribution.values, line_key)
        if abs(value - round(value)) > DECIDED_TOLERANCE:
            raise ForecastLogError(
                f"{distribution.quantity_key}: the decided line {line_key!r} is not 0 or 1"
            )
    return kind, lines


def _read_quantity_kind_and_lines(
    connection: sqlite3.Connection, quantity_key: str
) -> tuple[str, list[str]]:
    row = connection.execute(
        "SELECT distribution_kind, lines FROM quantities WHERE quantity_key = ?", (quantity_key,)
    ).fetchone()
    if row is None:
        raise ForecastLogError(f"quantity {quantity_key!r} is not defined in the log")
    return row[0], json.loads(row[1])


def _serialise_values(kind: str, values: object) -> tuple[str, float | None, str]:
    """The distribution's JSON text, its expected value (counts only) and the text's hash."""
    text = json.dumps(values, separators=(",", ":"), sort_keys=True)
    expected = compute_expected_count(values) if kind == COUNT else None
    return text, expected, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _insert_lines(
    connection: sqlite3.Connection,
    publication_id: int,
    distribution: PublishedDistribution,
    kind: str,
    lines: list[str],
) -> None:
    for line_key in lines:
        connection.execute(
            "INSERT INTO probabilities (publication_id, quantity_key, subject_id, window_key,"
            " line_key, probability, model_version, decided) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                publication_id,
                distribution.quantity_key,
                distribution.subject_id,
                distribution.window_key,
                line_key,
                _read_row_probability(kind, distribution, line_key),
                distribution.model_version,
                int(line_key in distribution.decided_lines),
            ),
        )


def _read_row_probability(kind: str, distribution: PublishedDistribution, line_key: str) -> float:
    value = read_line(kind, distribution.values, line_key)
    return float(round(value)) if line_key in distribution.decided_lines else value


def _refuse_empty(distributions: list[PublishedDistribution]) -> None:
    if not distributions:
        raise ForecastLogError("a publication without distributions publishes nothing")


def _to_json_or_null(value: object) -> str | None:
    return None if value is None else json.dumps(value, separators=(",", ":"), sort_keys=True)
