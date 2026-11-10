-- The forecast log of Vesta (docs/FORECAST_LOG.md). Append-only: every table refuses UPDATE and
-- DELETE by a trigger, except the two closing dates that are written once (a version superseded,
-- a quantity no longer published). A mistake is a new row: a withdrawal, a new outcome.
-- Instants are ISO 8601 UTC text (2026-10-24T19:00:00Z), which orders as the instants do.
-- `window_key` is '' for a single match and names the window of entity W otherwise.

CREATE TABLE model_versions (
    model_version TEXT PRIMARY KEY,
    family_member TEXT NOT NULL,
    model_detail TEXT,
    freeze_tag TEXT,
    verdict_id TEXT,
    published_from TEXT NOT NULL,
    superseded_from TEXT,
    what_changed TEXT NOT NULL
);

CREATE TABLE quantities (
    quantity_key TEXT PRIMARY KEY,
    catalogue_id TEXT NOT NULL,
    probability_kind TEXT NOT NULL CHECK (probability_kind IN ('results', 'goals', 'timing',
        'team_counts', 'scorers', 'player_counts', 'lineups', 'combinations', 'seasons')),
    subject_kind TEXT NOT NULL CHECK (subject_kind IN ('match', 'team', 'player', 'season_team',
        'season_player', 'competition')),
    name TEXT NOT NULL,
    period TEXT NOT NULL,
    breakdown TEXT NOT NULL DEFAULT '',
    distribution_kind TEXT NOT NULL CHECK (distribution_kind IN ('binary', 'categorical',
        'ordered', 'count', 'joint_count', 'time_to_event', 'rank')),
    conditional_on_playing INTEGER NOT NULL CHECK (conditional_on_playing IN (0, 1)),
    outcome_rule TEXT NOT NULL,
    unit TEXT,
    lines TEXT NOT NULL  -- JSON array of the line keys shown (vesta.forecast_reading)
);

CREATE TABLE published_quantities (
    quantity_key TEXT NOT NULL REFERENCES quantities,
    competition_id INTEGER NOT NULL,
    model_version TEXT NOT NULL REFERENCES model_versions,
    published_from TEXT NOT NULL,
    published_until TEXT,
    moves_live INTEGER NOT NULL CHECK (moves_live IN (0, 1)),
    live_headline INTEGER NOT NULL CHECK (live_headline IN (0, 1)),
    PRIMARY KEY (quantity_key, competition_id, model_version)
);

CREATE TABLE publications (
    publication_id INTEGER PRIMARY KEY AUTOINCREMENT,  -- grows, never reused (§14)
    stage TEXT NOT NULL CHECK (stage IN ('pre_match', 'live', 'full_time', 'season_day', 'day')),
    -- schema version 12 (8 November 2026): `after_kickoff`, a match of the round under way when
    -- the dataset was built, forecast from the facts known at the round's first kick-off
    trigger TEXT NOT NULL CHECK (trigger IN ('first', 'morning', 'lineup', 'minute',
        'final_whistle', 'after_kickoff')),
    published_at TEXT NOT NULL,
    match_id INTEGER,
    match_reference TEXT,  -- the source's reference of a Vesta v1 match the world does not link
    season_id INTEGER,
    scheduled_kickoff_utc TEXT,
    match_minute INTEGER,
    added_minute INTEGER,
    period INTEGER,
    match_state TEXT,  -- JSON: score and red cards the minute was computed from
    events_since_previous TEXT,  -- JSON: the live.db events since the previous minute
    moved_by TEXT,  -- JSON: the match ids finished since the previous morning
    world_build TEXT,
    live_observed_at TEXT,
    code_commit TEXT NOT NULL,
    -- season_day only: 1 when the tie-break rule used that morning is last season's carried
    -- forward by a decision (vesta.season.standings.CARRIED_FORWARD), 0 when verified
    tie_break_carried_forward INTEGER CHECK (tie_break_carried_forward IS NULL
        OR (tie_break_carried_forward IN (0, 1) AND stage = 'season_day')),
    -- live only (schema version 7): 1 for the publication of the half-time break, 0 for a
    -- minute in play
    at_break INTEGER CHECK (at_break IS NULL OR (at_break IN (0, 1) AND stage = 'live')),
    CHECK (stage IN ('season_day', 'day') OR match_id IS NOT NULL OR match_reference IS NOT NULL),
    CHECK (stage <> 'season_day' OR season_id IS NOT NULL),
    -- the morning across competitions (C4.*.day only, the writer's rule): no season, no match
    CHECK (stage <> 'day' OR (season_id IS NULL AND match_id IS NULL
        AND match_reference IS NULL)),
    CHECK (stage <> 'pre_match' OR trigger = 'after_kickoff'
        OR published_at < scheduled_kickoff_utc),
    CHECK (stage <> 'live' OR (match_minute IS NOT NULL AND period IS NOT NULL
        AND live_observed_at IS NOT NULL))
);
CREATE INDEX publications_by_match ON publications (match_id, publication_id);
CREATE INDEX publications_by_season ON publications (season_id, publication_id);

CREATE TABLE distributions (
    publication_id INTEGER NOT NULL REFERENCES publications,
    quantity_key TEXT NOT NULL REFERENCES quantities,
    subject_id INTEGER NOT NULL,
    window_key TEXT NOT NULL DEFAULT '',
    model_version TEXT NOT NULL REFERENCES model_versions,
    distribution_values TEXT NOT NULL,  -- JSON in the shape of the kind (§6)
    expected_value REAL,
    values_sha256 TEXT NOT NULL,
    -- JSON array of the line keys decided by the facts and the rules (their value exactly 0 or 1)
    decided_lines TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (publication_id, quantity_key, subject_id, window_key)
);
CREATE INDEX distributions_by_quantity ON distributions (quantity_key, subject_id, publication_id);

CREATE TABLE live_minutes (
    publication_id INTEGER PRIMARY KEY REFERENCES publications,
    match_id INTEGER NOT NULL,
    model_versions TEXT NOT NULL,  -- JSON: the version of each family member used
    distributions BLOB NOT NULL,  -- gzip of the JSON array of the minute's distributions
    distributions_sha256 TEXT NOT NULL
);

-- schema version 3 (30 October 2026): each live minute also writes the
-- lines of its series quantities (config/vesta/spine.toml [live]) as rows, in the same
-- transaction as its live_minutes row, so a window's series is read without opening the minutes
CREATE TABLE live_headlines (
    publication_id INTEGER NOT NULL REFERENCES publications,
    match_id INTEGER NOT NULL,
    quantity_key TEXT NOT NULL REFERENCES quantities,
    subject_id INTEGER NOT NULL,
    window_key TEXT NOT NULL DEFAULT '',
    line_key TEXT NOT NULL,
    probability REAL NOT NULL CHECK (probability BETWEEN 0 AND 1),
    decided INTEGER NOT NULL DEFAULT 0 CHECK (decided IN (0, 1)),
    model_version TEXT NOT NULL REFERENCES model_versions,
    PRIMARY KEY (publication_id, quantity_key, subject_id, window_key, line_key),
    CHECK (decided = 0 OR probability IN (0, 1))
);
CREATE INDEX live_headlines_by_match
    ON live_headlines (match_id, quantity_key, line_key, publication_id);

CREATE TABLE probabilities (
    publication_id INTEGER NOT NULL REFERENCES publications,
    quantity_key TEXT NOT NULL REFERENCES quantities,
    subject_id INTEGER NOT NULL,
    window_key TEXT NOT NULL DEFAULT '',
    line_key TEXT NOT NULL,
    probability REAL NOT NULL CHECK (probability BETWEEN 0 AND 1),
    decided INTEGER NOT NULL DEFAULT 0 CHECK (decided IN (0, 1)),
    model_version TEXT NOT NULL REFERENCES model_versions,
    PRIMARY KEY (publication_id, quantity_key, subject_id, window_key, line_key),
    CHECK (decided = 0 OR probability IN (0, 1))
);

CREATE TABLE forecasts_of_record (
    match_id INTEGER NOT NULL,
    quantity_key TEXT NOT NULL REFERENCES quantities,
    subject_id INTEGER NOT NULL,
    publication_id INTEGER NOT NULL REFERENCES publications,
    real_kickoff_utc TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (match_id, quantity_key, subject_id)
);
-- the movement view marks each publication that holds a forecast of record
CREATE INDEX forecasts_of_record_by_publication ON forecasts_of_record (publication_id);

CREATE TABLE line_checks (
    check_id INTEGER PRIMARY KEY AUTOINCREMENT,
    publication_id INTEGER NOT NULL REFERENCES publications,
    quantity_key TEXT NOT NULL REFERENCES quantities,
    subject_id INTEGER NOT NULL,
    line_key TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'checked', 'void')),
    happened INTEGER CHECK (happened IN (0, 1)),
    checked_at TEXT NOT NULL,
    CHECK ((status = 'checked') = (happened IS NOT NULL))
);
CREATE INDEX line_checks_by_line
    ON line_checks (publication_id, quantity_key, subject_id, line_key, check_id);

CREATE TABLE outcomes (
    outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
    quantity_key TEXT NOT NULL REFERENCES quantities,
    subject_id INTEGER NOT NULL,
    window_key TEXT NOT NULL DEFAULT '',
    match_id INTEGER,
    season_id INTEGER,
    outcome_value TEXT NOT NULL,  -- JSON in the shape of what happened for the kind
    source_tables TEXT NOT NULL,
    world_build TEXT NOT NULL,
    correction_reason TEXT,  -- null on the first outcome; the reason on a later correction
    recorded_at TEXT NOT NULL,
    CHECK (match_id IS NOT NULL OR season_id IS NOT NULL)
);
CREATE INDEX outcomes_by_quantity
    ON outcomes (quantity_key, subject_id, window_key, match_id, season_id, outcome_id);

CREATE TABLE voids (
    void_id INTEGER PRIMARY KEY AUTOINCREMENT,
    quantity_key TEXT REFERENCES quantities,  -- null: every quantity of the match
    subject_id INTEGER,  -- null: every subject
    match_id INTEGER,
    match_reference TEXT,
    -- awarded: a result decided off the pitch is void (28 October 2026)
    reason TEXT NOT NULL CHECK (reason IN ('cancelled', 'abandoned', 'awarded',
        'did_not_play')),
    evidence TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    CHECK (match_id IS NOT NULL OR match_reference IS NOT NULL)
);
CREATE INDEX voids_by_match ON voids (match_id);

CREATE TABLE withdrawals (
    withdrawal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    publication_id INTEGER NOT NULL UNIQUE REFERENCES publications,
    cause TEXT NOT NULL CHECK (cause IN ('bug', 'wrong_datum')),
    description TEXT NOT NULL,
    fixing_commit TEXT NOT NULL CHECK (length(fixing_commit) >= 7),
    reviewed_by TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);

CREATE TABLE calibration_buckets (
    probability_kind TEXT NOT NULL,
    model_version TEXT NOT NULL REFERENCES model_versions,
    competition_id INTEGER NOT NULL,  -- 0: every competition
    period TEXT NOT NULL,  -- season:<year>, month:<YYYY-MM>, all
    said_from REAL NOT NULL,
    said_to REAL NOT NULL,
    said_mean REAL NOT NULL,
    happened_share REAL NOT NULL,
    count INTEGER NOT NULL,
    chance_band_low REAL NOT NULL,
    chance_band_high REAL NOT NULL,
    computed_at TEXT NOT NULL,
    PRIMARY KEY (probability_kind, model_version, competition_id, period, said_from, computed_at)
);

-- schema version 9 (5 November 2026: Vesta is shown as one thing): model_version
-- 'all' is every version together, each key's own and reference losses pooled over all their
-- forecasts before its skill is taken (not a model version: no reference, as the hits' 'all');
-- never an average of the versions' rows.
CREATE TABLE track_record_scores (
    probability_kind TEXT NOT NULL,
    model_version TEXT NOT NULL,
    competition_id INTEGER NOT NULL,  -- 0: every competition
    period TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    baseline TEXT NOT NULL,
    baseline_value REAL NOT NULL,
    count INTEGER NOT NULL,
    void_count INTEGER NOT NULL,
    withdrawn_count INTEGER NOT NULL,
    unmatched_count INTEGER NOT NULL,
    computed_at TEXT NOT NULL,
    PRIMARY KEY (probability_kind, model_version, competition_id, period, metric, computed_at)
);

-- Each quantity key scored on its own beside the naive reference on the same forecasts, and its
-- skill (1 - value / baseline_value); `track_record_scores` holds each probability kind's
-- summary, the deciding metric's skill averaged across its keys (metric `skill`): raw losses of
-- different distribution kinds are never averaged (26 October 2026).
CREATE TABLE track_record_key_scores (
    probability_kind TEXT NOT NULL,
    quantity_key TEXT NOT NULL REFERENCES quantities,
    model_version TEXT NOT NULL REFERENCES model_versions,
    competition_id INTEGER NOT NULL,  -- 0: every competition
    period TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL,
    baseline_value REAL NOT NULL,
    skill REAL NOT NULL,
    count INTEGER NOT NULL,
    computed_at TEXT NOT NULL,
    PRIMARY KEY (quantity_key, model_version, competition_id, period, metric, computed_at)
);

-- schema version 4 (30 October 2026): per competition season,
-- in how many counted matches the likeliest outcome of the forecast of record happened; written
-- by the check job each round beside the scores (FORECAST_LOG §9). model_version 'all' is every
-- version of the season together (not a model version: no reference).
CREATE TABLE track_record_hits (
    quantity_key TEXT NOT NULL REFERENCES quantities,
    model_version TEXT NOT NULL,
    competition_id INTEGER NOT NULL,
    season_id INTEGER NOT NULL,
    matches INTEGER NOT NULL,
    hits INTEGER NOT NULL CHECK (hits <= matches),
    ties INTEGER NOT NULL,
    void_count INTEGER NOT NULL,
    withdrawn_count INTEGER NOT NULL,
    pending_count INTEGER NOT NULL,
    computed_at TEXT NOT NULL,
    PRIMARY KEY (quantity_key, model_version, competition_id, season_id, computed_at)
);

-- Our expected goals, a derived fact (decision of 22 October 2026, own metrics only): each
-- shot's chance of becoming a goal and each match's totals, by the model version that rated them.
CREATE TABLE shot_expected_goals (
    shot_id INTEGER NOT NULL,
    model_version TEXT NOT NULL REFERENCES model_versions,
    match_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    player_id INTEGER,
    chance REAL NOT NULL CHECK (chance BETWEEN 0 AND 1),
    rated_at TEXT NOT NULL,
    PRIMARY KEY (shot_id, model_version)
);
CREATE INDEX shot_expected_goals_by_match ON shot_expected_goals (match_id, model_version);
CREATE INDEX shot_expected_goals_by_player ON shot_expected_goals (player_id, model_version);

CREATE TABLE match_expected_goals (
    match_id INTEGER NOT NULL,
    model_version TEXT NOT NULL REFERENCES model_versions,
    home_expected_goals REAL NOT NULL,
    away_expected_goals REAL NOT NULL,
    home_shots INTEGER NOT NULL,
    away_shots INTEGER NOT NULL,
    rated_at TEXT NOT NULL,
    PRIMARY KEY (match_id, model_version)
);

CREATE TABLE explanations (
    publication_id INTEGER NOT NULL REFERENCES publications,
    quantity_key TEXT NOT NULL REFERENCES quantities,
    subject_id INTEGER NOT NULL,
    window_key TEXT NOT NULL DEFAULT '',
    line_key TEXT NOT NULL DEFAULT '',  -- '' explains the whole distribution, else one line
    model_version TEXT NOT NULL REFERENCES model_versions,
    sentence TEXT NOT NULL,
    factors TEXT NOT NULL,
    chain TEXT NOT NULL,
    sensitivity TEXT NOT NULL,
    data_note TEXT,
    PRIMARY KEY (publication_id, quantity_key, subject_id, window_key, line_key)
);

-- The expected eleven of a team at a pre-match publication: one player per slot of the likeliest
-- formation, the assignment that makes the eleven likeliest; written by the lineups model (2c).
CREATE TABLE expected_elevens (
    publication_id INTEGER NOT NULL REFERENCES publications,
    team_id INTEGER NOT NULL,
    formation_slot INTEGER NOT NULL CHECK (formation_slot BETWEEN 1 AND 11),
    formation TEXT NOT NULL,
    probability_formation REAL NOT NULL CHECK (probability_formation BETWEEN 0 AND 1),
    player_id INTEGER NOT NULL,
    position_code TEXT NOT NULL,
    probability_starts REAL NOT NULL CHECK (probability_starts BETWEEN 0 AND 1),
    probability_position REAL NOT NULL CHECK (probability_position BETWEEN 0 AND 1),
    model_version TEXT NOT NULL REFERENCES model_versions,
    PRIMARY KEY (publication_id, team_id, formation_slot),
    UNIQUE (publication_id, team_id, player_id)
);

-- schema version 5 (2 November 2026): what a publication's
-- chances do not yet account for, per probability kind, from a fixed list of codes (§4); written
-- in the same transaction as the publication, by a rule of the model and the world, so a caveat
-- disappears by itself once the code reads what it lacked and the data is there.
CREATE TABLE publication_caveats (
    publication_id INTEGER NOT NULL REFERENCES publications,
    probability_kind TEXT NOT NULL CHECK (probability_kind IN ('results', 'goals', 'timing',
        'team_counts', 'scorers', 'player_counts', 'lineups', 'combinations', 'seasons')),
    -- schema version 6 (3 November 2026): `suspensions_from_cards`, where the bans are read
    -- schema version 8 (4 November 2026): `shots_stale`, the scorers on earlier matches' shots
    -- schema version 10 (8 November 2026, VDF S1): `not_validated_on_dataset`, a forecast made
    -- from a dataset no blind verdict of Vesta covers
    -- schema version 11 (8 November 2026, the round snapshot): `before_previous_match`, a
    -- fixture of the round forecast before one of its teams' previous match was played
    -- schema version 12 (8 November 2026): `forecast_after_kickoff`, the caveat of every
    -- publication with the `after_kickoff` trigger
    code TEXT NOT NULL CHECK (code IN ('suspensions_not_included', 'suspensions_from_cards',
        'injuries_not_included', 'shots_stale', 'not_validated_on_dataset',
        'before_previous_match', 'forecast_after_kickoff')),
    description TEXT NOT NULL,
    model_version TEXT NOT NULL REFERENCES model_versions,
    PRIMARY KEY (publication_id, probability_kind, code)
);

-- schema version 5: why a pre-match forecast moved since the match's previous pre-match
-- publication, facts only, from a fixed list of codes (§4); `detail` is JSON of ids, never
-- prose. team_id 0: the whole match. None written for a first publication, nor when nothing
-- Vesta knows of changed.
CREATE TABLE publication_changes (
    publication_id INTEGER NOT NULL REFERENCES publications,
    previous_publication_id INTEGER REFERENCES publications,
    code TEXT NOT NULL CHECK (code IN ('weekly_fit', 'eleven_changed', 'sheet_announced',
        'ban_started', 'ban_ended', 'release_changed', 'source_stale', 'data_correction')),
    team_id INTEGER NOT NULL,
    detail TEXT NOT NULL,
    model_version TEXT NOT NULL REFERENCES model_versions,
    PRIMARY KEY (publication_id, code, team_id)
);

-- schema version 5: what a pre-match publication read for each side, kept so the next one can
-- say what changed (Vesta's own record; no reader need show it): the Monday of the weekly fit
-- (UTC date, null without a fit) and the eleven the player ratings read (JSON, sorted ids).
CREATE TABLE publication_inputs (
    publication_id INTEGER NOT NULL REFERENCES publications,
    team_id INTEGER NOT NULL,
    fit_week TEXT,
    eleven TEXT NOT NULL,
    PRIMARY KEY (publication_id, team_id)
);

-- The views a reader reads (§4).

-- Every publication of a match in time order, from the first to the final whistle, without the
-- late pre-match ones (published at or after the real kick-off written with the forecasts of
-- record). `of_record` marks the publications that hold a forecast of record.
CREATE VIEW movement AS
SELECT p.match_id, p.publication_id, p.stage, p.trigger, p.published_at, p.match_minute,
       p.added_minute, p.period, p.match_state,
       EXISTS (SELECT 1 FROM forecasts_of_record r
               WHERE r.publication_id = p.publication_id) AS of_record
FROM publications p
WHERE p.match_id IS NOT NULL AND p.stage IN ('pre_match', 'live', 'full_time')
  AND NOT (p.stage = 'pre_match' AND p.published_at >= COALESCE(
      (SELECT MIN(r.real_kickoff_utc) FROM forecasts_of_record r WHERE r.match_id = p.match_id),
      '9999-12-31T23:59:59Z'));

CREATE VIEW season_evolution AS
SELECT p.season_id, d.quantity_key, d.subject_id, d.window_key, p.publication_id,
       p.published_at, p.moved_by, d.model_version, d.distribution_values, d.expected_value
FROM publications p JOIN distributions d ON d.publication_id = p.publication_id
WHERE p.stage = 'season_day';

-- Each line of a forecast of record beside its last check, and its state on the track record.
CREATE VIEW checks AS
SELECT r.match_id, pr.publication_id, pr.quantity_key, pr.subject_id, pr.line_key,
       pr.probability, pr.model_version, p.published_at,
       c.status AS check_status, c.happened, c.checked_at,
       CASE
           WHEN w.withdrawal_id IS NOT NULL THEN 'withdrawn'
           WHEN c.status = 'void' THEN 'void'
           WHEN c.status = 'checked' THEN 'counted'
           ELSE 'pending'
       END AS state
FROM forecasts_of_record r
JOIN publications p ON p.publication_id = r.publication_id
JOIN probabilities pr ON pr.publication_id = r.publication_id
    AND pr.quantity_key = r.quantity_key AND pr.subject_id = r.subject_id
LEFT JOIN line_checks c ON c.check_id = (
    SELECT MAX(last.check_id) FROM line_checks last
    WHERE last.publication_id = pr.publication_id AND last.quantity_key = pr.quantity_key
      AND last.subject_id = pr.subject_id AND last.line_key = pr.line_key)
LEFT JOIN withdrawals w ON w.publication_id = r.publication_id;

CREATE VIEW withdrawn_publications AS
SELECT w.withdrawal_id, w.publication_id, p.stage, p.published_at, p.match_id, p.season_id,
       w.cause, w.description, w.fixing_commit, w.reviewed_by, w.recorded_at
FROM withdrawals w JOIN publications p ON p.publication_id = w.publication_id;
