# The Vesta data format (VDF 1)

Status: VDF 1, current since 9 November 2026. The tables as SQL are in `schema.sql` beside this
file, written from the dataset itself at every release.

The input contract of Vesta: what a football dataset must hold for Vesta to forecast from it.
The 11footy dataset (refreshed after each round) **is** this format. Another dataset is
normalised into it by whoever makes it. Vesta then serves every forecast the data allows and
says why it skips the rest (`python -m vesta.capabilities`).

## General rules

- **One SQLite file.** Table and column names are those listed here. A dataset may hold more
  tables and columns: Vesta ignores them.
- **Ids** are the dataset's own integers, unique per table and stable from one build to the
  next. Vesta never reads a provider's id.
- **No provider names.** No column says where a row came from. Where Vesta must tell kinds of
  rows apart, a neutral `layer` column does it (below).
- **Dates and times.** `kickoff_utc` is ISO 8601 in UTC, `YYYY-MM-DDTHH:MM:SSZ`; `local_date` is
  `YYYY-MM-DD` in the competition's time zone.
- **Seasons** are named by their first calendar year (2025 for 2025-26 and for the 2025
  calendar-year season).
- **Unknown is NULL**, never 0. A 0 is a count of zero.
- **Value sets** are closed: a value outside the lists below is a format error, unless the list
  says "other values allowed".

## Header (required)

`format_version`, one row:

| column | type | meaning |
|---|---|---|
| version | TEXT | `VDF 1` |
| dataset | TEXT | the dataset's name; `11footy` is ours (Vesta reuses its released fits only for it) |
| built_at | TEXT | ISO UTC instant of the build |

## Required block: results

Enough for result and goals in the cups and the nations, and for the baseline in the
leagues (see "What each block unlocks").

### competitions

| column | type | rules |
|---|---|---|
| competition_id | INTEGER PK | |
| code | TEXT UNIQUE | a short code; our codes (PL, LL, SA, BL, L1, CL, EL, ECL, CLQ, ELQ, ECLQ, WC, EURO, ...) carry Vesta's shipped rules; another code brings its own rules file |
| kind | TEXT | league, cup, cup qualifying, super cup, nations qualifying, nations league, nations tournament, friendly |
| time_zone | TEXT | IANA name, e.g. Europe/London |

### seasons

| column | type | rules |
|---|---|---|
| season_id | INTEGER PK | |
| competition_id | INTEGER | → competitions |
| season | INTEGER | first calendar year; UNIQUE with competition_id |
| status | TEXT | `closed` when every match is over; any other value is read as open |

### teams

| column | type | rules |
|---|---|---|
| team_id | INTEGER PK | |
| kind | TEXT | club, nation |
| name | TEXT | for a nation, the same string as `venues.country` of its grounds (nations' hosts) |

### matches

| column | type | rules |
|---|---|---|
| match_id | INTEGER PK | |
| season_id | INTEGER | → seasons |
| kickoff_utc | TEXT | ISO UTC; with `kickoff_precision` date, any time of that day |
| kickoff_precision | TEXT | minute, date |
| local_date | TEXT | the match's day where it is played |
| home_team_id, away_team_id | INTEGER | → teams; the home side as listed |
| neutral_venue | INTEGER | 0, 1 |
| status | TEXT | played, awarded, scheduled, postponed, suspended, cancelled, abandoned |
| home_goals, away_goals | INTEGER | the final result (with extra time, without the shoot-out); NULL until played; for `awarded`, the result awarded |
| home_goals_on_pitch, away_goals_on_pitch | INTEGER | optional: the score played when the result was awarded otherwise; NULL when the same |

A match is a result when `status` is played or awarded and both goals are set. A match is
to play when `status` is scheduled or postponed.

## Optional blocks

Each block is present when its tables exist and hold rows for the competition and seasons
concerned. The capability check reports, per competition, which blocks it found and how far
back.

Optional in the required block: `matches.matchday` (INTEGER), the round of a league or of a
cup's league phase. With it, `vesta.run` forecasts each competition's next round as the
dataset names it; without it, the round is derived (a league: the next matches until a team
repeats; a cup: the stage and leg; nations: each nation's next match of the window).

### B1 half time

`matches.home_goals_half_time`, `matches.away_goals_half_time` (INTEGER, NULL when unknown).

### B2 goals in time (events)

`match_events`:

| column | type | rules |
|---|---|---|
| event_id | INTEGER PK | |
| match_id | INTEGER | → matches |
| sequence | INTEGER | order within the match; UNIQUE with match_id |
| period | INTEGER | 1, 2 (halves); 3, 4 (extra time); 5 shoot-out |
| minute | INTEGER | the clock's minute in its period (45 for any first-half added time) |
| added_minute | INTEGER | the minute of added time, NULL when none |
| kind | TEXT | goal, penalty_goal, own_goal, yellow_card, second_yellow_card, red_card; other values allowed (ignored) |
| team_id | INTEGER | the side of the player; for an own goal, the scorer's side |
| related_team_id | INTEGER | for an own goal, the side it counts for |
| player_id | INTEGER | the scorer or the carded player; ids of the dataset's players |
| related_player_id | INTEGER | the assister of a goal, NULL when none or unknown |
| shot_id | INTEGER | → shots, the goal's shot when known |
| layer | TEXT | summary (the match's key events), play_by_play (a running commentary), correction |

A match's goals are read from one layer: `correction` over `summary` over `play_by_play`. Red
cards are known for a match when its summary or play-by-play layer is present. In
the layer read, the goals of each side in periods 1 and 2 must add up to its regular-time
score. A match where they do not is listed by the check and kept out of the goal-time history,
not recounted.

### B3 shots

`shots`:

| column | type | rules |
|---|---|---|
| shot_id | INTEGER PK | |
| match_id, team_id | INTEGER | |
| player_id, assist_player_id | INTEGER | NULL when unknown or unassisted |
| period, minute, added_minute | INTEGER | as in match_events |
| x_m, y_m | REAL | metres on a standard 105 x 68 pitch, in one fixed orientation (below); NULL on a commentary row without a position |
| result | TEXT | goal, saved, blocked, woodwork, off_target |
| situation | TEXT | open_play, corner, set_piece, direct_free_kick, penalty |
| body_part | TEXT | right_foot, left_foot, head, other, unknown |
| is_penalty | INTEGER | 0, 1 |
| zone | TEXT | six_yard_box, penalty_area, outside_box; NULL when unknown |
| layer | TEXT | tracked (a shot-by-shot feed: every row has its position and its context), commentary (shots read from a running commentary; a position when the commentary gives one) |
| xg | REAL | the chance of a goal. In our dataset Vesta writes ours (`vesta.jobs.dataset_xg`, the released model). In a foreign one it is that dataset's own, read as given, with `not_validated_on_dataset`; without it the league champion is not served (Vesta never computes xG for foreign data) |

Orientation: `x_m` runs from 0 on the shooting side's own goal line to 105 on the goal it
attacks (the shooting side attacks left to right); `y_m` runs from 0 to 68 across the pitch,
with y = 0 on the attacker's left touchline; the attacked goal's centre is at (105, 34). This
is the common analytics convention.

### B4 team statistics

`match_team_statistics`, one row per match, team and layer (PK match_id, team_id, `layer`):
`shots`, `shots_on_target`, `corners`, `fouls`, `offsides`, `yellow_cards`, `red_cards`
(INTEGER, NULL when not counted). `layer` is `report` (the match report's count) or `feed` (the
live feed's box score). A dataset with one count per match ships `report` rows only. Ours ships
both where both exist, as each was recorded (VDF S2, agreed on 8 November).

Vesta reads them as its verdicts did:
- in the leagues, the `report` row, else the `feed` row (offsides: `feed` only);
- in the cups, and for the five leagues inside the cups' model, the `feed` row;
- a `feed` row whose fouls and corners are both 0 on both sides is a stub: unknown, but for the
  cards.

### B5 lineups

`match_players` (PK match_id, team_id, player_id): `started` (0, 1), `minutes_played`
(INTEGER), `position_group` (goalkeeper, defender, midfielder, forward), `position_code`
(below), `formation_slot` (1-11 for a starter, NULL otherwise).

`match_team_sheets` (PK match_id, team_id): `formation` (TEXT such as `4-3-3`, NULL when unknown).

Position codes (`position_code`), read by group (`vesta.lineups.model.classify_position`):
goalkeeper G; defender D, SW, DC and every code starting CD, LB or RB; forward F, FW, LF, RF,
RCF and every code starting CF; SUB or NULL for a player who did not start; any other code is a
midfielder (e.g. M, DM, CM, AM, LM, RM).

### B6 player statistics

`player_match_statistics`, one row per match and player (PK match_id, team_id, player_id):
`assists`, `shots`, `shots_on_target`, `key_passes`, `fouls_committed`, `fouls_suffered`,
`yellow_cards`, `offsides`, `saves` (INTEGER; `saves` NULL for an outfield player).
One count per player and match; ours is the live feed's (no other provider's rows).

### B7 cup structure

On `matches`: `stage` (vocabulary below), `group_name`, `leg` (1, 2, NULL), `tie_id` (→
match_ties), `winner_team_id` (the side that goes through or wins, shoot-out included),
`home_goals_regular_time`, `away_goals_regular_time` (the score after 90 minutes where extra time
was played), `extra_time_played` (0, 1), `venue_id` (→ venues).

`venues`: `venue_id` INTEGER PK, `country` TEXT (the country's name, as the nations' names).

`match_ties`: `tie_id` INTEGER PK, `season_id`, `first_leg_match_id`, `second_leg_match_id`,
`winner_team_id`, `decided_by` (aggregate, away_goals, extra_time, shootout, ruling, NULL).

Stage vocabulary (case ignored): league phase; group stage, or `group ` followed by its letter;
knockout round playoffs; round of 32; round of 16; quarter-finals (or quarterfinals);
semi-finals (or semifinals); final; qualifying rounds named with `qualifying` or `preliminary`;
play-off round (or playoff round, playoffs, playoff); in a league, play-off (excluded from the
league's table).

### B8 standings

`standings` (PK season_id, after_matchday, team_id): `position`, `points_deducted` (INTEGER, 0
when none).

### B9 bans

`suspensions_by_rule`: `player_id`, `competition_id`, `served_in_match_ids` (TEXT, match ids
separated by commas). Optional: without it the lineups run without bans.

### R competition rules (beside the data)

Rules per competition code, in Vesta's own files: tie-break rules, tournament formats, tie rules
(away goals), qualification and play-off rules, cut lines. Vesta ships them for our codes. A
foreign competition with its own code brings a rules file in the same form; without one, its
season simulations are not served.

## What each block unlocks

| forecasts | blocks |
|---|---|
| result and goals, cups and nations | required, B7 |
| result and goals, leagues, the full model | required, B3 with `xg`, B5 |
| result and goals, leagues, the basic model | required |
| halves | B1 |
| when the goals come | B1, B2 |
| how the first goal is scored | B2, B3 |
| team counts (corners, cards, shots) | B4 (cups: and B7) |
| lineups | B5 (bans: B9) |
| scorers | B2, B3, B5 |
| player counts | B5, B6 |
| a league's final table | required, R (deductions: B8) |
| cup progression, nations' tournaments | required, B7, R |

Each also needs enough history, from Vesta's released settings; the capability check gives the
reason when it is short.

## Running Vesta on a dataset

    python -m vesta.capabilities <dataset.db>
    python -m vesta.run <dataset.db> [<more split files> ...] [--log <forecast-log.db>]

The first says what can be served and why the rest cannot. The second computes the forecasts of
the coming matches and seasons the dataset allows, and writes them to the forecast log. On a
dataset other than 11footy's, every model refits on it and every forecast says that no blind test
of Vesta covers it.
