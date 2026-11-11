# Vesta's track record

Every pre-match forecast of record Vesta 2.x has published, raw: each probability it
gave, and beside it what happened. Nothing is selected or removed: the misses with
the hits, the withdrawn forecasts marked, the matches still to be played marked
pending. Written each week from Vesta's forecast log by `python -m vesta.record`.

- versions: Vesta 2.x, every version
- dataset_built_at: 2026-11-10T12:21:18Z
- last_published_at: 2026-11-08T10:58:51Z
- forecast_lines: 395647
- counted_lines: 199160

The full record is attached to the latest dataset release of this repository
(Releases): `track-record.db` (SQLite) and `track-record.csv.gz` (the same rows as
CSV, gzipped). This folder holds this file, `calibration.csv` and `summary.md`
(the numbers shown in 11footy's README).

## track-record.db, view `forecast_lines` (and track-record.csv.gz)

One row per line of a forecast: a probability and whether that line happened. The
database keeps each match, quantity and forecast once (tables `matches`,
`quantities`, `forecasts`, `lines`); the view joins them back.

| column | meaning |
|---|---|
| match_id | the match's id in the dataset |
| competition | the competition's code |
| competition_name | the competition's name (its code where the dataset has none) |
| season | the season (the year it starts) |
| kickoff_utc | the kick-off the forecast was made for, UTC |
| home_team | the home side |
| away_team | the away side |
| published_at | when Vesta published the forecast, UTC, before the kick-off |
| model_version | the Vesta version that made it (a '-dataset' version: made on a dataset no verdict covered, with that caveat) |
| quantity_key | what is forecast (e.g. M1.result, the match result) |
| quantity_name | the same in words |
| probability_kind | the family of the quantity (results, goals, team counts...) |
| subject | who it is about: empty for the match, else the team or the player |
| line | the outcome the probability is for (e.g. outcome=home, over=2.5); for a quantity forecast over windows of the match, `<window>:<line>` |
| probability | the chance Vesta gave that line, from 0 to 1 |
| state | counted (the match was played and the line checked), pending (not yet), void (no outcome: cancelled, abandoned, awarded, or a player who did not play), withdrawn (published, then withdrawn for a bug or a wrong datum) |
| happened | 1 when the line happened, 0 when it did not; empty unless counted |
| withdrawn_because | for a withdrawn forecast: bug or wrong_datum |
| outcome | what happened, as recorded when the match was checked (JSON) |
| checked_at | when the line was last checked, UTC |
| subject_id | the id in the dataset of the match, the team or the player |

## calibration.csv (and track-record.db, table `calibration`)

The one summary: when Vesta said a chance within a tenth (say 30% to 40%), how often
it happened. When Vesta is well calibrated, happened_share is close to said_mean.
Counted lines only.

| column | meaning |
|---|---|
| probability_kind | the family of the quantities, or 'all' for every family |
| said_from | the tenth of the probabilities: from this value |
| said_to | to this one (the last tenth includes 1) |
| said_mean | the mean probability Vesta gave the lines of the tenth |
| happened_share | the share of those lines that happened |
| count | how many counted lines are in the tenth |
