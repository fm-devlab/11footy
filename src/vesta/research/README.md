# The research discipline of Vesta

Status: draft of 23 October 2026. The rules are those kept on 21 October 2026: judged
per kind of probability, a freeze that is the tag of the published model, English words. The
seasons of §3 were decided on 23 October 2026.

## 1. A change of a model is a registered hypothesis

Every change to a published model is first written in `config/vesta/hypotheses.toml` (its
fields at the top of the file) and committed: what changes and why, the catalogue items it
moves, the kind of distribution and the deciding metric, the gain expected, an inner grid of at
most six configurations, the world tables it reads. The runner refuses an entry that is not in
the committed file (`hypotheses.committed_hypothesis`), so no hypothesis is written after its
result is known. Entries are never removed.

## 2. A verdict, with its numbers, before anything is published

`verdicts.judge` compares the candidate with the published model on the same forecasts of the
development seasons, forecast walk-forward (each season by a model fitted only on the earlier
ones, every training row known before the season's first forecast: `seasons`). Per forecast it
takes the difference of the deciding metric (`scores.METRICS_BY_KIND`: the ranked probability
score for ordered outcomes, counts and places, the log loss for the others). The candidate is
kept only when:
- the whole 95 % interval of the mean difference, by a bootstrap of whole blocks (competition x
  ISO week, 2,000 draws, seeded), lies below zero;
- it improves the mean in at least three development seasons;
- its expected calibration error is not worse than the published model's by more than 0.005.

The verdict is written once to `docs/research/verdicts/<id>.md` with its numbers, the commit,
the world build and the command that reproduces it to the bit. The parameters are in
`config/vesta/research.toml`, where they were fixed before any verdict. A kept verdict does not
publish anything: the maintainers' approval does.

## 3. Which seasons, and where the clean measure comes from

Vesta v1 opened its test seasons 2024-25 and 2025-26 once, for its result model, and judged
six later hypotheses on its development seasons only, so its champion has no clean number.
Decided:
- every model is built and chosen on the development seasons 2019-20 to 2023-24
  (`[seasons] development`);
- 2024-25 and 2025-26 stay closed while the work goes on (`[seasons] closed`: no selection
  reads them) and are used once, by their own command, for the final verdict of each new model;
- the spine (the reproduced champion) is the exception: Vesta v1 was already judged on those
  seasons, so its clean measure is forward, on 2026-27, match by match from the forecast log;
  the matching of the old verdicts is a reproduction, not a clean measure, and is
  said so.

**The reads of the closed seasons, per key** (a standing rule since 1 November
2026):
- each key gets one first blind read of 2024-25 and 2025-26 if it has never been read blind,
  after its development passes; a key read blind once is never read blind again, whatever the
  result;
- no second read for any key, except the two granted on 1 November 2026, each
  once, 2 of 2, recorded in the verdict as the second read and bound by a forward guard (the
  2026-27 matches from 2 November, read once on the first world after 31 January 2027; a failure
  there pulls the key): the cups' goals in time and the cups-scope windows (the cups' one second
  read, moved from the cups' spine candidate that failed in development), and
  `T13.goals.next_30_days` (the leagues' one second read). After them, no second read of either
  scope for anything;
- where no read is left, a key is judged forward only, on 2026-27 matches, read once.

**The calibration test against a frequency reference** (`research.toml` `[verdict]`
`frequency_reference_*`): the model's expected calibration error under 0.02, and its buckets'
happened shares inside a chance band that covers them together, 90 % for all the buckets at once
(each bucket at `0.90 ** (1 / k)`). Changed on 1 November 2026 from a 90 % band for each bucket
alone, which fails a calibrated model about two times in three with ten buckets (family A's
finding); for plans declared from then on.

## 4. Leakage: checked on every input, alarmed on every verdict

Every input of a forecast carries the instant it became known, and that instant must not come
after the forecast's (`leakage.check_known_by`). Before a verdict, the alarm compares the
candidate with the rule of its kind (`[alarm.<kind>]`): a gain or an accuracy beyond any honest
change stops the work until
a research record explains it and the hypothesis names it (`alarm_cleared_by`). A kind without a
rule cannot be judged: its rule is written, with its reason, before its first verdict. There is
no switch to turn the alarm off.

## 5. The freeze

A version is published only from the git tag `vesta-<version>` made after its verdict and the
maintainers' approval, and changes only by a new verdict and a new tag.
`freeze.freeze_problems` proves
that the code and configuration of the model at HEAD are those of the tag, with nothing
uncommitted; every distribution in the forecast log carries its version
(`docs/FORECAST_LOG.md`). `vesta.releases.register_release_plan` writes a frozen version into the log
before its first publication (the version with its tag and verdict, its quantities, the map of
what it publishes where) and refuses when the code that runs is not the tag's.

## 6. The same metrics in public

The scores and the calibration (`scores`, `calibration`) are the ones the public check uses
(`docs/FORECAST_LOG.md` §9): a published version and a candidate are measured on one rule, and
every calibration group carries the band a perfectly calibrated forecaster would show on its
count.
