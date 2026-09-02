# xPoints research audit and production roadmap

Last updated: 2 September 2026  
Status: XP-01, XP-03 and the backtest harness are merged; a two-stage, price-blended challenger (XP-04/XP-05 slice) runs in shadow from GW3. See "Where things stand" for the hand-off.  
Scope: the xPoints model, its dbt data dependencies, evaluation, publication, and monitoring

## Executive summary

The paper referenced by the open pull requests is **OpenFPL**, an arXiv paper by Daniel Groos. It is not an FPL Review paper. OpenFPL compared its public-data models with paid, concurrent FPL Review projections.

The most important conclusion from the research is not “add more XGBoost features.” A reliable xPoints product needs this sequence:

1. Build deadline-valid, point-in-time, multi-season data.
2. Build a reproducible forward-testing and scoring system.
3. Establish simple, difficult-to-beat baselines.
4. Model availability and expected minutes.
5. Test position-aware and per-fixture models as challengers.
6. Promote a model only after it wins in shadow mode and passes operational gates.

The existing production feed should remain the official FPL `ep_next` baseline until a challenger passes those gates. The current XGBoost script is experimental and is not suitable for production promotion.

## Where things stand (2 September 2026) and what to do after GW3

This is the hand-off for the next working session. The reasoning is in the research artifact (https://claude.ai/code/artifact/2ca07fa6-120d-4941-b5a3-641f17002cd5) and the website roadmap (`fpl-app/roadmap.md`).

### State

- **Production feed:** FPL `ep_next` baseline via `baseline.py`, published by `.github/workflows/daily_update.yml` (cron `7 * * * *`; GitHub drops most ticks, accepted for now). Rows carry price tie-break ordering and `blend_rank`. Hard bounds −5..40 in `prediction_safety.py`.
- **Archives:** `predictions/gw{N}.json` frozen at each deadline by the `--archive-only` step; `predictions/gw{N}_model.json` is the shadow model's frozen forecast. GW1–GW3 baseline archives exist; GW3 is the first model archive.
- **Shadow model v2 (`model.py`):** `P(60+) × (0.5·E[pts|start] + 0.5·price-implied expectation)`, E[pts|start] a monotone-in-price Tweedie, trained on 2025/26 + 2026/27 joined on stable player `code`. Chosen because in `backtest.py` two-stage gave the best MAE while nothing learned from one season out-ranks price among starters, so the price prior is blended in.
- **Scoring:** `score.py` runs after every tick, grades gameweeks whose `data_checked` is true, writes `scores/gw{N}.json` + `gw{N}_rows.csv` and rebuilds `scorecard.json` (populations all / played / starters; MAE, RMSE, Spearman with bootstrap CI, precision@k with random tie-breaks, captain regret, power statement, `summary.model_vs_ep_next_starter_spearman`). `--gw N` and `--force` rescore.
- **Public view:** https://fplanaly.st/app/accuracy renders `scorecard.json` (30-minute ISR). Model columns are dashes until a paired gameweek exists.
- **Harness:** `backtest.py` walk-forward with as-of (shift-then-roll) features and a leakage test; results in `backtests/*.json`. Ablations complete: opponent form (no gain), prior-season priors (no gain), two-stage (best MAE, starters still lose to price ordering).
- **Numbers to remember:** GW1/GW2 `ep_next` MAE 1.598 / 1.467 against predict-zero 1.593 / 1.450 (zero wins pooled MAE both weeks); starter ρ 0.086 [−0.05, 0.22] and 0.173 [0.04, 0.32]; `ep_next` ≈ price ranking (ρ 0.73); starter-ρ sd ≈ 0.091, so detecting +0.05 needs ~27 gameweeks at 80% power.

### After GW3 (deadline Friday 4 September 2026 17:30 UTC; FPL usually flips `data_checked` on the Monday or Tuesday)

1. Confirm `scores/gw3.json` exists and the GW3 row of `scorecard.json` carries `model_*` fields, with `summary.model_vs_ep_next_starter_spearman` non-null. If the score step warned in the workflow run, run `python score.py --gw 3 --force` locally and open a PR with only the code fix; generated files are committed by the bot and never belong in feature PRs.
2. Check that `predictions/gw3.json` and `predictions/gw3_model.json` were written before the deadline. A missing model archive means the shadow step failed silently; read the run's warnings.
3. Treat one gameweek as one sample. A single-GW lead of ±0.1 in starter ρ is noise. Do not promote. Record the row in the research artifact's progress section and on `/app/accuracy` (automatic).
4. Then work the queue in this order:
   1. **XP-05 slice:** improve the P(60+) stage with FPL `status`, `news`, `chance_of_playing_next_round` and `starts` as of the deadline. Evaluate in `backtest.py` first; ship as shadow v3 with a new `model_version` and the same `gw{N}_model.json` naming so the scorecard stays paired.
   2. **Promotion gate in `score.py`:** a rolling rule over the last six graded gameweeks (model starter ρ and precision@20 at or above `ep_next`, paired CI excluding zero) that writes `promotion_ready` into `scorecard.json`. The site keeps `ep_next` until the flag is true and a human flips the mode.
   3. **DBT-01:** season-aware snapshot keys and blocking gates in `fpl_dbt`, required before any multi-season training mart.
   4. **Parked, do not reopen without new evidence:** opponent form, prior-season priors, a Railway-dispatch trigger for the hourly job, the dbt #18 rebase.

### Working rules agreed with the owner

- `ep_next` stays champion until the model demonstrably wins over a sustained window. Shadow first, always.
- Generated artifacts (`predictions*.json`, `scores/`, `scorecard.json`) never go in feature PRs.
- A PR is ready only after CI is green.
- Secrets are never pasted in chat; the owner creates tokens.

## What has already been done

### Production safety

- [PR #10](https://github.com/meharpalbasi/xPoints/pull/10) replaced the stale prior-season fallback with current-season FPL `ep_next` data.
- Prediction rows now include season, gameweek, player identity, source, model version, generated time, team, and position provenance.
- Pre-deadline prediction files are archived by gameweek instead of every run destroying the previous forecast.
- [PR #12](https://github.com/meharpalbasi/xPoints/pull/12) derives fixtures from one validated global fixture response, distinguishes an unknown fixture feed from a confirmed blank gameweek, validates complete player coverage, and writes atomically.
- Model mode is manual; the scheduled production mode is the baseline.
- If experimental model generation fails, the workflow can fall back to the baseline.
- Five publication and fixture-safety unit tests pass locally.

### Current operational snapshot

At the time of this update, the latest remote prediction artifact is:

| Field | Value |
|---|---|
| Season / target | 2026/27, GW3 |
| Generated | 31 August 2026 at 21:31:50 UTC |
| Players | 626 |
| Positive projections | 363 |
| Zero projections | 263 |
| Invalid or negative values | 0 |
| Source | `fpl_ep_next` |
| Version | `ep-next-baseline-1` |

The 31 August scheduled prediction run succeeded. Three preceding scheduled runs failed because FPL supplied negative `ep_next` values for a small number of players. The validator correctly prevented a rejected candidate from replacing the last good feed, but the incident exposes a remaining policy gap: a finite negative expected score can be legitimate in FPL. The next reliability change must use documented bounds and anomaly alerts instead of treating every negative value as a corrupt feed.

The Daily FPL Metrics Generation workflow most recently succeeded. It is not evidence that the model is accurate or production-ready. Its large legacy artifact and independent FPL crawl should be retired after dbt supplies the canonical history and feature export.

## What the research actually says

### OpenFPL

[OpenFPL](https://arxiv.org/html/2508.09992v1) trained on FPL and Understat data from 2020/21 to 2023/24, then evaluated on prospectively collected 2024/25 data from gameweeks 32–38. Its main design choices were:

- separate models by position;
- XGBoost and Random Forest regressors;
- player, team, and opponent history;
- rolling horizons of 1, 3, 5, 10, and 38 matches;
- current player availability;
- position-specific target weighting;
- a median ensemble of tuned models;
- one-, two-, and three-gameweek forecast horizons;
- evaluation by position and by zeros, blanks, tickers, and haulers;
- predictions collected before the gameweek deadline;
- comparison with a Last-5 baseline and concurrent FPL Review predictions.

OpenFPL is strong evidence that position-aware public-data ensembles deserve a controlled reproduction. It does **not** prove that every feature, window, sample-weighting choice, or ensemble member caused the reported improvement. The paper has a short prospective test window and does not publish a full ablation. Its architecture is therefore a challenger specification, not a recipe to copy directly into production.

OpenFPL also performed better on some higher-return cohorts while FPL Review was stronger on non-appearance and low-return cohorts. That makes expected minutes and availability the highest-value reliability track for this project.

### FPL Review

FPL Review’s public documentation describes:

- expected value built from probability distributions over scoring events;
- an expected-minutes process incorporating starts, cameos, injury, rotation, tactical context, news, and human review;
- team strength, tactical role, penalties, rotation, recency, and market information;
- frequent updates and a deadline rerun.

Its implementation is proprietary. We should use those concepts as hypotheses, not state that we know its internal architecture.

The old PR plan proposed:

```text
xPoints_if_plays * P(60+ minutes)
```

That is not an adequate expected-value calculation. FPL Review explicitly notes that xMins affects EV non-linearly. Starts, cameos, the 60-minute appearance threshold, clean-sheet eligibility, substitutions, and event rates must be represented separately or learned jointly.

### AIrsenal and related work

[AIrsenal’s prediction notes](https://github.com/alan-turing-institute/AIrsenal/blob/main/NOTES.md#player-points-predictions) support a component-oriented alternative: model team scoring, player goal/assist involvement, recent minutes and availability, then calculate each fixture before summing a gameweek.

[Deep Learning and Transfer Learning Architectures for EPL Player Performance Forecasting](https://arxiv.org/html/2405.02412) found useful signal in recent points, minutes, fixture difficulty, and ICT features and reported promising position-specific sequence models. Its evaluation excluded zero-minute players and is not sufficient evidence for using a neural model in a full-player production feed. Ridge and LightGBM-style baselines are more useful immediate additions; sequence models belong in a later challenger experiment.

[Multi-stream Data Analytics for Enhanced Performance Prediction in Fantasy Football](https://arxiv.org/abs/1912.07441) supports testing structured external signals such as market data and public information. Such data must have a captured availability time, usage rights, sufficient historical coverage, and demonstrated incremental value.

### Production ML engineering

[Google’s Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml/) reinforces the main ordering of this roadmap: establish metrics and a robust pipeline, keep the first model simple, test infrastructure independently, detect silent failures, validate before export, and prevent training-serving skew.

## Open pull-request disposition

| PR | Current state | Decision | Ideas to preserve |
|---|---|---|---|
| [#3 — quick wins](https://github.com/meharpalbasi/xPoints/pull/3) | Open, stale, conflicting, no CI evidence | Close as superseded; do not merge or rebase as one change | Gameweek-aware validation, BPS/bonus/saves/cards/ICT features, per-position and return-bucket evaluation, tree-only scaler removal |
| [#8 — cache summaries](https://github.com/meharpalbasi/xPoints/pull/8) | Open, stale, conflicting, no CI evidence | Close as superseded | One immutable response per player per run, retry/coverage tests, eventual canonical dbt input |
| [#4 — position models](https://github.com/meharpalbasi/xPoints/pull/4) | Closed | Research reference only | Modular layout, position awareness, XGB/RF challenger |
| [#5 — Understat/odds](https://github.com/meharpalbasi/xPoints/pull/5) | Closed | Research reference only | Player/team/opponent feature separation and optional market priors |

PR #3 cannot be safely merged because it predates the current publication contract and still contains point-in-time leakage, stale inference features, optimistic evaluation, duplicate API work, and no behavioral tests. PR #8’s original optimization is largely superseded by deriving fixtures from the global feed.

## Current model gaps

### Data correctness

- `script.py` trains only on the current season. Early-season samples are too small, particularly for goalkeeper and forward position models.
- Current bootstrap values such as `form`, ownership, status, and chance of playing are copied onto old match rows. This leaks future information into historical training and evaluation.
- Training examples do not record when each input became available. An event time alone does not prove that a feature was known before the deadline.
- Player IDs are reused between seasons. Stable player code plus season-aware identity is required.
- The dbt player snapshot currently keys only on `element_id`; it must include season before it can be a trustworthy training source.
- dbt staging intentionally exposes only the current season. A separate, tested multi-season training mart is missing.
- A failed player-summary request returns empty arrays and can look like genuine no-history data. Unknown, missing, and confirmed zero must be distinct states.
- The Vaastav historical dataset is useful for bootstrapping, but its own documentation warns that some `xP` values may be captured after matches. Any such field must be excluded unless its pre-deadline timestamp is proven.

### Feature correctness

- Shifted rolling values are correct for historical target rows, but next-gameweek inference copies the latest already-shifted row. This omits the most recently completed match.
- Multiple fixtures in one gameweek can leak within the same locked gameweek when features shift by fixture row instead of using a gameweek cutoff.
- DGWs are handled by multiplying one prediction by fixture count. Each fixture has a different opponent, venue, rest period, and minutes risk and must be predicted separately.
- Availability is represented by coarse live fields rather than a calibrated appearance/minutes model.
- The feature builder used for training is not the same versioned code path used for inference.

### Evaluation

- The current row-based `TimeSeriesSplit` can mix players from the same gameweek across training and validation.
- A last-three-gameweek holdout is too small and is repeatedly consulted during development.
- There is no immutable feature snapshot corresponding to each archived prediction.
- There is no automated scorer comparing deadline predictions with final actuals.
- There is no current comparison with `ep_next`, Last-5, position mean, or a regularised linear baseline.
- There are no confidence intervals, ablations, cold-start slices, non-appearance calibration, or decision metrics.
- The absolute RMSE/MAE/R-squared “ceiling” in PR #3 is not a valid acceptance target. It came from a non-comparable dataset and method. Promotion must use paired evaluation on this project’s own frozen snapshots.

### Operations

- The prediction workflow uses Python 3.9 while tests use Python 3.11.
- Training, hyperparameter search, evaluation, and deadline inference are coupled in one script.
- No persisted champion model artifact or reproducible training manifest exists.
- Fallback use is recorded in the run summary but does not create an alert when the overall workflow remains green.
- Scheduled jobs can start late and are not driven by the actual FPL deadline.
- There is no model registry, shadow deployment, automatic scorecard, rollback pointer, or drift monitoring.
- The JSON file in GitHub is functioning as a production data plane.

## Non-negotiable design rules

1. **Deadline truth:** every feature has `event_time` and `available_at`; only data available before the target deadline may be used.
2. **One feature path:** training and inference use the same versioned feature definitions.
3. **One canonical identity:** player identity is season-aware and anchored by stable player code where available.
4. **Unknown is not zero:** upstream failure, missing history, confirmed blank gameweek, and a genuine zero are different states.
5. **All selectable players count:** evaluation includes non-appearances. Dropping zero-minute players produces a misleadingly easy task.
6. **Per-fixture first:** calculate fixture-level predictions, then aggregate to player-gameweek output.
7. **Simple champion:** the production baseline remains champion until a challenger wins prospectively.
8. **Evidence before complexity:** every added data source, feature family, model type, and ensemble member requires an ablation.
9. **Offline training, fast inference:** never tune hundreds of models in the deadline publication job.
10. **Reversible publication:** validate an immutable candidate, then update one production pointer; retain instant rollback.

## Target data and prediction contract

The canonical modelling grain should be one player-fixture prediction plus one derived player-gameweek aggregate.

Required run-level fields:

- `schema_version`
- `run_id`
- `code_sha`
- `season`
- `target_gameweek`
- `deadline_time`
- `generated_at`
- `feature_cutoff`
- `data_snapshot_id`
- `training_snapshot_id`
- `feature_set_version`
- `model_version`
- `source`
- `model_status`
- `fallback_used`

Required player-fixture fields:

- current player ID and stable player code;
- team, opponent, fixture ID, position, venue, and kickoff;
- confirmed fixture count at the run cutoff;
- availability state and its source time;
- probability of appearance, start, 60+ minutes, and expected minutes;
- expected scoring components where available;
- `xpoints_mean` and calibrated distribution or interval fields;
- fallback reason and data-quality flags.

The aggregate player-gameweek record is the sum of its fixture records, not one forecast multiplied by fixture count.

## Evaluation protocol

### Frozen forward test

For each target gameweek:

```text
freeze all inputs available at the deadline
train only on earlier eligible data
predict every selectable player and fixture
store features, model, output, and manifest immutably
wait until official results are final/data_checked
score the untouched prediction
advance to the next gameweek
```

Use multi-season rolling-origin evaluation. Keep the latest available season or season segment as an outer prospective holdout. Use only earlier periods for hyperparameter selection.

### Required baselines

- official FPL `ep_next` captured at the same deadline;
- zero/position mean;
- previous five appearances and previous five gameweeks;
- a minutes-aware player/team/opponent rate model;
- regularised Ridge regression;
- current production champion.

### Metrics

Primary model metrics:

- MAE and RMSE;
- mean per-gameweek Spearman rank correlation;
- calibration of average predicted versus average actual points.

Required slices:

- GKP, DEF, MID, and FWD;
- non-appearance zeros, playing blanks, tickers, and haulers;
- projected minutes bands;
- home/away;
- single, double, and blank gameweeks;
- new signings, promoted teams, and players with little history;
- one-, two-, and three-gameweek horizons;
- top-10, top-30, and captain candidate cohorts.

Product metrics:

- captain regret against the best eligible candidate;
- transfer gain/regret using only deadline-valid information;
- top-k hit rate and ranking stability;
- Brier score and calibration error for appearance/start/60+ probabilities;
- interval coverage and width.

Use paired, gameweek-blocked bootstrap intervals. Do not promote on a single aggregate score. Pre-register the primary metric and practical improvement margin before evaluating a candidate.

## Prioritized implementation roadmap

Each item below should be a separate pull request with its own tests and plain-English release note.

### P0 — make the existing baseline dependable

#### XP-01: Workflow and source resilience

Purpose: keep the current baseline available without hiding degraded operation.

- Upgrade scheduled and test workflows to the same supported Python version.
- Add a shared `requests.Session` client with connect/read timeouts, bounded retry, exponential backoff, `Retry-After`, schema validation, and request metrics.
- Define documented hard and warning bounds for expected points. Preserve legitimate negative values in evaluation; do not silently clip them to zero.
- Add freshness, player-parity, duplicate, fixture-coverage, and distribution-change checks.
- Alert when fallback is used, when the previous good artifact is retained, or when the feed is older than its service-level objective.
- Trigger final runs from actual deadline times at multiple offsets, with a manual runbook.
- Prevent overlapping workflow runs and make publication idempotent.
- Test partial responses, malformed JSON, 429s, 5xx, timeouts, negative values, and stale data.

Acceptance:

- simulated upstream failures never replace the last good artifact;
- a small legitimate negative projection follows the documented policy;
- every degraded run is visible without reading raw logs;
- 14 consecutive scheduled shadow runs complete or preserve a valid artifact with an actionable alert.

#### DBT-01: Season-safe source foundation

Purpose: make dbt safe to use as the canonical model data source.

- Change snapshot identity from `element_id` to a season-aware key.
- Preserve immutable bootstrap, history, fixture, and rules snapshots with `loaded_at` and season.
- Make source freshness, uniqueness, season, coverage, and relationship tests blocking before promotion.
- Distinguish genuine empty arrays from failed fetches; retry and quarantine incomplete batches.
- Add ingestion run IDs, expected/received counts, failed IDs, and atomic warehouse promotion.
- Repair or quarantine rollover-corrupted price history before model use.

Acceptance:

- a season rollover cannot join a reused player ID to last season’s player;
- an incomplete ingest cannot become the latest production dataset;
- dbt tests fail the pipeline rather than logging and continuing.

### P1 — create trustworthy model evidence

#### DBT-02: Point-in-time, multi-season training mart

Purpose: provide one reproducible source for training and inference.

- Build player-fixture facts across available seasons.
- Add stable player/team/fixture identity and deadline timestamps.
- Preserve non-appearances and postponed/cancelled fixture state.
- Create point-in-time bootstrap/availability/price/ownership features using `available_at` joins.
- Add shifted player, team, and opponent aggregates for 1, 3, 5, 10, and 38 matches.
- Export an immutable Parquet dataset and manifest with row counts, schema hash, min/max dates, missingness, and source versions.
- Exclude leakage-prone fields such as post-match `xP` unless pre-deadline capture is proven.

Acceptance:

- a leakage test proves no feature timestamp exceeds its target deadline;
- re-running a versioned snapshot produces the same rows and hash;
- fixture and player coverage reconcile to official source counts.

#### XP-02: Shared feature builder and contracts

Purpose: remove training-serving skew and current inference bugs.

- Introduce typed/configured data, feature, model, and output modules.
- Use the same feature builder for historical examples and live inference.
- Fix next-gameweek rolling features to include the latest completed eligible match.
- Aggregate or cut off same-gameweek fixtures so one DGW match cannot inform another prediction made before the deadline.
- Add explicit cold-start priors and missingness indicators.
- Add golden-dataset, feature-value, DGW, promoted-player, and training-versus-serving parity tests.

Acceptance:

- the same frozen input produces identical features in training and inference;
- the latest completed match is included exactly once;
- no target-gameweek event is included in any target-gameweek feature.

#### XP-03: Evaluation and scorecard framework

Purpose: know whether any model is better before changing production.

- Score archived predictions against final official results.
- Store immutable evaluation JSON and a human-readable model card.
- Implement rolling-origin outer evaluation and gameweek-aware inner tuning.
- Add all required baselines, metrics, slices, and paired confidence intervals.
- Add ablation support and side-by-side candidate reports.
- Publish a cumulative and per-gameweek baseline scorecard.

Acceptance:

- one command reproduces a report from a data and code manifest;
- intentional leakage and same-GW mixing tests fail;
- aggregate metrics reconcile to per-gameweek scored rows;
- no candidate can publish without an attached evaluation artifact.

### P1 — establish strong simple challengers

#### XP-04: Corrected pooled direct-points champion candidate

Purpose: create a reliable benchmark before position-specific complexity.

- Add Last-5, Ridge, Random Forest/LightGBM-style, and pooled XGBoost candidates.
- Use position interactions and position-aware features while keeping enough data for early-season and sparse positions.
- Add shifted FPL points, minutes, starts, saves, BPS, bonus, cards, ICT, official expected stats, team strength, opponent strength, venue, rest, and congestion.
- Remove StandardScaler only from tree-only pipelines; keep preprocessing inside reproducible model pipelines where required.
- Tune offline with early stopping and persist the winning artifact.
- Run feature-family and horizon ablations.

Acceptance:

- the candidate beats the strongest simple baseline under the pre-registered protocol or remains experimental;
- no important position or minutes cohort has an unexplained material regression;
- artifact load/predict parity is tested.

#### XP-05: Appearance and expected-minutes model

Purpose: improve the weakest and most operationally important cohort: players who may not play or may not reach 60 minutes.

- Model probability of appearance, probability of starting, probability of 60+, and conditional minutes.
- Use recent starts/minutes, substitution patterns, availability history, rest, congestion, role security, and timestamped team news where licensed.
- Calibrate probabilities by position and role.
- Support a clearly attributed manual override with author, reason, timestamp, and expiry.
- Combine minutes with event probabilities non-linearly; compare staged and joint approaches.

Acceptance:

- probabilities are calibrated on forward data;
- non-appearance and cameo cohorts improve without degrading regular starters;
- every manual override is auditable and expires automatically.

### P2 — reproduce and challenge the research

#### XP-06: OpenFPL-style position ensemble experiment

Purpose: test the strongest idea from the arXiv paper on this project’s data.

- Reproduce position-specific XGBoost and Random Forest challengers.
- Test horizons 1, 3, 5, 10, and 38 by ablation rather than assuming all are useful.
- Reproduce position-specific sample weighting faithfully, report effective weights, and compare it with unweighted and robust-loss alternatives.
- Compare pooled-with-interactions, separate-position, and hierarchical/partial-pooling approaches.
- Test median ensembles against single models and smaller ensembles.
- Document divergences from the paper.

Acceptance:

- gains survive the frozen outer evaluation and gameweek-blocked uncertainty analysis;
- sample weighting does not create extreme influence from tiny bins;
- each retained model or feature family demonstrates incremental value.

#### XP-07: Per-fixture component and hybrid model

Purpose: improve interpretability, DGWs, clean-sheet logic, and expected-minutes interaction.

- Predict each fixture independently.
- Estimate team scoring/conceding and clean-sheet probabilities.
- Estimate player goal, assist, appearance, 60+, save, card, defensive-contribution, and bonus/BPS components where supportable.
- Convert components using versioned FPL scoring rules.
- Compare direct-points, component, and stacked hybrid predictions.
- Model correlations where they materially affect distribution/decision outputs.

Acceptance:

- component expectations reconcile to the total;
- DGW totals equal the sum of individually inspectable fixtures;
- the hybrid improves calibration or decision metrics, not only one aggregate RMSE.

### P3 — optional data and advanced reliability

#### XP-08: External-data experiments

Purpose: add information only when rights, reliability, and measurable lift justify it.

Order of evaluation:

1. licensed historical market probabilities as team-scoring and clean-sheet priors;
2. licensed lineup, injury, and team-news data for expected minutes;
3. Understat-style player/team/opponent features from an approved, stable source;
4. sequence models such as a compact 1D CNN;
5. NLP/news only after structured sources plateau.

For every source, record terms, licence, redistribution rights, cost, latency, historical availability, coverage, outage behavior, and a kill switch. No scraped source should sit on the deadline-critical path without explicit permission and a cached fallback.

Acceptance:

- source passes a multi-gameweek coverage and latency trial;
- historical records have point-in-time timestamps;
- ablation shows incremental value after cost and missingness;
- loss of the source degrades gracefully.

#### XP-09: Uncertainty, champion/challenger, and serving

Purpose: make model promotion reversible and understandable.

- Add quantile, bootstrap, or conformal uncertainty experiments and test empirical coverage.
- Persist champion and challenger artifacts with model/data/code manifests.
- Run challengers in shadow for at least six completed gameweeks and across multiple deadline cycles.
- Define promotion and rollback approvals.
- Publish a versioned model card, data card, change log, and public scorecard.
- Move production artifacts from mutable repository files to versioned object storage/API serving with a manifest pointer.
- Monitor freshness, coverage, feature missingness/drift, output drift, calibration, latency, and fallback rate.

Acceptance:

- nominal intervals meet their stated coverage within agreed tolerance overall and by position;
- the challenger wins the pre-registered forward test and has no unresolved safety regression;
- rollback is exercised successfully;
- the website always exposes source, version, target, generated time, and degraded/fallback state.

## Recommended first work queue

Do these in order:

- [ ] XP-01 — workflow/source resilience and negative-value policy
- [ ] DBT-01 — season-safe snapshots and blocking data gates
- [ ] DBT-02 — point-in-time multi-season training mart
- [ ] XP-02 — shared feature builder and leakage tests
- [ ] XP-03 — forward evaluation and public baseline scorecard
- [ ] XP-04 — corrected pooled direct-points challengers
- [ ] XP-05 — expected-minutes model
- [ ] XP-06 — OpenFPL reproduction
- [ ] XP-07 — component/hybrid model
- [ ] XP-08 — approved external data experiments
- [ ] XP-09 — uncertainty and champion/challenger promotion

No position-specific or external-data model should be promoted before XP-03 exists. Otherwise the project can create a more sophisticated model without knowing whether it is more accurate.

## Production promotion gate

A candidate is production-grade only when all of the following are true:

- [ ] data is multi-season, season-safe, point-in-time, and reproducible;
- [ ] training and serving feature parity is tested;
- [ ] all selectable players and non-appearances are evaluated;
- [ ] rolling-origin evaluation beats pre-registered baselines with uncertainty reported;
- [ ] position, minutes, return, cold-start, and DGW slices have no unexplained material regression;
- [ ] expected-minutes probabilities are calibrated;
- [ ] uncertainty statements achieve measured coverage;
- [ ] six or more completed current-season gameweeks pass in shadow;
- [ ] 14 or more consecutive operational runs pass across at least two deadlines;
- [ ] model artifact, data snapshot, features, code SHA, and predictions are reproducible;
- [ ] complete-player, fixture, schema, freshness, and drift checks pass before publication;
- [ ] fallback, alerting, rollback, and manual runbook have been exercised;
- [ ] the site labels the source and limitations truthfully;
- [ ] data-source terms and redistribution rights are documented.

If evidence is inconclusive, the baseline remains champion. “The workflow ran” is not evidence that the model is ready.

## Things deliberately not recommended

- Do not merge PR #3 or #8 as written.
- Do not resurrect the large PR #4/#5 implementations as monoliths.
- Do not train on current live metadata copied backward onto historical matches.
- Do not backtest with data fetched now and assume it existed at an earlier deadline.
- Do not turn missing upstream data into zero.
- Do not drop non-appearances to improve headline accuracy.
- Do not multiply one prediction by fixture count for a DGW.
- Do not grid-search during deadline-critical inference.
- Do not add CatBoost, neural networks, news NLP, or hundreds of features before simple baselines and evaluation are trustworthy.
- Do not make Understat, FBref, unlicensed odds, or scraped team news a critical dependency without explicit rights and reliability evidence.
- Do not describe the `ep_next` baseline as a proprietary AI model.

## Research and engineering references

- [OpenFPL arXiv paper](https://arxiv.org/html/2508.09992v1)
- [OpenFPL reference repository](https://github.com/daniegr/OpenFPL)
- [Vaastav FPL historical dataset](https://github.com/vaastav/Fantasy-Premier-League)
- [AIrsenal player-points notes](https://github.com/alan-turing-institute/AIrsenal/blob/main/NOTES.md#player-points-predictions)
- [FPL Review Massive Data Model](https://docs.fplreview.com/the-model/projections/massive-data-model/)
- [FPL Review expected minutes](https://docs.fplreview.com/the-model/projections/xmins/)
- [FPL Review expected value](https://docs.fplreview.com/the-model/projections/expected-value/)
- [Deep Learning and Transfer Learning Architectures for EPL Player Performance Forecasting](https://arxiv.org/html/2405.02412)
- [Multi-stream Data Analytics for Enhanced Performance Prediction in Fantasy Football](https://arxiv.org/abs/1912.07441)
- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [XGBoost prediction intervals example](https://xgboost.readthedocs.io/en/latest/python/examples/prediction_intervals.html)
- [Google Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml/)
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)

## Decision record

This roadmap replaces the unsourced accuracy targets and bundled implementation plan in PR #3. The safe path is a sequence of small, testable changes from current `main`, with the official FPL baseline retained until a candidate proves both accuracy and operational reliability.
