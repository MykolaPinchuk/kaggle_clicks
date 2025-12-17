# Project Plan: Avazu CTR (OOT) + Time Aggregation Features

## 1) Goal

Build a binary classifier to predict `click` on the Avazu CTR dataset with **strict out-of-time (OOT)** evaluation and a feature pipeline that supports rapid iteration on **entity-level time aggregation features**.

Primary metric: **ROC-AUC**  
Secondary metric: **PR-AUC** (and optionally logloss as a sanity check)

Core hypothesis: **recent historical behavior by entity** (e.g., `device_ip`, `device_id`, `app_id`, `site_id`) contains strong predictive signal when encoded as trailing-window statistics.

## 2) Constraints / Principles

- Compute: single PC; use all vCPUs except one.
- Iteration: start with a **very small sample (~1%)** to validate assumptions and avoid accidental leakage; scale later.
- Modeling: **XGBoost** to start (keep feature representation compatible with large-scale sparse/high-cardinality categoricals).
- Evaluation: `train/val/test` are **strictly OOT** relative to each other (time order). If we later add more validation sets, they only need to be OOT w.r.t. train.
- Online simulation: features are computed **online**, i.e., during test we keep updating features using outcomes observed earlier in the test period.

## 3) Dataset Notes

Competition: https://www.kaggle.com/competitions/avazu-ctr-prediction

Expected fields (train): `id`, `click`, `hour` plus categorical fields like `device_id`, `device_ip`, `app_id`, `site_id`, etc.

Important: `hour` is at **hour resolution** (format like `YYMMDDHH`), so the smallest *true* refresh interval supported by timestamp is **1 hour**. We can still implement the online feature store as “refresh every 1 minute” conceptually, but the event time only changes hourly; within an hour we’ll update features in **row chunks** to mimic frequent refresh without inventing timestamps.

## 4) Repository Layout (planned)

- `data/raw/`: Kaggle zip(s) and extracted `train.csv`
- `data/interim/`: sampled Parquet, aggregated tables
- `data/processed/`: final feature tables (if needed)
- `src/`: code (ingest, sampling, splits, features, training)
- `runs/`: metrics + configs per run
- `models/`: trained model artifacts
- `docs/`: notes and results summaries

## 5) Split Strategy (v1)

We will create **three non-overlapping, chronological** splits:

- **Train**: earliest period
- **Val**: later period (used for early stopping and model selection)
- **Test**: latest period (final offline estimate)

Implementation detail:
- Determine dataset’s min/max day from `hour`.
- Choose cut points by day:
  - `test_days = last 1 day` (or last N hours if dataset slice is smaller)
  - `val_days = the day before test`
  - `train_days = all earlier days`

This ensures: `train < val < test` strictly by time.

## 6) Sampling Strategy (v0: 1% for logic validation)

We want sampling that preserves time structure and is deterministic:

- Parse `hour` to an integer hour-index.
- Keep ~1% of rows via stable hash of `id` (or row index if no `id`), e.g. `hash(id) % 100 == 0`.
- After sampling, recompute the OOT split boundaries on the sampled data (boundaries still based on day/hour).

Rationale:
- Avoid random shuffling that can hide leakage.
- Keep distribution across time reasonably intact.

## 7) Baseline Modeling (no time-agg yet)

### Feature representation (v0): Target Encoding (TE)

We will not use OHE/hashed OHE. For categorical columns we use **time-aware, smoothed target encoding** (TE), which is both compact and naturally aligned with online feature computation.

For each categorical column `c`, and for each event at hour bucket `H`, define:

- `imps_c(H)`: number of impressions with category value `c=v` observed in hours `< H`
- `clicks_c(H)`: number of clicks for `c=v` observed in hours `< H`
- `te_c(H)`: smoothed CTR estimate for `c=v` at time `H`:
  - `prior(H)` = global CTR estimated from hours `< H`
  - `te_c(H) = (clicks_c(H) + m * prior(H)) / (imps_c(H) + m)`

Where `m` is a regularization strength (larger = more shrinkage toward the prior).

Time features (numeric):
- hour-of-day (0–23)
- day index (relative)

### Model

- XGBoost with histogram tree method (CPU), early stopping on `val`.
- Track: ROC-AUC, PR-AUC on val and test.

This baseline is primarily to validate:
- data loading and split correctness,
- end-to-end training/eval,
- experiment logging format.

## 8) Time Aggregation Features (core work)

### Entities (initial)

Start with exactly these four:

- `device_ip`
- `device_id`
- `app_id`
- `site_id`

### Windows (initial)

Expressed in hours (because `hour` granularity is hourly):

- `1h`
- `6h`
- `24h`

### Online feature definition (no leakage)

For each entity `e` and hour bucket `H`, define:

- `imps_e_w(H)`: number of impressions for `e` in the trailing window `[H-w, H)`  
- `clicks_e_w(H)`: number of clicks for `e` in `[H-w, H)`  
- `ctr_e_w(H)`: smoothed CTR, e.g. `(clicks + α) / (imps + α + β)`
- `recency_e(H)`: hours since `e` was last seen (or last clicked as an option)

Key anti-leakage rule:
- For events occurring at hour `H`, features must depend only on **hours < H** (strictly earlier).

### Efficient computation approach

To scale, compute per-entity-per-hour aggregates first:

1. Build table: `(entity, hour) -> imps, clicks`
2. For each entity, compute rolling sums over hour-index for each window:
   - `imps_1h`, `imps_6h`, `imps_24h`, same for clicks
   - shift by 1 hour to exclude the current hour
3. Join back to row-level data on `(entity, hour)`

Library choice:
- EDA in `pandas`
- Aggregations in `polars` or `duckdb` (whichever is faster/cleaner after a quick spike)

### Online simulation during validation/test

Because we are doing “online” features, the feature tables for `val` and `test` are computed using:

- all history from train, plus
- earlier hours from within `val`/`test` themselves (still shifted by 1 hour).

Practically, this works naturally with the `(entity, hour)` rolling method as long as the rolling computation includes the entire timeline and is shifted.

## 9) Experiment Tracking (minimal, for speed)

Each run writes to `runs/<timestamp>_<tag>/`:

- `config.json` (sample fraction, split cutoffs, entities, windows, model params)
- `metrics.json` (val/test ROC-AUC, PR-AUC, runtime)
- optionally: feature importance dump from XGB
- `report.md`: human-readable summary (data coverage, feature config, metrics table, top features)

## 10) Phased Milestones

### Phase A: sanity + plumbing (1% sample)

1. Download/extract data to `data/raw/`
2. Parse `hour` and build 1% deterministic sample to `data/interim/`
3. Implement strict OOT `train/val/test` split
4. Train XGB baseline (hashed categoricals + time features)
5. Confirm metrics compute + run logging

Exit criteria:
- Reproducible run in a few minutes
- OOT splits verified (min/max hour by split checks)

### Phase B: time-agg features on sample

1. Implement `(entity, hour)` aggregation + rolling windows (shifted)
2. Add join-back to row level
3. Add features for the four entities and three windows
4. Run ablations:
   - baseline
   - + each entity alone
   - + all entities

Exit criteria:
- Clear lift from at least one entity/window configuration
- No evidence of leakage (see checks below)

### Phase C: scale-up

1. Move from 1% to larger fractions, then full dataset
2. Optimize IO/CPU with Parquet scans + efficient groupby/rolling
3. Stabilize memory (streaming/partitioning by day if needed)

Exit criteria:
- Full-data runs complete within acceptable wall time on the PC

## 11) Leakage / Correctness Checks (must-have)

- Split monotonicity: `max(hour_train) < min(hour_val) < min(hour_test)`
- Feature temporal constraint: for any row at hour `H`, verify `*_w` features only use `< H`
- Spot-check on a tiny slice with a slow reference implementation:
  - compute trailing counts by iterating hours and updating a dict
  - compare against vectorized/rolling output for a few entities

## 12) Open Questions (for approval)

1. Dataset location: do we assume Kaggle API credentials will be available on this machine, or should we plan for manual download?
2. Smoothing defaults: acceptable starting values for `(α, β)` in smoothed CTR (e.g., `α=1, β=10`)?
3. Minimum refresh interval: given hour-resolution timestamps, confirm we treat “1-minute refresh” as chunked updates within each hour (no extra timestamps).

## Approval Checklist

If you approve this plan, next step is to scaffold `src/` with the Phase A scripts and wire a single command to run a 1% baseline + log metrics.
