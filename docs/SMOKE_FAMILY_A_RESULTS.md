# Smoke Test Results: Family A Variants (2025-12-18)

This document summarizes the **end-to-end smoke tests** for the newly implemented Family A time-aggregation variants (gap windows, bucketized windows, calendar windows, event-count windows). These runs are intended to validate that each codepath works and produces sane outputs, not to select the best variant.

## Where the detailed reports are

Each run writes a detailed per-run report at:

- `runs/<timestamp>_<run-tag>/report.md`

Those reports include: data split ranges/sizes, feature configuration, model params, train/val/test ROC-AUC + PR-AUC, and top feature importances.

## Data + protocol (common across runs)

- Dataset: Avazu `train.csv`
- Sample: deterministic 1% sample cached at `data/interim/train_sample.parquet` (403,698 rows)
- Splits: strict OOT by day (train=all but last 2 days, val=day before last, test=last day)
- Runner: `python -m kaggle_clicks.run_baseline_te`

## Summary table

Note: the runs below used slightly different `n_estimators` / `learning_rate` (optimized for speed), so **compare metrics only as a smoke-test sanity check**.

| Run dir | Variant focus | Entities | Trailing windows | Gap | Buckets | Calendar | Event windows | Val ROC-AUC | Val PR-AUC | Test ROC-AUC | Test PR-AUC |
| --- | --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| `runs/20251218_025623_smoke_familyA_all` | all variants enabled | `app_id` | `24h` | `1h` | `1,6,24,168` | yes | `50` | 0.7420 | 0.3489 | 0.7358 | 0.3548 |
| `runs/20251218_025715_smoke_gap_only_ip` | gap windows | `device_ip` | `24h` | `6h` | - | - | - | 0.7416 | 0.3476 | 0.7318 | 0.3578 |
| `runs/20251218_025733_smoke_bucket_only_ip` | bucketized windows | `device_ip` | - | - | `1,6,24,168` | - | - | 0.7400 | 0.3488 | 0.7310 | 0.3551 |
| `runs/20251218_025754_smoke_calendar_only_ip` | calendar windows | `device_ip` | - | - | - | yes | - | 0.7387 | 0.3476 | 0.7287 | 0.3521 |
| `runs/20251218_025808_smoke_event_only_ip` | event-count windows | `device_ip` | - | - | - | - | `50` | 0.7402 | 0.3465 | 0.7309 | 0.3530 |

## What each variant means (implementation-level)

- **Trailing windows**: per-entity rolling counts/CTR in `[H-w, H)` (no same-hour leakage; hour `H` only uses `<H`).
- **Gap windows** (`--time-agg-gap-hours g`): uses `[H-w-g, H-g)` by subtracting a `g`-hour trailing window from a `(w+g)`-hour trailing window.
- **Bucketized windows** (`--time-agg-bucket-edges ...`): disjoint windows like `0–1h`, `1–6h`, `6–24h`, `24–168h`, constructed from cumulative trailing windows via differences.
- **Calendar windows** (`--time-agg-calendar`): per-entity “today so far” and “yesterday” aggregates aligned to day boundaries (still no same-hour leakage).
- **Event-count windows** (`--time-agg-event-windows N...`): “last N impressions” per entity implemented as a rolling store of **hour blocks**; when trimming to exactly `N`, the oldest block is partially trimmed proportionally.

## How to inspect or reproduce

- Open a run report (example): `runs/20251218_025623_smoke_familyA_all/report.md`
- Re-run a smoke test quickly (example):
  - `python -m kaggle_clicks.run_baseline_te --sample-pct 1 --sample-parquet data/interim/train_sample.parquet --run-tag smoke_gap --time-agg-entities device_ip --time-agg-windows 24 --time-agg-gap-hours 6 --n-estimators 150 --learning-rate 0.15`

