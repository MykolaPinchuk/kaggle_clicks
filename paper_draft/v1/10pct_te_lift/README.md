# 10% TE-only Lift vs Time Aggregation (Fold A/B)

This folder contains the “big lift” comparison against the TE-only baseline, using the same 10% sample and rolling-tail Fold A/B protocol as the paper grid.

## Main artifact

- `te_vs_timeagg_summary.md` (test metrics + paired inference)

## Data files

- `inference_test_te_vs_timeagg.csv` (paired DeLong for ROC-AUC + paired block bootstrap for PR-AUC)
- `metrics_selected_runs.csv` (metrics extracted from the referenced run dirs)

## Run reports (copied)

See `run_reports/` for `report.md`, `metrics.json`, and `config.json` copies for:
- TE-only Fold A/B
- A3 trailing Fold A/B
- A3 event50 Fold A/B

## Notes

- Large prediction exports (`preds_*.parquet`) are not copied here; they remain in the original `runs/...` directories referenced inside `te_vs_timeagg_summary.md`.

