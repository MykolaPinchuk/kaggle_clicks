# kaggle_clicks (Avazu CTR)

This repo is set up to run time-based (OOT) CTR experiments on the Kaggle Avazu CTR Prediction dataset, with a focus on entity-level time aggregation features.

- Project plan: `docs/PLAN.md`
- Time-aggregation experiment plan: `docs/TIME_AGG_EXPERIMENTS.md`
- Data download: `docs/DATA.md`
- EDA report: `docs/EDA_REPORT.md`

## Quickstart (once data is downloaded)

- Put `train.csv` at `data/raw/train.csv`
- Run 1% sample baseline (TE + time-agg by default): `python -m kaggle_clicks.run_baseline_te --sample-pct 1`
- Each run creates a timestamped folder under `runs/` containing `config.json`, `metrics.json`, the trained `model.json`, and a human-readable `report.md`.
