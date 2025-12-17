# kaggle_clicks (Avazu CTR)

This repo is set up to run time-based (OOT) CTR experiments on the Kaggle Avazu CTR Prediction dataset, with a focus on entity-level time aggregation features.

- Project plan: `docs/PLAN.md`
- Data download: `docs/DATA.md`

## Quickstart (once data is downloaded)

- Put `train.csv` at `data/raw/train.csv`
- Run 1% sample + TE baseline: `python -m kaggle_clicks.run_baseline_te --sample-pct 1`
