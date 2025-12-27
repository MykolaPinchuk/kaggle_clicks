# Problem Context (v1)

## What data is this?
This bundle summarizes experiments on the **Kaggle Avazu CTR Prediction** dataset (`train.csv`), where each row is an ad impression with:
- target `click` (0/1)
- timestamp `hour` at **hour resolution**
- many high-cardinality categorical fields (e.g., `device_ip`, `device_id`, `app_id`, `site_id`, …).

## What problem are we solving?
Predict the probability of click (CTR) for each impression. Primary metric is **ROC-AUC**; secondary is **PR-AUC**.

## Why is evaluation tricky?
CTR data is strongly temporal. Naive feature engineering can leak future information.
We therefore use:
- **Strict out-of-time (OOT) splits** (train < val < test chronologically)
- **Online-safe feature simulation**: features for impressions at hour `H` may use only information from **hours `< H`** (no same-hour leakage).

We report results using a “rolling-tail” stability check with two folds:
- Fold A: train up to day D-1, validate on D-1, test on D
- Fold B: train up to day D-2, validate on D-2, test on D-1

## What methods are compared?
All runs use the same model family (XGBoost) and the same fixed entity set for history features:
`device_ip`, `device_id`, `app_id`, `site_id`.

We compare **Family A time-aggregation feature constructions**:
- **A.2 window length sets** (how many hours-back windows)
- **A.1 window shapes** (how the “recent history” window is defined):
  - trailing windows
  - gap windows (`gap1`)
  - bucketized windows (`bucket`)
  - calendar windows (`calendar`)
  - event-count windows (`event50`, “last N impressions”)

We report results both:
- with **time-aware target encoding (TE)** enabled (“TE + time-agg”), and
- with **no TE** (“time-agg only”) as a sensitivity view.

## XGBoost training protocol (exact, as used in the 10% grids)
Implementation: `kaggle_clicks/run_baseline_te.py` (the sweeps call this entrypoint).

Training/evaluation:
- Train an `xgboost.XGBClassifier` on the **train** split.
- Use **early stopping on the validation split** with `early_stopping_rounds=50`.
- Evaluation metrics during training: `eval_metric=["auc","aucpr"]`.
- Final reporting: compute ROC-AUC and PR-AUC on train/val/test from predicted probabilities.

Key hyperparameters (everything else is XGBoost default):
- `n_estimators=800`, `learning_rate=0.05`, `max_depth=6`
- `subsample=0.8`, `colsample_bytree=0.8`
- `min_child_weight=10.0`, `reg_lambda=5.0`
- `objective="binary:logistic"`, `tree_method="hist"`, `random_state=42`
- `n_jobs=1` (to avoid CPU oversubscription during sweeps)
