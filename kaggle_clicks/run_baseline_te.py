from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

from kaggle_clicks.metrics import compute_binary_metrics
from kaggle_clicks.paths import get_paths
from kaggle_clicks.sampling import deterministic_pct_mask
from kaggle_clicks.te import TEConfig, add_time_target_encoding
from kaggle_clicks.time_utils import add_time_columns, assert_strict_oot, make_oot_splits_by_day


def _load_or_make_sample(train_csv: str, sample_pct: int, out_parquet: str) -> pd.DataFrame:
    p = os.path.abspath(out_parquet)
    if os.path.exists(p):
        return pd.read_parquet(p)

    chunks = []
    for chunk in pd.read_csv(train_csv, chunksize=500_000):
        if "id" not in chunk.columns:
            raise ValueError("Expected 'id' column in train.csv for deterministic sampling.")
        mask = deterministic_pct_mask(chunk["id"], pct=sample_pct)
        if mask.any():
            chunks.append(chunk.loc[mask])
    if not chunks:
        raise RuntimeError("Sampling produced 0 rows; increase sample_pct or check data.")
    df = pd.concat(chunks, axis=0, ignore_index=True)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_parquet(p, index=False)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default=str(get_paths().data_raw / "train.csv"))
    ap.add_argument("--sample-pct", type=int, default=1)
    ap.add_argument("--sample-parquet", default=str(get_paths().data_interim / "train_sample.parquet"))
    ap.add_argument("--m", type=float, default=100.0, help="TE smoothing strength (larger = more shrinkage).")
    ap.add_argument("--n-estimators", type=int, default=2000)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--reg-lambda", type=float, default=5.0)
    ap.add_argument("--min-child-weight", type=float, default=10.0)
    ap.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--run-tag", default="baseline_te")
    args = ap.parse_args()

    paths = get_paths()
    run_dir = paths.runs / f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{args.run_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.train_csv):
        raise SystemExit(f"Missing data file: {args.train_csv} (place it at data/raw/train.csv)")

    raw = _load_or_make_sample(args.train_csv, args.sample_pct, args.sample_parquet)
    df = add_time_columns(raw, hour_col="hour").sort_values("hour_dt").reset_index(drop=True)

    splits = make_oot_splits_by_day(df)
    assert_strict_oot(df, splits)

    label_col = "click"
    drop_cols = {"id", "hour", "hour_dt", "day", "hour_of_day", label_col}
    cat_cols = [c for c in df.columns if c not in drop_cols]

    fe = add_time_target_encoding(df, cat_cols=cat_cols, label_col=label_col, cfg=TEConfig(m=args.m))

    feature_cols = (
        ["hour_of_day", "prior_ctr"]
        + [f"{c}__te" for c in cat_cols]
        + [f"{c}__hist_imps" for c in cat_cols]
    )

    X_train = fe.loc[splits["train"], feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_train = fe.loc[splits["train"], label_col].to_numpy(dtype=np.int8, copy=False)
    X_val = fe.loc[splits["val"], feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_val = fe.loc[splits["val"], label_col].to_numpy(dtype=np.int8, copy=False)
    X_test = fe.loc[splits["test"], feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_test = fe.loc[splits["test"], label_col].to_numpy(dtype=np.int8, copy=False)

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        min_child_weight=args.min_child_weight,
        objective="binary:logistic",
        tree_method="hist",
        n_jobs=args.n_jobs,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc", "aucpr"],
        verbose=False,
        early_stopping_rounds=50,
    )

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    val_metrics = compute_binary_metrics(y_val, val_prob)
    test_metrics = compute_binary_metrics(y_test, test_prob)

    config = vars(args) | {"cat_cols": cat_cols, "n_rows": int(len(df))}
    metrics = {
        "val": {"roc_auc": val_metrics.roc_auc, "pr_auc": val_metrics.pr_auc},
        "test": {"roc_auc": test_metrics.roc_auc, "pr_auc": test_metrics.pr_auc},
        "best_iteration": int(getattr(model, "best_iteration", -1)),
    }

    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))

    model.save_model(str(run_dir / "model.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
