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
from kaggle_clicks.time_agg import TimeAggConfig, add_time_agg_features
from kaggle_clicks.time_utils import add_time_columns, assert_strict_oot, make_oot_splits_by_day


def _fmt_ts(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return "NA"
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")


def _split_summary(df: pd.DataFrame, splits: dict[str, pd.Index]) -> list[tuple[str, int, str, str]]:
    rows: list[tuple[str, int, str, str]] = []
    for name in ("train", "val", "test"):
        idx = splits[name]
        subset = df.loc[idx, "hour_dt"]
        rows.append((name, int(len(idx)), _fmt_ts(subset.min()), _fmt_ts(subset.max())))
    return rows


def _write_report(
    path: str,
    run_tag: str,
    args: argparse.Namespace,
    df: pd.DataFrame,
    splits: dict[str, pd.Index],
    feature_cols: list[str],
    te_cols: list[str],
    metrics: dict[str, dict[str, float]],
    feature_importance: list[tuple[str, float]],
    time_agg_cols: list[str],
    best_iteration: int,
) -> None:
    lines: list[str] = []
    lines.append(f"# Run Report: `{run_tag}`")
    lines.append("")
    lines.append("## Data")
    lines.append(f"- Sample fraction: {args.sample_pct}% (deterministic hash on `id`)")
    lines.append(f"- Rows: {len(df):,}")
    lines.append("")
    lines.append("| Split | Rows | Start hour | End hour |")
    lines.append("| --- | ---: | --- | --- |")
    for name, count, start, end in _split_summary(df, splits):
        lines.append(f"| {name} | {count:,} | {start} | {end} |")

    lines.append("")
    lines.append("## Features")
    lines.append(f"- Numeric base features: 2 (`hour_of_day`, `prior_ctr`)")
    lines.append(f"- Target-encoded features: {len(te_cols)}")
    if time_agg_cols:
        lines.append(
            f"- Time aggregation: {len(args.time_agg_entities)} entities × {len(args.time_agg_windows)} windows "
            f"(alpha={args.time_agg_alpha}, beta={args.time_agg_beta})"
        )
        lines.append(f"  - Entities: {', '.join(args.time_agg_entities)}")
        lines.append(f"  - Windows (hours): {', '.join(map(str, args.time_agg_windows))}")
    else:
        lines.append("- Time aggregation: disabled")

    lines.append("")
    lines.append("## Model")
    lines.append(
        f"- XGBoost (hist) with learning_rate={args.learning_rate}, max_depth={args.max_depth}, "
        f"n_estimators={args.n_estimators}, subsample={args.subsample}, "
        f"colsample_bytree={args.colsample_bytree}, reg_lambda={args.reg_lambda}, "
        f"min_child_weight={args.min_child_weight}"
    )
    lines.append(f"- Early stopping on validation with patience 50, best_iteration={best_iteration}")

    lines.append("")
    lines.append("## Metrics")
    lines.append("| Split | ROC-AUC | PR-AUC |")
    lines.append("| --- | ---: | ---: |")
    for split in ("train", "val", "test"):
        m = metrics.get(split)
        if m:
            lines.append(f"| {split} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} |")

    if feature_importance:
        lines.append("")
        lines.append("## Top Features (gain)")
        lines.append("| Rank | Feature | Gain share |")
        lines.append("| ---: | --- | ---: |")
        for rank, (feat, gain) in enumerate(feature_importance[:15], start=1):
            lines.append(f"| {rank} | `{feat}` | {gain:.4f} |")

    content = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


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
    ap.add_argument(
        "--time-agg-entities",
        nargs="*",
        default=["device_ip", "device_id", "app_id", "site_id"],
        help="Entity columns for time aggregation features (empty list to disable).",
    )
    ap.add_argument(
        "--time-agg-windows",
        nargs="*",
        type=int,
        default=[1, 6, 24],
        help="Window sizes in hours for time aggregation features.",
    )
    ap.add_argument("--time-agg-alpha", type=float, default=1.0)
    ap.add_argument("--time-agg-beta", type=float, default=10.0)
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

    time_agg_cols: list[str] = []
    if args.time_agg_entities:
        fe, time_agg_cols = add_time_agg_features(
            fe,
            cfg=TimeAggConfig(
                entities=tuple(args.time_agg_entities),
                windows=tuple(args.time_agg_windows),
                alpha=args.time_agg_alpha,
                beta=args.time_agg_beta,
            ),
        )

    te_cols = [f"{c}__te" for c in cat_cols] + [f"{c}__hist_imps" for c in cat_cols]
    base_cols = ["hour_of_day", "prior_ctr"]
    feature_cols = base_cols + te_cols + time_agg_cols

    X_train = fe.loc[splits["train"], feature_cols].astype(np.float32)
    y_train = fe.loc[splits["train"], label_col].to_numpy(dtype=np.int8, copy=False)
    X_val = fe.loc[splits["val"], feature_cols].astype(np.float32)
    y_val = fe.loc[splits["val"], label_col].to_numpy(dtype=np.int8, copy=False)
    X_test = fe.loc[splits["test"], feature_cols].astype(np.float32)
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

    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]
    train_metrics = compute_binary_metrics(y_train, train_prob)
    val_metrics = compute_binary_metrics(y_val, val_prob)
    test_metrics = compute_binary_metrics(y_test, test_prob)

    config = vars(args) | {
        "cat_cols": cat_cols,
        "n_rows": int(len(df)),
        "feature_cols": feature_cols,
    }

    metrics_by_split = {
        "train": {"roc_auc": train_metrics.roc_auc, "pr_auc": train_metrics.pr_auc},
        "val": {"roc_auc": val_metrics.roc_auc, "pr_auc": val_metrics.pr_auc},
        "test": {"roc_auc": test_metrics.roc_auc, "pr_auc": test_metrics.pr_auc},
    }
    best_iteration = int(getattr(model, "best_iteration", -1))
    metrics_payload = {"splits": metrics_by_split, "best_iteration": best_iteration}

    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, sort_keys=True))

    booster = model.get_booster()
    booster.save_model(str(run_dir / "model.json"))

    raw_importance = booster.get_score(importance_type="gain")
    feature_importance: list[tuple[str, float]] = []
    if raw_importance:
        total_gain = sum(raw_importance.values())
        name_map = (
            {f"f{i}": name for i, name in enumerate(booster.feature_names)}
            if booster.feature_names
            else {}
        )
        feature_importance = sorted(
            [(name_map.get(key, key), gain / total_gain) for key, gain in raw_importance.items()],
            key=lambda x: x[1],
            reverse=True,
        )
    report_path = run_dir / "report.md"
    _write_report(
        str(report_path),
        args.run_tag,
        args,
        df,
        splits,
        feature_cols,
        te_cols,
        metrics_by_split,
        feature_importance,
        time_agg_cols,
        best_iteration,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
