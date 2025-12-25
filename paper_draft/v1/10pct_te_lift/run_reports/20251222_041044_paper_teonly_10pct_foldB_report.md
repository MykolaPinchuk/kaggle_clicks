# Run Report: `paper_teonly_10pct_foldB`

## Data
- Sample fraction: 10% (deterministic hash on `id`)
- Rows: 4,042,711
- Split mode: rolling tail (fold B)

| Split | Rows | Start hour | End hour |
| --- | ---: | --- | --- |
| train | 2,708,333 | 2014-10-21 00:00 | 2014-10-27 23:00 |
| val | 530,178 | 2014-10-28 00:00 | 2014-10-28 23:00 |
| test | 382,897 | 2014-10-29 00:00 | 2014-10-29 23:00 |

## Features
- Numeric base features: 2 (`hour_of_day`, `prior_ctr`)
- Target-encoded features: 42
- Time aggregation: disabled

## Model
- XGBoost (hist) with learning_rate=0.05, max_depth=6, n_estimators=800, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=10.0
- Early stopping on validation with patience 50, best_iteration=348

## Metrics
| Split | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| train | 0.7512 | 0.3899 |
| val | 0.7607 | 0.3656 |
| test | 0.7438 | 0.3480 |

## Prediction Exports
- preds_val.parquet
- preds_test.parquet

## Top Features (gain)
| Rank | Feature | Gain share |
| ---: | --- | ---: |
| 1 | `site_id__te` | 0.3091 |
| 2 | `C14__te` | 0.1295 |
| 3 | `app_id__te` | 0.1032 |
| 4 | `site_domain__te` | 0.0580 |
| 5 | `C17__te` | 0.0278 |
| 6 | `hour_of_day` | 0.0235 |
| 7 | `site_id__hist_imps` | 0.0215 |
| 8 | `device_model__te` | 0.0194 |
| 9 | `app_id__hist_imps` | 0.0185 |
| 10 | `device_id__hist_imps` | 0.0180 |
| 11 | `device_id__te` | 0.0144 |
| 12 | `app_domain__te` | 0.0136 |
| 13 | `site_domain__hist_imps` | 0.0129 |
| 14 | `prior_ctr` | 0.0126 |
| 15 | `C21__te` | 0.0114 |
