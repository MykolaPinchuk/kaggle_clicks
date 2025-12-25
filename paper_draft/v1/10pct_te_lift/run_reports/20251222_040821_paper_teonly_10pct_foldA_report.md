# Run Report: `paper_teonly_10pct_foldA`

## Data
- Sample fraction: 10% (deterministic hash on `id`)
- Rows: 4,042,711
- Split mode: rolling tail (fold A)

| Split | Rows | Start hour | End hour |
| --- | ---: | --- | --- |
| train | 3,238,511 | 2014-10-21 00:00 | 2014-10-28 23:00 |
| val | 382,897 | 2014-10-29 00:00 | 2014-10-29 23:00 |
| test | 421,303 | 2014-10-30 00:00 | 2014-10-30 23:00 |

## Features
- Numeric base features: 2 (`hour_of_day`, `prior_ctr`)
- Target-encoded features: 42
- Time aggregation: disabled

## Model
- XGBoost (hist) with learning_rate=0.05, max_depth=6, n_estimators=800, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=10.0
- Early stopping on validation with patience 50, best_iteration=523

## Metrics
| Split | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| train | 0.7574 | 0.3925 |
| val | 0.7494 | 0.3553 |
| test | 0.7417 | 0.3693 |

## Prediction Exports
- preds_val.parquet
- preds_test.parquet

## Top Features (gain)
| Rank | Feature | Gain share |
| ---: | --- | ---: |
| 1 | `site_id__te` | 0.3310 |
| 2 | `app_id__te` | 0.1140 |
| 3 | `C14__te` | 0.1099 |
| 4 | `site_domain__te` | 0.0535 |
| 5 | `C17__te` | 0.0273 |
| 6 | `hour_of_day` | 0.0219 |
| 7 | `site_id__hist_imps` | 0.0203 |
| 8 | `device_model__te` | 0.0180 |
| 9 | `device_id__hist_imps` | 0.0175 |
| 10 | `app_id__hist_imps` | 0.0173 |
| 11 | `device_id__te` | 0.0134 |
| 12 | `app_domain__te` | 0.0131 |
| 13 | `app_domain__hist_imps` | 0.0123 |
| 14 | `C21__te` | 0.0121 |
| 15 | `site_domain__hist_imps` | 0.0115 |
