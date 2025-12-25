# Run Report: `paper_full_grid_10pct_A3_event50_foldB`

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
- Time aggregation: enabled (alpha=1.0, beta=10.0)
  - Entities: device_ip, device_id, app_id, site_id
  - Trailing windows (hours): 1, 6, 24, 48, 168
  - Gap (hours): 0
  - Event windows (last N imps): 50

## Model
- XGBoost (hist) with learning_rate=0.05, max_depth=6, n_estimators=800, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=10.0
- Early stopping on validation with patience 50, best_iteration=418

## Metrics
| Split | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| train | 0.7543 | 0.3937 |
| val | 0.7643 | 0.3695 |
| test | 0.7524 | 0.3574 |

## Prediction Exports
- preds_val.parquet
- preds_test.parquet

## Top Features (gain)
| Rank | Feature | Gain share |
| ---: | --- | ---: |
| 1 | `site_id__ctr_last50imps` | 0.2911 |
| 2 | `site_id__ctr_w168h` | 0.0978 |
| 3 | `app_id__ctr_w168h` | 0.0845 |
| 4 | `app_id__ctr_last50imps` | 0.0451 |
| 5 | `C14__te` | 0.0439 |
| 6 | `site_id__ctr_w24h` | 0.0373 |
| 7 | `app_id__ctr_w48h` | 0.0322 |
| 8 | `site_id__ctr_w48h` | 0.0221 |
| 9 | `device_id__ctr_w168h` | 0.0165 |
| 10 | `site_id__te` | 0.0130 |
| 11 | `site_id__ctr_w6h` | 0.0123 |
| 12 | `C17__te` | 0.0103 |
| 13 | `device_ip__ctr_w1h` | 0.0094 |
| 14 | `device_model__te` | 0.0088 |
| 15 | `app_id__imps_log_w1h` | 0.0079 |
