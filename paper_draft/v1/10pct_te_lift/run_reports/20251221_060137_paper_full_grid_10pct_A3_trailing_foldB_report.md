# Run Report: `paper_full_grid_10pct_A3_trailing_foldB`

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

## Model
- XGBoost (hist) with learning_rate=0.05, max_depth=6, n_estimators=800, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=10.0
- Early stopping on validation with patience 50, best_iteration=397

## Metrics
| Split | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| train | 0.7540 | 0.3932 |
| val | 0.7643 | 0.3683 |
| test | 0.7520 | 0.3563 |

## Prediction Exports
- preds_val.parquet
- preds_test.parquet

## Top Features (gain)
| Rank | Feature | Gain share |
| ---: | --- | ---: |
| 1 | `site_id__ctr_w168h` | 0.2067 |
| 2 | `site_id__ctr_w48h` | 0.0900 |
| 3 | `app_id__ctr_w168h` | 0.0871 |
| 4 | `site_id__ctr_w24h` | 0.0735 |
| 5 | `site_id__ctr_w6h` | 0.0474 |
| 6 | `C14__te` | 0.0470 |
| 7 | `app_id__ctr_w48h` | 0.0369 |
| 8 | `site_id__te` | 0.0301 |
| 9 | `app_id__ctr_w24h` | 0.0247 |
| 10 | `site_id__ctr_w1h` | 0.0216 |
| 11 | `C17__te` | 0.0171 |
| 12 | `device_id__ctr_w168h` | 0.0168 |
| 13 | `app_id__te` | 0.0165 |
| 14 | `app_id__ctr_w6h` | 0.0141 |
| 15 | `hour_of_day` | 0.0109 |
