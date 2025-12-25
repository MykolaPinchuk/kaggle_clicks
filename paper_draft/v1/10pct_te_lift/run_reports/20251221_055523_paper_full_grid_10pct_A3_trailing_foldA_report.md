# Run Report: `paper_full_grid_10pct_A3_trailing_foldA`

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
- Time aggregation: enabled (alpha=1.0, beta=10.0)
  - Entities: device_ip, device_id, app_id, site_id
  - Trailing windows (hours): 1, 6, 24, 48, 168
  - Gap (hours): 0

## Model
- XGBoost (hist) with learning_rate=0.05, max_depth=6, n_estimators=800, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=10.0
- Early stopping on validation with patience 50, best_iteration=476

## Metrics
| Split | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| train | 0.7582 | 0.3929 |
| val | 0.7536 | 0.3590 |
| test | 0.7483 | 0.3787 |

## Prediction Exports
- preds_val.parquet
- preds_test.parquet

## Top Features (gain)
| Rank | Feature | Gain share |
| ---: | --- | ---: |
| 1 | `site_id__ctr_w168h` | 0.2407 |
| 2 | `app_id__ctr_w168h` | 0.0878 |
| 3 | `site_id__ctr_w48h` | 0.0812 |
| 4 | `site_id__ctr_w24h` | 0.0660 |
| 5 | `site_id__te` | 0.0482 |
| 6 | `C14__te` | 0.0456 |
| 7 | `app_id__ctr_w48h` | 0.0382 |
| 8 | `site_id__ctr_w6h` | 0.0360 |
| 9 | `app_id__ctr_w24h` | 0.0247 |
| 10 | `device_id__ctr_w168h` | 0.0186 |
| 11 | `app_id__te` | 0.0182 |
| 12 | `site_id__ctr_w1h` | 0.0138 |
| 13 | `app_id__ctr_w6h` | 0.0123 |
| 14 | `hour_of_day` | 0.0111 |
| 15 | `device_ip__ctr_w1h` | 0.0110 |
