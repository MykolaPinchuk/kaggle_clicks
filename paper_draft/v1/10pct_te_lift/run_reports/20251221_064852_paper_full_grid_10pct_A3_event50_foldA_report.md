# Run Report: `paper_full_grid_10pct_A3_event50_foldA`

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
  - Event windows (last N imps): 50

## Model
- XGBoost (hist) with learning_rate=0.05, max_depth=6, n_estimators=800, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0, min_child_weight=10.0
- Early stopping on validation with patience 50, best_iteration=781

## Metrics
| Split | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| train | 0.7622 | 0.3993 |
| val | 0.7546 | 0.3595 |
| test | 0.7486 | 0.3784 |

## Prediction Exports
- preds_val.parquet
- preds_test.parquet

## Top Features (gain)
| Rank | Feature | Gain share |
| ---: | --- | ---: |
| 1 | `site_id__ctr_last50imps` | 0.2594 |
| 2 | `site_id__ctr_w168h` | 0.1278 |
| 3 | `app_id__ctr_w168h` | 0.0965 |
| 4 | `app_id__ctr_last50imps` | 0.0439 |
| 5 | `C14__te` | 0.0416 |
| 6 | `site_id__ctr_w24h` | 0.0401 |
| 7 | `app_id__ctr_w48h` | 0.0342 |
| 8 | `device_id__ctr_w168h` | 0.0167 |
| 9 | `site_id__ctr_w48h` | 0.0158 |
| 10 | `site_id__te` | 0.0108 |
| 11 | `device_id__ctr_last50imps` | 0.0107 |
| 12 | `device_ip__ctr_w1h` | 0.0087 |
| 13 | `device_model__te` | 0.0080 |
| 14 | `C17__te` | 0.0076 |
| 15 | `device_ip__ctr_last50imps` | 0.0071 |
