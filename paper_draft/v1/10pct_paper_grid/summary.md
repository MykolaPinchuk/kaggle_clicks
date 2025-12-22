# Sweep Summary: `paper_full_grid_10pct`

- Created (UTC): `20251221_183006`
- Sample: `10%` (`data/interim/train_sample_10pct.parquet`)
- Train CSV: `/home/mykola/repos/kaggle_clicks/data/raw/train.csv`

| Run | Fold | Description | Return | Skipped | Val ROC-AUC | Test ROC-AUC | Run dir |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| A1_trailing | A | A1 trailing windows (1, 6, 24) | 0 | yes | 0.7543 | 0.7473 | `/home/mykola/repos/kaggle_clicks/runs/20251221_035159_paper_full_grid_10pct_A1_trailing_foldA` |
| A1_trailing | B | A1 trailing windows (1, 6, 24) | 0 | yes | 0.7624 | 0.7511 | `/home/mykola/repos/kaggle_clicks/runs/20251221_035729_paper_full_grid_10pct_A1_trailing_foldB` |
| A1_gap1 | A | A1 gap g=1 windows (1, 6, 24) | 0 | yes | 0.7522 | 0.7441 | `/home/mykola/repos/kaggle_clicks/runs/20251221_040218_paper_full_grid_10pct_A1_gap1_foldA` |
| A1_gap1 | B | A1 gap g=1 windows (1, 6, 24) | 0 | yes | 0.7615 | 0.7496 | `/home/mykola/repos/kaggle_clicks/runs/20251221_040732_paper_full_grid_10pct_A1_gap1_foldB` |
| A1_bucket | A | A1 bucket edges (1, 6, 24) | 0 | yes | 0.7533 | 0.7445 | `/home/mykola/repos/kaggle_clicks/runs/20251221_041251_paper_full_grid_10pct_A1_bucket_foldA` |
| A1_bucket | B | A1 bucket edges (1, 6, 24) | 0 | yes | 0.7630 | 0.7506 | `/home/mykola/repos/kaggle_clicks/runs/20251221_041803_paper_full_grid_10pct_A1_bucket_foldB` |
| A1_calendar | A | A1 calendar windows on trailing (1, 6, 24) | 0 | yes | 0.7540 | 0.7468 | `/home/mykola/repos/kaggle_clicks/runs/20251221_042302_paper_full_grid_10pct_A1_calendar_foldA` |
| A1_calendar | B | A1 calendar windows on trailing (1, 6, 24) | 0 | yes | 0.7630 | 0.7518 | `/home/mykola/repos/kaggle_clicks/runs/20251221_042913_paper_full_grid_10pct_A1_calendar_foldB` |
| A1_event50 | A | A1 event windows last50 | 0 | yes | 0.7543 | 0.7486 | `/home/mykola/repos/kaggle_clicks/runs/20251221_043522_paper_full_grid_10pct_A1_event50_foldA` |
| A1_event50 | B | A1 event windows last50 | 0 | yes | 0.7632 | 0.7522 | `/home/mykola/repos/kaggle_clicks/runs/20251221_044119_paper_full_grid_10pct_A1_event50_foldB` |
| A2_trailing | A | A2 trailing windows (1, 3, 6, 12, 24) | 0 | yes | 0.7537 | 0.7446 | `/home/mykola/repos/kaggle_clicks/runs/20251221_044636_paper_full_grid_10pct_A2_trailing_foldA` |
| A2_trailing | B | A2 trailing windows (1, 3, 6, 12, 24) | 0 | yes | 0.7622 | 0.7515 | `/home/mykola/repos/kaggle_clicks/runs/20251221_045315_paper_full_grid_10pct_A2_trailing_foldB` |
| A2_gap1 | A | A2 gap g=1 windows (1, 3, 6, 12, 24) | 0 | yes | 0.7522 | 0.7458 | `/home/mykola/repos/kaggle_clicks/runs/20251221_045917_paper_full_grid_10pct_A2_gap1_foldA` |
| A2_gap1 | B | A2 gap g=1 windows (1, 3, 6, 12, 24) | 0 | yes | 0.7610 | 0.7494 | `/home/mykola/repos/kaggle_clicks/runs/20251221_050614_paper_full_grid_10pct_A2_gap1_foldB` |
| A2_bucket | A | A2 bucket edges (1, 3, 6, 12, 24) | 0 | yes | 0.7541 | 0.7457 | `/home/mykola/repos/kaggle_clicks/runs/20251221_051247_paper_full_grid_10pct_A2_bucket_foldA` |
| A2_bucket | B | A2 bucket edges (1, 3, 6, 12, 24) | 0 | yes | 0.7622 | 0.7500 | `/home/mykola/repos/kaggle_clicks/runs/20251221_051958_paper_full_grid_10pct_A2_bucket_foldB` |
| A2_calendar | A | A2 calendar windows on trailing (1, 3, 6, 12, 24) | 0 | yes | 0.7539 | 0.7478 | `/home/mykola/repos/kaggle_clicks/runs/20251221_052644_paper_full_grid_10pct_A2_calendar_foldA` |
| A2_calendar | B | A2 calendar windows on trailing (1, 3, 6, 12, 24) | 0 | yes | 0.7625 | 0.7516 | `/home/mykola/repos/kaggle_clicks/runs/20251221_053406_paper_full_grid_10pct_A2_calendar_foldB` |
| A2_event50 | A | A2 event windows last50 | 0 | yes | 0.7545 | 0.7484 | `/home/mykola/repos/kaggle_clicks/runs/20251221_054046_paper_full_grid_10pct_A2_event50_foldA` |
| A2_event50 | B | A2 event windows last50 | 0 | yes | 0.7630 | 0.7522 | `/home/mykola/repos/kaggle_clicks/runs/20251221_054840_paper_full_grid_10pct_A2_event50_foldB` |
| A3_trailing | A | A3 trailing windows (1, 6, 24, 48, 168) | 0 | yes | 0.7536 | 0.7483 | `/home/mykola/repos/kaggle_clicks/runs/20251221_055523_paper_full_grid_10pct_A3_trailing_foldA` |
| A3_trailing | B | A3 trailing windows (1, 6, 24, 48, 168) | 0 | yes | 0.7643 | 0.7520 | `/home/mykola/repos/kaggle_clicks/runs/20251221_060137_paper_full_grid_10pct_A3_trailing_foldB` |
| A3_gap1 | A | A3 gap g=1 windows (1, 6, 24, 48, 168) | 0 | yes | 0.7521 | 0.7460 | `/home/mykola/repos/kaggle_clicks/runs/20251221_060734_paper_full_grid_10pct_A3_gap1_foldA` |
| A3_gap1 | B | A3 gap g=1 windows (1, 6, 24, 48, 168) | 0 | yes | 0.7623 | 0.7504 | `/home/mykola/repos/kaggle_clicks/runs/20251221_061425_paper_full_grid_10pct_A3_gap1_foldB` |
| A3_bucket | A | A3 bucket edges (1, 6, 24, 48, 168) | 0 | yes | 0.7536 | 0.7471 | `/home/mykola/repos/kaggle_clicks/runs/20251221_062108_paper_full_grid_10pct_A3_bucket_foldA` |
| A3_bucket | B | A3 bucket edges (1, 6, 24, 48, 168) | 0 | yes | 0.7622 | 0.7506 | `/home/mykola/repos/kaggle_clicks/runs/20251221_062804_paper_full_grid_10pct_A3_bucket_foldB` |
| A3_calendar | A | A3 calendar windows on trailing (1, 6, 24, 48, 168) | 0 | yes | 0.7534 | 0.7483 | `/home/mykola/repos/kaggle_clicks/runs/20251221_063436_paper_full_grid_10pct_A3_calendar_foldA` |
| A3_calendar | B | A3 calendar windows on trailing (1, 6, 24, 48, 168) | 0 | yes | 0.7642 | 0.7518 | `/home/mykola/repos/kaggle_clicks/runs/20251221_064207_paper_full_grid_10pct_A3_calendar_foldB` |
| A3_event50 | A | A3 event windows last50 | 0 | yes | 0.7546 | 0.7486 | `/home/mykola/repos/kaggle_clicks/runs/20251221_064852_paper_full_grid_10pct_A3_event50_foldA` |
| A3_event50 | B | A3 event windows last50 | 0 | yes | 0.7643 | 0.7524 | `/home/mykola/repos/kaggle_clicks/runs/20251221_065648_paper_full_grid_10pct_A3_event50_foldB` |
| A4_trailing | A | A4 trailing windows (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | yes | 0.7536 | 0.7456 | `/home/mykola/repos/kaggle_clicks/runs/20251221_070343_paper_full_grid_10pct_A4_trailing_foldA` |
| A4_trailing | B | A4 trailing windows (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | yes | 0.7638 | 0.7520 | `/home/mykola/repos/kaggle_clicks/runs/20251221_071321_paper_full_grid_10pct_A4_trailing_foldB` |
| A4_gap1 | A | A4 gap g=1 windows (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7525 | 0.7451 | `/home/mykola/repos/kaggle_clicks/runs/20251221_183006_paper_full_grid_10pct_A4_gap1_foldA` |
| A4_gap1 | B | A4 gap g=1 windows (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7626 | 0.7510 | `/home/mykola/repos/kaggle_clicks/runs/20251221_183007_paper_full_grid_10pct_A4_gap1_foldB` |
| A4_bucket | A | A4 bucket edges (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7537 | 0.7465 | `/home/mykola/repos/kaggle_clicks/runs/20251221_183007_paper_full_grid_10pct_A4_bucket_foldA` |
| A4_bucket | B | A4 bucket edges (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7619 | 0.7508 | `/home/mykola/repos/kaggle_clicks/runs/20251221_184655_paper_full_grid_10pct_A4_bucket_foldB` |
| A4_calendar | A | A4 calendar windows on trailing (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7536 | 0.7473 | `/home/mykola/repos/kaggle_clicks/runs/20251221_184655_paper_full_grid_10pct_A4_calendar_foldA` |
| A4_event50 | A | A4 event windows last50 | 0 | no | 0.7535 | 0.7475 | `/home/mykola/repos/kaggle_clicks/runs/20251221_190253_paper_full_grid_10pct_A4_event50_foldA` |
| A4_calendar | B | A4 calendar windows on trailing (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7639 | 0.7519 | `/home/mykola/repos/kaggle_clicks/runs/20251221_190252_paper_full_grid_10pct_A4_calendar_foldB` |
| A4_event50 | B | A4 event windows last50 | 0 | no | 0.7639 | 0.7517 | `/home/mykola/repos/kaggle_clicks/runs/20251221_191654_paper_full_grid_10pct_A4_event50_foldB` |
