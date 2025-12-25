# Sweep Summary: `paper_full_grid_1pct_noTE`

- Created (UTC): `20251225_024117`
- Sample: `1%` (`data/interim/train_sample.parquet`)
- Train CSV: `/home/mykola/repos/kaggle_clicks/data/raw/train.csv`

| Run | Fold | Description | Return | Skipped | Val ROC-AUC | Test ROC-AUC | Run dir |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| A1_trailing | A | A1 trailing windows (1, 6, 24) | 0 | no | 0.7311 | 0.7257 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024118_paper_full_grid_1pct_noTE_A1_trailing_foldA` |
| A1_trailing | B | A1 trailing windows (1, 6, 24) | 0 | no | 0.7388 | 0.7299 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024118_paper_full_grid_1pct_noTE_A1_trailing_foldB` |
| A1_gap1 | A | A1 gap g=1 windows (1, 6, 24) | 0 | no | 0.7277 | 0.7238 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024152_paper_full_grid_1pct_noTE_A1_gap1_foldA` |
| A1_gap1 | B | A1 gap g=1 windows (1, 6, 24) | 0 | no | 0.7354 | 0.7275 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024158_paper_full_grid_1pct_noTE_A1_gap1_foldB` |
| A1_bucket | A | A1 bucket edges (1, 6, 24) | 0 | no | 0.7299 | 0.7250 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024223_paper_full_grid_1pct_noTE_A1_bucket_foldA` |
| A1_bucket | B | A1 bucket edges (1, 6, 24) | 0 | no | 0.7363 | 0.7264 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024234_paper_full_grid_1pct_noTE_A1_bucket_foldB` |
| A1_calendar | A | A1 calendar windows on trailing (1, 6, 24) | 0 | no | 0.7323 | 0.7268 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024310_paper_full_grid_1pct_noTE_A1_calendar_foldA` |
| A1_calendar | B | A1 calendar windows on trailing (1, 6, 24) | 0 | no | 0.7393 | 0.7323 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024316_paper_full_grid_1pct_noTE_A1_calendar_foldB` |
| A1_event50 | A | A1 event windows last50 | 0 | no | 0.7333 | 0.7271 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024355_paper_full_grid_1pct_noTE_A1_event50_foldA` |
| A1_event50 | B | A1 event windows last50 | 0 | no | 0.7437 | 0.7360 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024401_paper_full_grid_1pct_noTE_A1_event50_foldB` |
| A2_trailing | A | A2 trailing windows (1, 3, 6, 12, 24) | 0 | no | 0.7279 | 0.7245 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024427_paper_full_grid_1pct_noTE_A2_trailing_foldA` |
| A2_trailing | B | A2 trailing windows (1, 3, 6, 12, 24) | 0 | no | 0.7357 | 0.7268 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024441_paper_full_grid_1pct_noTE_A2_trailing_foldB` |
| A2_gap1 | A | A2 gap g=1 windows (1, 3, 6, 12, 24) | 0 | no | 0.7267 | 0.7231 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024502_paper_full_grid_1pct_noTE_A2_gap1_foldA` |
| A2_gap1 | B | A2 gap g=1 windows (1, 3, 6, 12, 24) | 0 | no | 0.7347 | 0.7251 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024515_paper_full_grid_1pct_noTE_A2_gap1_foldB` |
| A2_bucket | B | A2 bucket edges (1, 3, 6, 12, 24) | 0 | no | 0.7345 | 0.7268 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024554_paper_full_grid_1pct_noTE_A2_bucket_foldB` |
| A2_bucket | A | A2 bucket edges (1, 3, 6, 12, 24) | 0 | no | 0.7297 | 0.7237 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024541_paper_full_grid_1pct_noTE_A2_bucket_foldA` |
| A2_calendar | A | A2 calendar windows on trailing (1, 3, 6, 12, 24) | 0 | no | 0.7323 | 0.7265 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024649_paper_full_grid_1pct_noTE_A2_calendar_foldA` |
| A2_calendar | B | A2 calendar windows on trailing (1, 3, 6, 12, 24) | 0 | no | 0.7396 | 0.7309 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024653_paper_full_grid_1pct_noTE_A2_calendar_foldB` |
| A2_event50 | A | A2 event windows last50 | 0 | no | 0.7373 | 0.7306 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024743_paper_full_grid_1pct_noTE_A2_event50_foldA` |
| A2_event50 | B | A2 event windows last50 | 0 | no | 0.7438 | 0.7364 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024747_paper_full_grid_1pct_noTE_A2_event50_foldB` |
| A3_trailing | A | A3 trailing windows (1, 6, 24, 48, 168) | 0 | no | 0.7366 | 0.7302 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024834_paper_full_grid_1pct_noTE_A3_trailing_foldA` |
| A3_trailing | B | A3 trailing windows (1, 6, 24, 48, 168) | 0 | no | 0.7472 | 0.7377 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024836_paper_full_grid_1pct_noTE_A3_trailing_foldB` |
| A3_gap1 | A | A3 gap g=1 windows (1, 6, 24, 48, 168) | 0 | no | 0.7337 | 0.7284 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024911_paper_full_grid_1pct_noTE_A3_gap1_foldA` |
| A3_gap1 | B | A3 gap g=1 windows (1, 6, 24, 48, 168) | 0 | no | 0.7437 | 0.7331 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024922_paper_full_grid_1pct_noTE_A3_gap1_foldB` |
| A3_bucket | A | A3 bucket edges (1, 6, 24, 48, 168) | 0 | no | 0.7357 | 0.7309 | `/home/mykola/repos/kaggle_clicks/runs/20251225_024951_paper_full_grid_1pct_noTE_A3_bucket_foldA` |
| A3_bucket | B | A3 bucket edges (1, 6, 24, 48, 168) | 0 | no | 0.7438 | 0.7345 | `/home/mykola/repos/kaggle_clicks/runs/20251225_025000_paper_full_grid_1pct_noTE_A3_bucket_foldB` |
| A3_calendar | A | A3 calendar windows on trailing (1, 6, 24, 48, 168) | 0 | no | 0.7368 | 0.7299 | `/home/mykola/repos/kaggle_clicks/runs/20251225_025042_paper_full_grid_1pct_noTE_A3_calendar_foldA` |
| A3_calendar | B | A3 calendar windows on trailing (1, 6, 24, 48, 168) | 0 | no | 0.7470 | 0.7364 | `/home/mykola/repos/kaggle_clicks/runs/20251225_025056_paper_full_grid_1pct_noTE_A3_calendar_foldB` |
| A3_event50 | A | A3 event windows last50 | 0 | no | 0.7384 | 0.7319 | `/home/mykola/repos/kaggle_clicks/runs/20251225_025126_paper_full_grid_1pct_noTE_A3_event50_foldA` |
| A3_event50 | B | A3 event windows last50 | 0 | no | 0.7451 | 0.7362 | `/home/mykola/repos/kaggle_clicks/runs/20251225_025147_paper_full_grid_1pct_noTE_A3_event50_foldB` |
| A4_trailing | A | A4 trailing windows (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7374 | 0.7310 | `/home/mykola/repos/kaggle_clicks/runs/20251225_025215_paper_full_grid_1pct_noTE_A4_trailing_foldA` |
| A4_trailing | B | A4 trailing windows (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7461 | 0.7362 | `/home/mykola/repos/kaggle_clicks/runs/20251225_025230_paper_full_grid_1pct_noTE_A4_trailing_foldB` |
| A4_gap1 | A | A4 gap g=1 windows (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7368 | 0.7303 | `/home/mykola/repos/kaggle_clicks/runs/20251225_025312_paper_full_grid_1pct_noTE_A4_gap1_foldA` |
| A4_gap1 | B | A4 gap g=1 windows (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7449 | 0.7368 | `/home/mykola/repos/kaggle_clicks/runs/20251225_025325_paper_full_grid_1pct_noTE_A4_gap1_foldB` |
| A4_bucket | A | A4 bucket edges (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7349 | 0.7294 | `/home/mykola/repos/kaggle_clicks/runs/20251225_025422_paper_full_grid_1pct_noTE_A4_bucket_foldA` |
| A4_bucket | B | A4 bucket edges (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7412 | 0.7325 | `/home/mykola/repos/kaggle_clicks/runs/20251225_025435_paper_full_grid_1pct_noTE_A4_bucket_foldB` |
| A4_calendar | A | A4 calendar windows on trailing (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7383 | 0.7321 | `/home/mykola/repos/kaggle_clicks/runs/20251225_025539_paper_full_grid_1pct_noTE_A4_calendar_foldA` |
| A4_calendar | B | A4 calendar windows on trailing (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0 | no | 0.7455 | 0.7347 | `/home/mykola/repos/kaggle_clicks/runs/20251225_025554_paper_full_grid_1pct_noTE_A4_calendar_foldB` |
| A4_event50 | A | A4 event windows last50 | 0 | no | 0.7384 | 0.7311 | `/home/mykola/repos/kaggle_clicks/runs/20251225_025651_paper_full_grid_1pct_noTE_A4_event50_foldA` |
| A4_event50 | B | A4 event windows last50 | 0 | no | 0.7471 | 0.7387 | `/home/mykola/repos/kaggle_clicks/runs/20251225_025655_paper_full_grid_1pct_noTE_A4_event50_foldB` |
