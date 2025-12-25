# Working Paper v1 — Results Manifest

This folder is the “single source of truth” for the v1 working paper artifacts (tables + plots + key supporting reports).

Writer quickstart: `paper_draft/v1/WRITER_ONBOARDING.md`

## Combined tables (TE vs no-TE)

Directory: `paper_draft/v1/combined/`

- Per-fold paired comparisons: `paper_draft/v1/combined/te_vs_note_by_fold.csv`
- Fold-averaged summary: `paper_draft/v1/combined/te_vs_note_summary.md`, `paper_draft/v1/combined/te_vs_note_summary.csv`

## 10% TE + time-agg (main comparable grid + inference)

Directory: `paper_draft/v1/10pct_paper_grid/`

- Grid metrics: `paper_draft/v1/10pct_paper_grid/summary.md`, `paper_draft/v1/10pct_paper_grid/summary.csv`
- Master table (mean/std across folds): `paper_draft/v1/10pct_paper_grid/master_results.md`, `paper_draft/v1/10pct_paper_grid/master_results.csv`
- Inference vs baseline (paired, per fold): `paper_draft/v1/10pct_paper_grid/inference_vs_baseline.csv`
- Shape-vs-trailing contrasts (paired DeLong ROC-AUC, per length set & fold): `paper_draft/v1/10pct_paper_grid/contrasts_vs_trailing_test.md`, `paper_draft/v1/10pct_paper_grid/contrasts_vs_trailing_test.csv`
- Plots: `paper_draft/v1/10pct_paper_grid/plots/`
- Repro (exact sweep inputs): `paper_draft/v1/10pct_paper_grid/commands.sh`, `paper_draft/v1/10pct_paper_grid/sweep_config.json`

Notes:
- Inference is computed from per-row prediction exports stored in the original run dirs referenced by the CSVs.

## 10% time-agg only (no TE; “sensitivity” view)

Directory: `paper_draft/v1/10pct_paper_grid_noTE/`

- Grid metrics: `paper_draft/v1/10pct_paper_grid_noTE/summary.md`, `paper_draft/v1/10pct_paper_grid_noTE/summary.csv`
- Master table: `paper_draft/v1/10pct_paper_grid_noTE/master_results.md`, `paper_draft/v1/10pct_paper_grid_noTE/master_results.csv`
- Inference vs baseline (paired, per fold): `paper_draft/v1/10pct_paper_grid_noTE/inference_vs_baseline.csv`
- Shape-vs-trailing contrasts (paired DeLong ROC-AUC): `paper_draft/v1/10pct_paper_grid_noTE/contrasts_vs_trailing_test.md`, `paper_draft/v1/10pct_paper_grid_noTE/contrasts_vs_trailing_test.csv`
- Plots: `paper_draft/v1/10pct_paper_grid_noTE/plots/`
- Repro (exact sweep inputs): `paper_draft/v1/10pct_paper_grid_noTE/commands.sh`, `paper_draft/v1/10pct_paper_grid_noTE/sweep_config.json`

Notes:
- Inference baseline: `A3_trailing`.

## 1% time-agg only (no TE; fast sanity/sensitivity)

Directory: `paper_draft/v1/1pct_paper_grid_noTE/`

- Grid metrics: `paper_draft/v1/1pct_paper_grid_noTE/summary.md`, `paper_draft/v1/1pct_paper_grid_noTE/summary.csv`
- Master table: `paper_draft/v1/1pct_paper_grid_noTE/master_results.md`, `paper_draft/v1/1pct_paper_grid_noTE/master_results.csv`
- Plots: `paper_draft/v1/1pct_paper_grid_noTE/plots/`

## TE-only lift vs time-agg (headline “big lift” comparison)

Directory: `paper_draft/v1/10pct_te_lift/`

- Summary: `paper_draft/v1/10pct_te_lift/te_vs_timeagg_summary.md`
- Paired inference: `paper_draft/v1/10pct_te_lift/inference_test_te_vs_timeagg.csv`
- Copied run reports/configs: `paper_draft/v1/10pct_te_lift/run_reports/`

## Scripts used / added during this work

- Contrast generator: `kaggle_clicks/postprocess_paper_grid_contrasts.py`
- Inference vs baseline: `kaggle_clicks/postprocess_sweep_inference.py`
- No-TE mode: `kaggle_clicks/run_baseline_te.py` (`--no-te`)
- No-TE sweep support: `kaggle_clicks/run_sweep_family_a_full_grid.py` (`--no-te`)
- TE vs no-TE comparison: `kaggle_clicks/postprocess_paper_grid_te_vs_note.py`
- Master table generator (optional): `kaggle_clicks/postprocess_paper_grid_master.py`

## Remaining optional work (depending on draft goals)

- Write the paper draft text/figures using the tables/plots above.

## Supporting snapshot

- EDA: `paper_draft/v1/eda/EDA_REPORT.md`
