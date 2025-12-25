# 10% Paper Grid (Time-Agg Only, Fold A/B)

This folder contains the full 20×2 “paper grid” run on the 10% deterministic sample with **per-category TE disabled** (`--no-te`), keeping only:

- base time features (`hour_of_day`) and online prior (`prior_ctr`)
- Family A time-aggregation features (windows/shapes)

## Contents

- `summary.md` / `summary.csv`: per-run metrics for 20 specs × Fold A/B (40 runs)
- `master_results.md` / `master_results.csv`: mean/std across folds, plus per-fold columns
- `inference_vs_baseline.csv`: paired inference vs a chosen baseline run id (ROC-AUC: DeLong; PR-AUC: block bootstrap)
- `contrasts_vs_trailing_test.md` / `contrasts_vs_trailing_test.csv`: within-length-set shape-vs-trailing ROC-AUC contrasts (paired DeLong)
- `plots/`: digestible plots (shape/length-set averages + heatmap)
- Repro: `commands.sh`, `sweep_config.json`, `source_sweep_dir.txt`

## Notes

- Inference baseline: `A3_trailing`.
- Use `paper_draft/v1/10pct_paper_grid/` for the comparable TE+time-agg results (with inference).
