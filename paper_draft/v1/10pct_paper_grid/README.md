# 10% Paper Grid (TE + Time-Agg, Fold A/B)

This folder contains the full 20×2 “paper grid” run on the 10% deterministic sample with:

- base time features (`hour_of_day`) and online prior (`prior_ctr`)
- time-aware per-category TE features
- Family A time-aggregation features (windows/shapes)

## Contents

- `summary.md` / `summary.csv`: per-run metrics for 20 specs × Fold A/B (40 runs)
- `master_results.md` / `master_results.csv`: mean/std across folds, plus per-fold inference columns
- `inference_vs_baseline.csv`: paired inference vs `A3_trailing` (ROC-AUC: DeLong; PR-AUC: hour-block bootstrap)
- `contrasts_vs_trailing_test.md` / `contrasts_vs_trailing_test.csv`: within-length-set shape-vs-trailing ROC-AUC contrasts (paired DeLong)
- `plots/`: digestible plots (shape/length-set averages + heatmap + delta CIs)
- Repro: `commands.sh`, `sweep_config.json`, `source_sweep_dir.txt`

## Notes

- Inference baseline: `A3_trailing`.

