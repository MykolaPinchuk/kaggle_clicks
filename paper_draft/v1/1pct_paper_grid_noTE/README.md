# 1% Paper Grid (Time-Agg Only, Fold A/B)

This folder contains a full 20×2 “paper grid” run on the 1% deterministic sample with **per-category TE disabled** (`--no-te`), keeping only:

- base time features (`hour_of_day`) and online prior (`prior_ctr`)
- Family A time-aggregation features (windows/shapes)

## Contents

- `summary.md` / `summary.csv`: per-run metrics for 20 specs × Fold A/B (40 runs)
- `master_results.md` / `master_results.csv`: mean/std across folds, plus per-fold columns
- `plots/`: digestible plots (shape/length-set averages + heatmap)

## Notes

- No statistical inference is included here because prediction exports were not generated for this sweep.
- The primary purpose is to see whether removing TE increases sensitivity/variation across A.1/A.2 specs.

