# Research Agent Bundle (v1) — 20-file-cap friendly

This directory is the **entire** package intended for the Writer agent. It is intentionally small (well under 20 files) and flat (no subdirectories).

## Start here (2 Markdown docs)
- `WRITER_BRIEF.md` — narrative writeup + key claims + how to use the evidence
- `BUNDLE_README.md` — this file (what each artifact is)

## Key evidence tables (CSV)
- `te_vs_timeagg_inference_test.csv` — paired inference for TE-only → time-agg lift (ΔROC DeLong; ΔPR hour-block bootstrap)
- `te_vs_timeagg_metrics_selected_runs.csv` — absolute metrics for TE-only, A3 trailing, A3+event50 (Fold A/B)
- `te_grid_master_results.csv` — main TE+time-agg grid (A.1×A.2), mean/std across folds + per-fold deltas
- `note_grid_master_results.csv` — no-TE grid (time-agg only), same structure
- `te_vs_note_summary.csv` — fold-averaged TE uplift vs no-TE per spec
- `te_vs_note_by_fold.csv` — per-fold TE uplift vs no-TE per spec

## Figures (PNG)
- `te_lift_metrics.png` — TE-only vs time-agg absolute ROC/PR by fold
- `te_lift_deltas.png` — TE-only vs time-agg paired deltas + confidence intervals
- `heatmap_test_roc_te_vs_note.png` — side-by-side heatmap of Test ROC-AUC over (length set × shape), TE vs no-TE
- `shape_contrasts_te_vs_note.png` — mean shape-vs-trailing contrasts, faceted by length set, TE vs no-TE
- `te_vs_note_uplift.png` — TE uplift vs no-TE by spec (ROC and PR)
- `eda_ctr_by_day.png` — dataset drift context (appendix)
- `eda_unseen_rate_top.png` — cold-start/unseen-rate context (appendix)

## Additional main figures requested by Writer agent (PNG + PDF)
Directory: `figures_time_aggregation_xgb_v1/`
- `fig_decision_lengths_then_shapes.(png|pdf)`
- `fig_league_table_top_bottom.(png|pdf)`
- `fig_traffic_light_shape_contrasts.(png|pdf)`
- `fig_protocol_schematic.(png|pdf)`
- `fig_avg_performance_by_shape_panels.(png|pdf)` (2×2: test/val × ROC/PR; averaged over length tuples and folds)
- `fig_avg_performance_by_length_tuple_panels.(png|pdf)` (2×2: test/val × ROC/PR; averaged over shapes and folds)
- Trimmed no-TE-only (Test PR-AUC only; y-axis starts at 0.34; no error bars):
  - `fig_trimmed_note_test_prauc_by_shape.(png|pdf)`
  - `fig_trimmed_note_test_prauc_by_length_tuple.(png|pdf)`
