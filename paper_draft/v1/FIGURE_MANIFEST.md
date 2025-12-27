# Figure Manifest (v1)

This file lists the most useful “paper-ready” figures for the v1 draft, with suggested placement. When in doubt, prefer putting big tables and secondary plots into the Appendix.

## Main text (recommended)

1) **Headline lift: TE-only → time-aggregation**
- Absolute metrics (Fold A/B): `paper_draft/v1/plots/te_lift_metrics.png`
- Paired deltas + CIs (Fold A/B): `paper_draft/v1/plots/te_lift_deltas.png`

2) **Grid overview: which (length set × shape) wins**
- TE+time-agg heatmap: `paper_draft/v1/10pct_paper_grid/plots/heatmap_test_roc.png`
- no-TE heatmap: `paper_draft/v1/10pct_paper_grid_noTE/plots/heatmap_test_roc.png`
- Side-by-side combined (same color scale): `paper_draft/v1/plots/heatmap_test_roc_te_vs_note.png`

3) **Within-length-set contrasts: shape vs trailing**
- TE contrasts: `paper_draft/v1/10pct_paper_grid/contrasts_vs_trailing_test.md`
- no-TE contrasts: `paper_draft/v1/10pct_paper_grid_noTE/contrasts_vs_trailing_test.md`
- Combined bar chart (TE vs no-TE): `paper_draft/v1/plots/shape_contrasts_te_vs_note.png`

4) **Interaction summary: TE vs no-TE uplift**
- Sorted bars: `paper_draft/v1/plots/te_vs_note_uplift.png`
- Table (backup / appendix): `paper_draft/v1/combined/te_vs_note_summary.md`

## Appendix (recommended)

- Master result tables (main evidence):
  - `paper_draft/v1/10pct_paper_grid/master_results.md`
  - `paper_draft/v1/10pct_paper_grid_noTE/master_results.md`
- Full per-fold inference vs baseline:
  - `paper_draft/v1/10pct_paper_grid/inference_vs_baseline.csv`
  - `paper_draft/v1/10pct_paper_grid_noTE/inference_vs_baseline.csv`
- Drift / long-tail / cold-start context (pick 1–2):
  - `paper_draft/v1/eda/eda_assets/ctr_by_day.png`
  - `paper_draft/v1/eda/eda_assets/unseen_rate_top.png`

## Regenerating the plots
- Script: `human_src/generate_paper_plots_v1.py`
- Default output directory: `paper_draft/v1/plots/`

