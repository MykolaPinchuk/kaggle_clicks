# Writer Brief (v1): Online-Safe Time-Aggregation Features on Avazu CTR

This is a writer-facing narrative brief for producing a paper draft from the existing v1 artifacts under `paper_draft/v1/`. It is intentionally “results-first”: it states the key claims we can support, points to the exact tables/plots, and suggests a figure set that is easier to digest than the full tables.

## One-paragraph premise
We study **entity-history time-aggregation features** for CTR prediction on Avazu under a strict, leakage-averse setup: **out-of-time (OOT)** splits and an **online constraint** where features for hour `H` are computed only from events strictly before `H` (no same-hour leakage). Within this controlled protocol, seemingly minor design choices—**which window lengths** to include (A.2) and **how windows are constructed** (A.1 shapes: trailing, gap, bucketized, calendar-aligned, event-count windows)—produce consistent, statistically supported differences in performance.

## What is “fixed” vs “varied”
Fixed across the main grid:
- Dataset sample and evaluation folds: 10% deterministic sample; rolling-tail **Fold A/B**.
- Entities included in time-aggregation: `device_ip`, `device_id`, `app_id`, `site_id`.
- Model family and hyperparameters: XGBoost (frozen).
- Time-aware TE implementation (in the TE grid).

Varied in the grid (Family A only):
- **A.2 window length sets**: A1/A2/A3/A4.
- **A.1 shapes** (applied to a length set): trailing, gap1, bucket, calendar, event50.

Reference for the experimental grid/protocol (conceptual): `docs/TIME_AGG_EXPERIMENTS.md`.

## Core results to emphasize (with supporting artifacts)

### R1) Time-aggregation adds a clear lift over TE-only (online-safe, OOT)
On the 10% sample with rolling-tail folds, adding time-aggregation (A3 trailing) improves both ROC-AUC and PR-AUC vs TE-only, with tight paired confidence intervals.

Use:
- Headline table + paired inference: `paper_draft/v1/10pct_te_lift/te_vs_timeagg_summary.md`

Suggested phrasing:
- “Under our online-safe OOT protocol, entity-history features deliver a statistically decisive lift over a strong TE baseline.”

### R2) Within the comparable TE+time-agg grid, **A3 trailing** is a strong default; **event-count windows** are the only consistently helpful shape tweak
In the TE+time-agg grid, the baseline spec is `A3_trailing` and the best (or near-best) variant is typically **`event50`** on top of a length set (especially A3). The gain is modest in ROC-AUC (but consistent across folds) and PR-AUC is mixed/small.

Use:
- Main table (means/stds + per-fold deltas): `paper_draft/v1/10pct_paper_grid/master_results.md`
- Per-length-set shape contrasts vs trailing: `paper_draft/v1/10pct_paper_grid/contrasts_vs_trailing_test.md`
- (Optional) Full sweep details + additional plots already present: `paper_draft/v1/10pct_paper_grid/README.md` and `paper_draft/v1/10pct_paper_grid/plots/`

Suggested phrasing:
- “The dominant choice is the window-length set: A3 trailing is a robust baseline. Among shapes, adding an event-count window (`event50`) yields the only consistently positive contrast vs trailing within length sets.”

### R3) Gap windows and bucketized windows are reliably worse (in this dataset/protocol)
Both the TE grid and the no-TE grid show **gap1** and **bucket** as consistently negative vs trailing within a length set, with confidence intervals excluding zero in most cases.

Use:
- TE contrasts: `paper_draft/v1/10pct_paper_grid/contrasts_vs_trailing_test.md`
- no-TE contrasts: `paper_draft/v1/10pct_paper_grid_noTE/contrasts_vs_trailing_test.md`

Suggested phrasing:
- “More ‘clever’ window constructions are not automatically better: gap windows and bucketized recency buckets underperform simple trailing windows under online-safe evaluation.”

### R4) TE changes the picture: without TE, event-count windows matter more; calendar effects are small and mixed
In the no-TE grid, the **event50** shape yields larger improvements vs trailing than in the TE grid (still under the same online/OOT protocol). Calendar windows show small, inconsistent effects (sometimes positive, sometimes negative depending on length set), and do not dominate.

Use:
- no-TE table: `paper_draft/v1/10pct_paper_grid_noTE/master_results.md`
- TE vs no-TE deltas across specs (nice “interaction” summary): `paper_draft/v1/combined/te_vs_note_summary.md`

Suggested phrasing:
- “TE absorbs part of the signal that event-count windows recover in the no-TE setting; nevertheless, event-count windows remain the most robust shape modification.”

## How to tell the story (recommended section flow)
1) **Why online-safe OOT matters** (leakage risk + drift).
2) **Define Family A** and the grid (A.1 shapes × A.2 length sets).
3) **Headline: TE-only → time-agg lift** (simple, strong first result).
4) **Main grid: which constructions win/lose** (emphasize patterns over tiny deltas).
5) **Sensitivity (no-TE)**: shape conclusions largely persist; event windows become more important; TE interaction summary.
6) **Practical guidance**: recommended default + “avoid list”.

## Suggested figure set (readers-first)
Aim for 4–6 figures total; put the full tables in the appendix.

Main-text candidates:
1) **Figure: TE-only vs time-agg lift** (Fold A/B bars + paired deltas).
   - Source: `paper_draft/v1/10pct_te_lift/te_vs_timeagg_summary.md`
2) **Figure: Heatmap of Test ROC-AUC across (length set × shape)** for TE+time-agg and no-TE (side-by-side).
   - Existing per-grid heatmaps:
     - `paper_draft/v1/10pct_paper_grid/plots/heatmap_test_roc.png`
     - `paper_draft/v1/10pct_paper_grid_noTE/plots/heatmap_test_roc.png`
3) **Figure: Shape contrasts vs trailing within each length set** (forest or grouped bars; show CIs).
   - Sources:
     - `paper_draft/v1/10pct_paper_grid/contrasts_vs_trailing_test.md`
     - `paper_draft/v1/10pct_paper_grid_noTE/contrasts_vs_trailing_test.md`
4) **Figure: TE vs no-TE uplift by spec** (sorted bars).
   - Source: `paper_draft/v1/combined/te_vs_note_summary.md`

Appendix candidates:
- Full master tables:
  - `paper_draft/v1/10pct_paper_grid/master_results.md`
  - `paper_draft/v1/10pct_paper_grid_noTE/master_results.md`
- Inference vs baseline tables (per-fold):
  - `paper_draft/v1/10pct_paper_grid/inference_vs_baseline.csv`
  - `paper_draft/v1/10pct_paper_grid_noTE/inference_vs_baseline.csv`
- EDA context figure(s) to justify drift/cold-start:
  - `paper_draft/v1/eda/eda_assets/ctr_by_day.png`
  - `paper_draft/v1/eda/eda_assets/unseen_rate_top.png`

## “Default spec” recommendation (for Discussion / Practical guidance)
Under this protocol on Avazu:
- Default: **A3 trailing** (TE+time-agg).
- Optional tweak: **A3 + event50** if you want a small additional ROC lift (PR is small/mixed).
- Avoid: **gap1** and **bucket** shapes (consistently negative).

## Caveats / limitations to acknowledge
- Single dataset (Avazu) and hour-level timestamps; results may differ elsewhere.
- Only two rolling-tail folds (A/B); we use paired inference but not full TSCV.
- Scope is Family A only (A.1/A.2); we do not claim optimality over other families or extensive smoothing/regularization sweeps.

