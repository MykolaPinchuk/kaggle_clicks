# Plan for Working Paper Results (A.1 + A.2 only)

This document specifies a concrete, implementable plan to produce **paper-quality**, **statistically supported** results for **Family A** time aggregation features on Avazu CTR, scoped strictly to:

- **A.1 Window shapes** (trailing, gap, bucketized, calendar-aligned, event-count windows)
- **A.2 Window length sets** (which hours-back windows are included)

Everything in **A.3 aggregation targets** and **A.4 regularization choices** is explicitly **out of scope** for the working paper results at this stage.

## 0) Goals (what “done” looks like)

Deliverables for the working paper:

1. A single **master table** of results covering the full A.1/A.2 sweep on a 5% deterministic sample:
   - mean + dispersion across **2 test-day cutoffs** (see §2)
   - per-cutoff metrics (for transparency)
   - runtime + feature count per spec (for practicality/reproducibility)
2. For the comparisons that matter, provide **uncertainty**:
   - **paired DeLong** CI/p-value for ΔROC-AUC vs a baseline (cheap)
   - **block bootstrap** CI for ΔPR-AUC vs baseline (cheap enough if scoped)
3. A final “precision” run on **full dataset** for 1–2 best specs, plus the same inference framework.

## 1) Scope + “frozen” settings (to isolate time construction)

To attribute differences to time-window construction only:

- **Entities are fixed**: `device_ip`, `device_id`, `app_id`, `site_id`.
- **Model config is fixed** across the whole sweep:
  - XGBoost params, seed, early stopping settings, etc.
- **TE config is fixed** (e.g., `m=100`) and time-aware as implemented.
- **CTR smoothing (alpha/beta) in time-agg is fixed** (do not sweep).
- We do not change which statistics are emitted besides what is required by A.1/A.2 variants.

Note: “event-count windows last N imps” is in-scope as an A.1 *shape* choice; see §3.2.

## 2) Stability without full TSCV: “2-day rolling tail”

Proper time-series cross-validation over many folds is likely too slow. Instead, we use a minimal stability check:

Define the last 3 days in the sample timeline as `D-2`, `D-1`, `D` (where `D` is the last day in the sampled dataset).

Run two folds (each fold is a full train/val/test run):

- **Fold A**
  - Train: all days `< D-1`
  - Val: day `D-1`
  - Test: day `D`
- **Fold B**
  - Train: all days `< D-2`
  - Val: day `D-2`
  - Test: day `D-1`

Report per-config metrics on Fold A and Fold B test sets, plus mean/std.

Why this is acceptable for a paper:
- It exposes whether a “winner” only wins on one specific day (common in ad CTR).
- It costs ~2× per spec (still much cheaper than full TSCV).

## 3) The A.1/A.2 sweep grid (what we will run)

### 3.1 A.2 window length sets (trailing contiguous, g=0)

Run these **length sets** (hours):

- **A1 (compact)**: `{1, 6, 24}`
- **A2 (balanced)**: `{1, 3, 6, 12, 24}`
- **A3 (multi-scale)**: `{1, 6, 24, 48, 168}`
- **A4 (log-spaced)**: optional; include only if runtime/memory is acceptable on 5%

This phase chooses a “base length set” (typically top 1–2) to be used for A.1 shape comparisons.

### 3.2 A.1 window shapes (on top of a chosen length set)

Given a base length set `W` (e.g., A3), compare these shapes:

1. **Trailing contiguous windows** (baseline within A.1)
   - Features for `[H-w, H)` for all `w ∈ W`.

2. **Gap windows**
   - `g=1`: use `[H-w-g, H-g)` for all `w ∈ W`
   - `g=6`: same, but gap is 6h

3. **Bucketized disjoint recency windows**
   - Canonical buckets: `0–1h`, `1–6h`, `6–24h`, `24–168h`
   - Implemented as differences of cumulative trailing windows; bucket edges are `{1,6,24,168}`.
   - (Important) Do not introduce multiple bucket schemes unless pre-registered; otherwise the grid explodes.

4. **Calendar-aligned windows**
   - Add “today so far” + “yesterday” features on top of the base trailing windows.

5. **Event-count windows** (“last N impressions”)
   - `N ∈ {10, 50, 200}` (these are *shape* hyperparameters)
   - Implemented as rolling over hour-blocks with trimming of the oldest block (approx but deterministic).

### 3.3 Comparison policy (to avoid quadratic inference cost)

Do **not** compute inference for all pairs.

Instead:
- Choose one baseline spec per phase (e.g., A3 trailing).
- Compare every other spec to that baseline (paired inference).
- Optionally do a second baseline comparison against TE-only for “absolute lift”.

## 4) Statistical inference plan (must stay < 2× overhead)

We want inference that is:
- paired (uses the same rows),
- cheap enough (under ~2× the runtime of training),
- robust to time dependence.

### 4.1 Required artifact: per-run prediction export (test and val)

For each (run, fold), export predictions:

- `runs/<timestamp>_<tag>/preds_val.parquet`
- `runs/<timestamp>_<tag>/preds_test.parquet`

Minimal columns (avoid huge files):
- `row_id` (int): stable row identifier within the sampled dataset after sorting by time
- `hour_dt` (timestamp): for block bootstrap grouping (by hour)
- `y` (int8): label
- `p` (float32): predicted probability
- `fold_id` (string): `"A"` or `"B"` (or similar)

Implementation detail:
- Define `row_id` as the DataFrame index after `sort_values("hour_dt").reset_index(drop=True)`.
- This avoids relying on `id` (which may be float in existing sample parquet).

### 4.2 ROC-AUC inference: paired DeLong

For each comparison (model vs baseline) on the *same test set*:
- compute ΔROC-AUC
- compute DeLong standard error + 95% CI + p-value

This is fast (seconds for ~200–400k examples).

### 4.3 PR-AUC inference: block bootstrap by hour

PR-AUC lacks a simple analytic paired test like DeLong. Use a **paired block bootstrap**:

- block = `hour_dt` (one hour)
- sample hours with replacement to form a bootstrap replicate
- compute PR-AUC for both models on the replicate and take Δ

To keep overhead bounded:
- use `B=200` replicates for routine reporting
- only run PR-AUC bootstrap for:
  - the top-K candidate specs vs baseline, and/or
  - the specs that appear in the paper’s main table

Expected overhead:
- DeLong is negligible.
- Bootstrap can be minutes, but if we only do it for a limited set of comparisons, total overhead stays well under 2×.

## 5) Implementation checklist (for the next agent)

### 5.1 Add “rolling tail fold” support

Add a new split mode in `kaggle_clicks/time_utils.py` (or in the sweep runner) that yields Fold A / Fold B splits based on the final 3 days present in the current sampled dataset.

Constraints:
- Must maintain strict OOT within each fold.
- Fold definitions must be deterministic and logged.

### 5.2 Export predictions in `kaggle_clicks/run_baseline_te.py`

Enhance the baseline runner to optionally export predictions:

New CLI flags (suggested):
- `--export-preds` (bool)
- `--fold-id A|B` (string, written into preds files)

When confirming export correctness:
- Ensure `row_id` aligns with the `fe` DataFrame (post sort/reset).
- Ensure `preds_test.parquet` contains only test rows.

### 5.3 Add inference utilities

Add module(s), e.g.:
- `kaggle_clicks/inference_auc.py`:
  - `delong_roc_test(y, p_a, p_b) -> delta, ci_low, ci_high, p_value`
- `kaggle_clicks/inference_bootstrap.py`:
  - `paired_block_bootstrap_pr_auc(df_pred_a, df_pred_b, group_col='hour_dt', B=200, seed=42)`

Add quick self-tests on small synthetic arrays (no heavy test framework required if repo doesn’t use one).

### 5.4 Extend sweep runner to orchestrate folds + inference

Extend `kaggle_clicks/run_sweep_family_a.py` (or add a new `run_sweep_working_paper.py`) to:

1. Enumerate the full A.2 then A.1 grid (as specified in §3).
2. Run each spec for Fold A and Fold B (2 runs per spec).
3. Export predictions for each run.
4. Post-process the sweep into:
   - `runs/sweeps/.../master_results.csv` (one row per (spec, fold))
   - `runs/sweeps/.../summary.md` (main table: means, stds)
   - `runs/sweeps/.../inference_vs_baseline.csv` (Δ metrics + CIs)

### 5.5 Full-data “precision” phase (after selection)

After the 5% sweep narrows down 1–2 best specs:
- rerun those specs on full dataset
- repeat Fold A/B inference (still only 2 folds)
- produce a final “paper table” artifact (CSV + markdown).

## 6) Output structure (what to expect on disk)

- Per-run (already used today):
  - `runs/<timestamp>_<tag>/{config.json,metrics.json,report.md,model.json}`
- New:
  - `runs/<timestamp>_<tag>/preds_val.parquet`
  - `runs/<timestamp>_<tag>/preds_test.parquet`
- Per-sweep:
  - `runs/sweeps/<timestamp>_<tag>_s5pct/summary.md`
  - `runs/sweeps/<timestamp>_<tag>_s5pct/master_results.csv`
  - `runs/sweeps/<timestamp>_<tag>_s5pct/inference_vs_baseline.csv`

## 7) Guardrails / “don’t accidentally change scope”

- Do not add new aggregation targets beyond what the current Family A implementation emits.
- Do not tune smoothing/backoff thresholds.
- If a feature change would be classified as A.3/A.4, stop and get explicit approval.

