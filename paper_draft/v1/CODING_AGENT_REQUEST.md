# Coding Agent Request: Improved Main Figures for `time_aggregation_xgb_v1`

## Goal
Create a small set of reader friendly figures that communicate the main empirical finding:
1) window *length choices* dominate,
2) trailing windows are a strong default,
3) event count windows are the only consistently helpful shape tweak,
4) gap and bucketized windows are consistently worse,
5) calendar alignment effects are small and mixed.

Do not optimize for showing the full grid. Optimize for the fastest possible reader comprehension.

## Naming and labels
- Do **not** use internal labels like `A1`, `A2`, `A3`, `A4` in figure text, legends, or captions.
- Represent a window length set by the explicit hour tuple, e.g. `(1, 6, 24, 48, 168)`.
- Shapes should use the names: `trailing`, `gap1`, `bucket`, `calendar`, `event50`.
- All reported deltas should be paired, on the same test set, fold by fold.

## Primary metric and uncertainty
- Primary metric: **Test ROC AUC**.
- Secondary metric (optional but preferred): **Test PR AUC**.
- Wherever possible, show **paired uncertainty** (CI bars) for deltas:
  - ROC AUC: DeLong paired CI.
  - PR AUC: blocked bootstrap CI (block by hour or day; pick a defensible choice and document it).

If paired CIs are impossible from available artifacts, still generate the plots using point estimates and clearly mark the uncertainty as “across folds only” (mean and fold markers). Prefer paired CIs if you can compute them from predictions.

## Figures to generate (deliverables)

### Figure 1: “Two stage decision” effect size figure (main figure)
Purpose: make “choose lengths first, then shapes” visually obvious.

**Panel A: Length set comparison (trailing only)**
- For each length tuple, show the test ROC AUC delta relative to a chosen reference trailing length tuple.
- Suggested reference: `(1, 6, 24, 48, 168)` trailing (it is the practical default in the draft).
- Show Fold A and Fold B as separate points (or a dumbbell), plus an aggregate marker.
- Add CI bars if you can compute paired CIs for these comparisons.

**Panel B: Shape contrasts within length set**
- For each length tuple, show shape deltas relative to trailing for that same length tuple:
  - `event50 - trailing`
  - `gap1 - trailing`
  - `bucket - trailing`
  - `calendar - trailing`
- Use grouped horizontal dot plus CI bars (forest plot style).
- Facet by length tuple or use color coding per length tuple.
- If space is tight, collapse across length tuples by averaging contrasts and show per length tuple as faint points.

Notes:
- The plot should make it hard to overinterpret tiny differences.
- Avoid heatmaps here.

### Figure 2: “League table” ranking plot (top and bottom specs)
Purpose: give readers an intuitive “what wins / what loses” view without a dense grid.

- Rank specifications by Test ROC AUC (or by delta to the reference).
- Show:
  - top 6 specs
  - bottom 6 specs
- Each row: point estimate with CI (preferred) and fold markers.
- Encode the design choices with minimal ink:
  - length tuple as a short text label,
  - shape as a small symbol or color.

### Figure 3: “Traffic light” significance summary matrix (main or appendix)
Purpose: show the qualitative stability of shape effects without numbers.

- Rows: length tuples.
- Columns: shapes (`event50`, `calendar`, `gap1`, `bucket`), each compared to trailing.
- Each cell is a discrete label:
  - Better (CI excludes 0, positive)
  - Worse (CI excludes 0, negative)
  - No clear difference (CI overlaps 0)
- Optional: print the numeric delta in small font in each cell, but keep readability.

### Figure 4: Protocol schematic (small, main or appendix)
Purpose: make the evaluation constraint instantly obvious.

One simple timeline schematic that communicates:
- out of time split (train < val < test),
- rolling tail folds A/B,
- online safe feature computation: for an event at hour `H`, history uses only events from `< H` (no same hour).

Keep it minimal. A single figure that can be referenced in Problem Setup / Method.

## Optional appendix items (only if cheap)
- A compact “recommended default / avoid list” table:
  - Default: trailing with the best length tuple.
  - Optional: add `event50`.
  - Avoid: `gap1`, `bucket`.
  - Include paired deltas and CIs.

## Output format and placement
- Provide each figure as:
  - `PNG` (for quick iteration), and
  - `PDF` or `SVG` (vector) for final paper.
- Use filenames that match the paper narrative, e.g.:
  - `fig_decision_lengths_then_shapes.(png|pdf)`
  - `fig_league_table_top_bottom.(png|pdf)`
  - `fig_traffic_light_shape_contrasts.(png|pdf)`
  - `fig_protocol_schematic.(png|pdf)`
- Place outputs under `papers/time_aggregation_xgb_v1/figures/` (or provide a bundle that the writer can copy into that folder).

## Short narrative expectation (for captions)
Each figure should be captionable with one sentence that matches the claims in the Goal section.
