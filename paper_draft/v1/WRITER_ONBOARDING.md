# Writer Onboarding (v1)

Goal: write the preliminary paper and answer reviewer questions without loading unnecessary repo context.

## Read (in order)

1) `paper_draft/v1/README.md` (artifact manifest; follow links)
2) Main result tables:
   - `paper_draft/v1/10pct_paper_grid/master_results.md` (TE+time-agg)
   - `paper_draft/v1/10pct_paper_grid_noTE/master_results.md` (no-TE time-agg-only)
3) Headline comparison:
   - `paper_draft/v1/10pct_te_lift/te_vs_timeagg_summary.md` (TE-only → time-agg lift)
4) Context for narrative:
   - `paper_draft/v1/eda/EDA_REPORT.md` (drift + long-tail + cold-start)

## Optional (only if needed for a specific paragraph)

- Shape-vs-trailing contrasts:
  - `paper_draft/v1/10pct_paper_grid/contrasts_vs_trailing_test.md`
  - `paper_draft/v1/10pct_paper_grid_noTE/contrasts_vs_trailing_test.md`
- TE vs no-TE per-spec deltas:
  - `paper_draft/v1/combined/te_vs_note_summary.md`

## Avoid (unless debugging)

- `runs/` and `runs/sweeps/*/sweep.log` (too much detail for writing; use the tables instead).

