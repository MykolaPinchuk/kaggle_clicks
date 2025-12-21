# TODO

## High priority

1. Speed up feature engineering (dominant runtime)
   - Refactor `kaggle_clicks/time_agg.py` to avoid per-row Python work (process per-hour + per-entity-key aggregation, then map back to rows).
   - Preserve invariants: strict online constraint (hour `H` uses only `<H`), no same-hour leakage, deterministic results.
   - Add a small profiling harness (0.1% / 1%) to quantify before/after.

2. Speed up sweeps (total wall time)
   - Cache/reuse shared intermediates across specs (at minimum: TE outputs and/or hourly entity counts) to avoid recomputing from scratch 40×.
   - Consider limited parallelism across specs (e.g., 2 workers) if RAM allows.

