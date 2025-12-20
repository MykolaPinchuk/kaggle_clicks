# Family A Results (OOT) — Current Best Snapshot

This file records the latest **comparable** results for Family A (entity-level time aggregation) using the repo’s strict OOT protocol.

## Where to look for details

Every run has a detailed report at `runs/<timestamp>_<tag>/report.md` (splits, config, metrics, top features).

## Phase 1 (1% sample): trailing window set sweep

Sweep summary:

- `runs/sweeps/20251218_034459_familyA_phase01_1pct_s1pct/summary.md`

Top result in this sweep:

- R3 (A3 windows `{1,6,24,48,168}`): `runs/20251218_034629_familyA_phase01_1pct_R3/report.md`

## Phase 2 (1% sample): window shape sweep (using A3 base windows)

Sweep summary:

- `runs/sweeps/20251218_035018_familyA_phase2_1pct_s1pct/summary.md`

Top result in this sweep:

- R9 (A3 + event windows last50imps): `runs/20251218_035741_familyA_phase2_1pct_R9/report.md`

## Promotion check (5% sample): TE vs A3 vs A3+event(50)

All three runs below use the same model params (`n_estimators=800`, `learning_rate=0.05`, etc.) and are directly comparable:

- R0 (TE-only): `runs/20251218_035935_familyA_5pct_R0_TEonly/report.md`
- R3 (TE + A3 trailing `{1,6,24,48,168}`): `runs/20251218_040020_familyA_5pct_R3_A3trailing/report.md`
- R9 (TE + A3 trailing + event last50imps): `runs/20251218_041038_familyA_5pct_R9_A3_event50/report.md`

