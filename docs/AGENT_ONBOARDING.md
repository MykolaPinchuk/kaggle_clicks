# Agent Onboarding Guide (kaggle_clicks)

This repo is developed via successive short-lived agents. This guide is the “standard onboarding packet” an agent should read to become productive quickly and safely.

## 0) Non-negotiable rules (read first)

- Read and follow `agents.md` (repo-wide operating rules).
- Do not touch git workflows (no commits, no resets). Only update `.gitignore` if needed.
- Keep runs fast: default to small samples; avoid commands expected to run >5 minutes unless explicitly authorized.
- Store agent summaries in `logs/agents/agent_logs/` (one markdown file per agent session).

## 1) What we are building (the one-paragraph mental model)

We train CTR models on Avazu with **strict out-of-time (OOT)** evaluation. Features include **time-aware target encoding (TE)** and **online-safe entity history features** (“time aggregation”) computed so that events at hour `H` only use information from `<H` (no same-hour leakage). The research goal is to systematically compare **time-window constructions** (A.1) and **time-window length sets** (A.2) while keeping everything else fixed.

## 2) Read these docs in this order

1. `README.md` — high-level usage + entrypoints.
2. `docs/DATA.md` — where `train.csv` must live.
3. `docs/TIME_AGG_EXPERIMENTS.md` — experiment taxonomy and what “Family A” means.
4. `docs/plan_for_working_paper_results.md` — current execution plan for paper-quality results (scope guardrails, stability protocol, inference).
5. `docs/FAMILY_A_RESULTS.md` — pointers to the latest sweep outputs and promotion runs.
6. `docs/EDA_REPORT.md` — dataset properties relevant to modeling (drift, long tails, cold-start).

Optional historical context:
- `logs/agents/agent_logs/` — short changelogs per agent iteration. Avoid reading anything except the last entry.

If you are at the stage of reviewing results for a paper draft, do not read code and code-related docs. Read only paper_draft directory. Read more files only if actually needed.

## 3) Code entrypoints (what to open next if need to run code)

Minimal “know these files” set:

- `kaggle_clicks/run_baseline_te.py`
  - Single-run pipeline: load sample → add time columns → OOT split → TE → time-agg → XGB → write `runs/<timestamp>_<tag>/...`.
  - Supports fractional smoke runs via `--sample-frac` (e.g. `0.001` = 0.1%) and prediction export via `--export-preds`.
- `kaggle_clicks/time_utils.py`
  - Parsing `hour` → `hour_dt` and constructing strict OOT splits.
- `kaggle_clicks/te.py`
  - Time-aware TE (shifted by hour) + global prior CTR.
- `kaggle_clicks/time_agg.py`
  - Streaming time-aggregation features with online constraint (no same-hour leakage) + Family A variants.
- `kaggle_clicks/run_sweep_family_a.py`
  - Orchestrates multiple comparable runs; writes `runs/sweeps/.../summary.*`.
- `kaggle_clicks/run_sweep_family_a_full_grid.py`
  - Runs the full “paper grid” (4 length sets × 5 shapes) and supports rolling-tail Fold A/B with resume/skip; writes per-run logs under `runs/sweeps/.../logs/`.

## 4) Key invariants (don’t break these)

1. **Strict OOT splits**: `train < val < test` chronologically; assert it.
2. **Online simulation constraint**: for a row at hour `H`, time-agg features must only use history from `<H`.
3. **Comparability**: when doing sweeps, keep model params / TE params fixed unless the change is explicitly part of the sweep.
4. **Scope discipline**: working paper is currently scoped to Family A phases A.1 + A.2 only; do not drift into A.3/A.4.

## 5) Where results live (and how to read them)

- Per run: `runs/<timestamp>_<tag>/report.md`
- Sweeps: `runs/sweeps/<timestamp>_<tag>_s<pct>pct/summary.md` and `summary.csv`
- Research pointers:
  - `docs/FAMILY_A_RESULTS.md`
  - `docs/SMOKE_FAMILY_A_RESULTS.md` (sanity runs; not paper results)

## 6) Standard workflows

### A) Run a single experiment (fast)

- `python -m kaggle_clicks.run_baseline_te --sample-pct 1 --run-tag my_test`

Environment note:
- The checked-in `.venv/` may be stale/missing deps; if `python -m kaggle_clicks...` fails on imports (e.g. `xgboost`, `pyarrow`, `sklearn`), use the system `python` (pyenv) or install deps into the active env.

### A0) Precompute TE once (recommended for sweeps)

If you plan to run many comparable specs on the same sample parquet, precompute TE once and reuse it across all runs:

- `python -m kaggle_clicks.precompute_te --sample-parquet data/interim/train_sample_10pct.parquet --out-parquet data/interim/te_cache/train_sample_10pct_te_m100.parquet --m 100`

Then pass it to runs/sweeps via `--te-parquet ...` to skip TE recomputation.

Useful options:
- Rolling-tail folds: `--rolling-tail-fold A` or `--rolling-tail-fold B`
- Export predictions: `--export-preds` (writes `preds_val.parquet` and `preds_test.parquet`)
- Fractional smoke runs: `--sample-frac 0.001` (0.1%) + set `--sample-parquet` to a dedicated path

### B) Run a sweep (comparable configs)

- `python -m kaggle_clicks.run_sweep_family_a --sample-pct 1 --sweep-tag familyA_phase01_1pct`

Useful options:
- Rolling-tail folds across runs: `--rolling-tail`
- Reuse TE cache: `--te-parquet data/interim/te_cache/<...>.parquet`
- Export predictions for inference: `--export-preds`
- Enable Phase 2 (shapes): `--enable-phase2 --phase2-base-windows 1 6 24 48 168`

### C) Postprocess inference (DeLong + block bootstrap)

- `python -m kaggle_clicks.postprocess_sweep_inference --sweep-dir runs/sweeps/<...> --baseline-run-id R3 --split test --bootstrap-reps 200 --seed 42`

### D) Paper-grid sweep (long)

- Full grid (20 specs × Fold A/B = 40 runs):
  - `python -m kaggle_clicks.run_sweep_family_a_full_grid --sample-pct 10 --sample-parquet data/interim/train_sample_10pct.parquet --sweep-tag paper_full_grid_10pct --rolling-tail --export-preds --max-wall-seconds 57600 --per-run-timeout-seconds 7200`
  - With TE cache: add `--te-parquet data/interim/te_cache/train_sample_10pct_te_m100.parquet`
- Monitor:
  - progress: `runs/sweeps/<...>/sweep.log`
  - per-run stdout/stderr: `runs/sweeps/<...>/logs/`
  - rolling summary: `runs/sweeps/<...>/summary.md`
- Baseline run id for inference in the paper grid is typically `A3_trailing`.

Note: For long runs, avoid running inside a VSCode/Chrome-integrated terminal if you see OOM issues; use a plain terminal/session and keep memory-heavy apps minimal.

### E) EDA (readable report)

- Read: `docs/EDA_REPORT.md`
- Regenerate: `python human_src/generate_eda_report.py --sample-parquet data/interim/train_sample.parquet --out-md docs/EDA_REPORT.md`

## 6.1) Resource troubleshooting (CPU/RAM)

- If you see low CPU utilization during sweeps, it’s usually because feature engineering is the bottleneck; consider running multiple specs concurrently via `kaggle_clicks/run_sweep_family_a_full_grid.py --max-parallel-runs ...` and keep per-run `--n-jobs` low to avoid oversubscription.
- If you hit RAM OOMs on large samples (notably A4 configs), prefer `--max-parallel-runs 1` and reduce XGBoost threads (e.g. `--n-jobs 1` or `2`). The peak RAM driver is time-aggregation feature construction.
- For unattended runs, use the built-in memory throttle:
  - `--ram-budget-gib 25 --max-parallel-runs 2 --n-jobs 1`
  - The sweep log records `PAUSE_MEM ...` lines when it temporarily stops launching new runs to avoid OOM/swap thrash.
- If you want to keep other apps stable (e.g. Chrome), increase the cushion:
  - `--mem-safety-margin-gib 8` (default is `6`)

## 7) Current state / handoff pointers

- Latest “what to do next” for the full paper grid is documented in `logs/agents/agent_logs/2025-12-20T20-49_codex.md`.
- v1 paper artifacts are consolidated under `paper_draft/v1/README.md` (recommended entrypoint for writing).
- Inference utilities live in:
  - `kaggle_clicks/inference_auc.py` (paired DeLong for ROC-AUC)
  - `kaggle_clicks/inference_bootstrap.py` (paired block bootstrap for PR-AUC)
  - `kaggle_clicks/postprocess_sweep_inference.py` (sweep → `inference_vs_baseline.csv`)
  - `kaggle_clicks/postprocess_paper_grid_contrasts.py` (shape-vs-trailing contrasts within a length set)
  - `kaggle_clicks/postprocess_paper_grid_te_vs_note.py` (TE vs no-TE comparisons)

## 8) Paper-writing agent quickstart (keep context small)

If your job is to write a preliminary paper (and not to re-run experiments), avoid reading raw sweep logs / run dirs.

Minimal read list:
- `paper_draft/v1/README.md` (manifest + where to look)
- `paper_draft/v1/10pct_paper_grid/master_results.md` (TE main table)
- `paper_draft/v1/10pct_paper_grid_noTE/master_results.md` (no-TE main table)
- `paper_draft/v1/10pct_te_lift/te_vs_timeagg_summary.md` (headline TE-only → time-agg lift)
- `paper_draft/v1/eda/EDA_REPORT.md` (dataset drift + long-tail context)

Optional:
- `paper_draft/v1/10pct_paper_grid/contrasts_vs_trailing_test.md`
- `paper_draft/v1/10pct_paper_grid_noTE/contrasts_vs_trailing_test.md`
- `paper_draft/v1/combined/te_vs_note_summary.md`

## 9) Agent handoff checklist (end of session)

Before terminating:

- Write a short log to `logs/agents/agent_logs/<timestamp>_codex.md`:
  - what changed, what ran, where outputs are.
- If you created new run folders under `runs/`, ensure they include `report.md` and `metrics.json`.
- If you added large artifacts, ensure `.gitignore` covers them.
- Makse sure that this onboarding guide is up to date and will be useful for the next agent.
