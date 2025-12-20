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
- `logs/agents/agent_logs/` — short changelogs per agent iteration.

## 3) Code entrypoints (what to open next)

Minimal “know these files” set:

- `kaggle_clicks/run_baseline_te.py`
  - Single-run pipeline: load sample → add time columns → OOT split → TE → time-agg → XGB → write `runs/<timestamp>_<tag>/...`.
- `kaggle_clicks/time_utils.py`
  - Parsing `hour` → `hour_dt` and constructing strict OOT splits.
- `kaggle_clicks/te.py`
  - Time-aware TE (shifted by hour) + global prior CTR.
- `kaggle_clicks/time_agg.py`
  - Streaming time-aggregation features with online constraint (no same-hour leakage) + Family A variants.
- `kaggle_clicks/run_sweep_family_a.py`
  - Orchestrates multiple comparable runs; writes `runs/sweeps/.../summary.*`.

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

### B) Run a sweep (comparable configs)

- `python -m kaggle_clicks.run_sweep_family_a --sample-pct 1 --sweep-tag familyA_phase01_1pct`

### C) EDA (readable report)

- Read: `docs/EDA_REPORT.md`
- Regenerate: `python human_src/generate_eda_report.py --sample-parquet data/interim/train_sample.parquet --out-md docs/EDA_REPORT.md`

## 7) Agent handoff checklist (end of session)

Before terminating:

- Write a short log to `logs/agents/agent_logs/<timestamp>_codex.md`:
  - what changed, what ran, where outputs are.
- If you created new run folders under `runs/`, ensure they include `report.md` and `metrics.json`.
- If you added large artifacts, ensure `.gitignore` covers them.
- Makse sure that this onboarding guide is up to date and will be useful for the next agent.

