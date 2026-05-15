---
title: Search Ranking Benchmark Dashboard
emoji: 🧪
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
---

# Search Ranking Demo with Safe Public Benchmark Layer

This repository contains a Streamlit + Python ranking evaluation dashboard (BM25 vs learned ranker) with:
- offline train/evaluate CLI
- deterministic benchmark artifacts for public demo
- optional visitor-triggered live benchmark judging (credit-consuming, guarded)

Scope note: this codebase is a ranking evaluation project, not a full `src/run_redteam.py` orchestration repo.

## Live Demo (Hugging Face Spaces)

Default public-safe mode:
- `DEMO_MODE=true`
- `ALLOW_LIVE_RUNS=false`
- no API key required for visitors

Optional visitor live mode:
- `DEMO_MODE=false`
- `ALLOW_LIVE_RUNS=true`
- `REQUIRE_SESSION_API_KEY=true` (recommended to avoid using host credits)
- add Space Secret `OPENAI_API_KEY`
- configure safeguards (`LIVE_RUN_MAX_CASES`, `LIVE_RUN_COOLDOWN_SECONDS`, `LIVE_RUN_MAX_RUNS_PER_PROCESS`)

## Safe Public Demo Behavior

- Sidebar exposes runtime settings and key presence (never key value).
- Demo benchmark uses deterministic local artifacts.
- Live benchmark button is disabled unless live-mode gates are satisfied.
- Session key input is in-memory only (not persisted).

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Offline commands:

```bash
python -m src.train_ranker
python -m src.evaluate_ranker
python -m src.generate_demo_benchmark
```

## Docker Setup

```bash
docker build -t llm-redteam .
docker run --rm -p 8501:8501 --env-file .env llm-redteam
```

## Hugging Face Spaces Deployment (Docker SDK)

1. Create Space.
2. SDK: `Docker`.
3. Push repository.
4. Set Space Variables:
   - `DEMO_MODE=true`
   - `ALLOW_LIVE_RUNS=false`
   - `REQUIRE_SESSION_API_KEY=false`
   - `DEFAULT_CONFIG_PATH=evals/config.yaml`
   - `REPORTS_DIR=reports`
   - `BENCHMARK_MODE=demo`
   - `LIVE_RUN_MAX_CASES=3`
   - `LIVE_RUN_COOLDOWN_SECONDS=60`
   - `LIVE_RUN_MAX_RUNS_PER_PROCESS=3`
5. Optional Space Secret:
   - `OPENAI_API_KEY`

HF free storage is ephemeral; runtime-generated files may be lost after restart.

## Benchmark

Cases source:
- `benchmarks/qualitative_redteam_cases.yaml`

Deterministic demo artifacts:
- `reports/demo_benchmark/demo_results.json`
- `reports/demo_benchmark/demo_summary.csv`
- `reports/demo_benchmark/qualitative_case_gallery.md`

Generate demo benchmark:

```bash
python -m src.generate_demo_benchmark
```

Live benchmark:
- Run from UI: **Benchmark & Qualitative Evaluation** -> **Run live benchmark (may consume API credits)**
- Writes timestamped output: `reports/live_benchmark/<timestamp>/`
- Continues per-case on failures and records `runtime.error`

## Security

- Do not commit `.env` or API keys.
- `OPENAI_API_KEY` is never displayed.
- Public mode defaults to no paid calls.
- Live runs require explicit flags and key.
- If `REQUIRE_SESSION_API_KEY=true`, only visitor session keys are accepted.

## Limitations

- Ranking-focused project; not full provider/judge orchestration stack.
- Demo benchmark is deterministic and illustrative.
- Docker runtime verification depends on local daemon availability.

## CV / Portfolio Positioning

Demonstrates:
- deployable Streamlit MLOps packaging
- deterministic + live qualitative benchmark UX
- safe public-hosting controls
- reproducible artifact-backed reporting
