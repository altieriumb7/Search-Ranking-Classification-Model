---
title: LLM Red Team Evaluation Dashboard
emoji: 🧪
colorFrom: red
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
---

# Search Ranking Demo with Safe Public Benchmark Layer

This repository contains a **Streamlit + Python ranking evaluation dashboard** (BM25 vs learned ranker) with:
- offline train/evaluate CLI
- public-safe demo mode by default
- deterministic qualitative benchmark artifacts for hosted demos

Important scope note: this codebase is currently a search-ranking evaluation project, not an API-calling LLM red-team runner.

## Live Demo (Hugging Face Spaces)

Hosted mode is designed to be safe by default:
- `DEMO_MODE=true`
- `ALLOW_LIVE_RUNS=false`
- no API key required for public demo browsing

The app exposes benchmark/demo artifacts and ranking behavior without external paid model calls.

## Safe Public Demo Behavior

- Sidebar shows active runtime settings (`DEMO_MODE`, `ALLOW_LIVE_RUNS`, key presence, config/report paths).
- Public banner explains that live paid calls are disabled.
- Demo benchmark data is generated deterministically from local artifacts.
- User-entered keys (if ever used in live mode) are session-only and never persisted.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Offline pipeline commands:

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

`Dockerfile` generates train/eval + demo benchmark artifacts during image build.

## Hugging Face Spaces Deployment (Docker SDK)

1. Create Space.
2. SDK: `Docker`.
3. Push this repository.
4. Set Space Variables:
   - `DEMO_MODE=true`
   - `ALLOW_LIVE_RUNS=false`
   - `DEFAULT_CONFIG_PATH=evals/config.yaml`
   - `REPORTS_DIR=reports`
   - `BENCHMARK_MODE=demo`
5. Optional Space Secret:
   - `OPENAI_API_KEY` (not required for current deterministic demo path)

HF storage caveat: free-tier storage is ephemeral. Generated runtime files may not persist across restarts.

## Benchmark Section

Benchmark source:
- `benchmarks/qualitative_redteam_cases.yaml`

Generated artifacts:
- `reports/demo_benchmark/demo_results.json`
- `reports/demo_benchmark/demo_summary.csv`
- `reports/demo_benchmark/qualitative_case_gallery.md`

Generate deterministically (no external API calls):

```bash
python -m src.generate_demo_benchmark
```

Metrics shown:
- total cases
- pass/warning/fail counts
- pass rate
- category coverage/breakdown

## Security

- Do not commit `.env` or API keys.
- Public demo mode is default.
- `OPENAI_API_KEY` is never displayed in UI.
- Session-provided key (when used) is not persisted.

## Full Live Evaluation

This repository currently does **not** include a provider/judge API execution pipeline (`src/run_redteam.py` style flow is not present here).  
Live model-cost evaluations therefore require adding that pipeline first.

## Limitations

- Project scope is ranking evaluation, not full LLM red-team orchestration.
- Demo benchmark outputs are illustrative and deterministic.
- HF free storage is ephemeral.

## CV / Portfolio Positioning

This project demonstrates:
- deployable Streamlit MLOps packaging
- deterministic demo benchmarking for public hosting safety
- reproducible offline evaluation workflow
- transparent runtime configuration and artifact-backed reporting
