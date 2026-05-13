# An Open and Deployable Dashboard for LLM Red-Teaming and Qualitative Evaluation

## 1. Abstract
This repository provides a deployable Streamlit dashboard with deterministic qualitative benchmark reporting and safe public-demo defaults. The current implementation evaluates a learning-to-rank pipeline (BM25 baseline vs learned ranker) rather than a live provider/judge LLM red-team pipeline. The system is packaged for Hugging Face Docker Spaces and supports reproducible local generation of benchmark artifacts without external API calls.

## 2. Introduction
Public technical demos often need to be safely viewable without exposing API keys or triggering paid calls. This project adds a benchmark-first, demo-safe deployment layer so hosted viewers can inspect qualitative evaluation outputs and summary metrics from real generated artifacts.

## 3. System Overview
- UI entrypoint: `app.py`
- Training CLI: `python -m src.train_ranker`
- Evaluation CLI: `python -m src.evaluate_ranker`
- Demo benchmark CLI: `python -m src.generate_demo_benchmark`
- Benchmark case definitions: `benchmarks/qualitative_redteam_cases.yaml`
- Demo benchmark outputs: `reports/demo_benchmark/*`

## 4. Architecture
1. Load dataset (`data/*.csv`).
2. Train learned ranker and persist pipeline (`models/ranking_pipeline.pkl`).
3. Evaluate baseline vs learned metrics (`reports/metrics.json`).
4. Generate deterministic qualitative benchmark from local model artifact and case YAML.
5. Serve ranking + benchmark tabs in Streamlit.

## 5. Deployment Model
- Hugging Face Spaces frontmatter configured in `README.md`.
- Docker runtime:
  - installs `requirements.txt`
  - runs train/evaluate/demo benchmark generation during image build
  - serves Streamlit on port `8501`
- Public-safe defaults:
  - `DEMO_MODE=true`
  - `ALLOW_LIVE_RUNS=false`

## 6. Benchmark Design
The benchmark is deterministic and artifact-backed:
- 6 qualitative cases
- categories:
  - `prompt_injection_resistance`
  - `robustness_to_ambiguous_requests`
  - `policy_compliance`
  - `hallucination_risk_control`
  - `config_and_path_handling`
  - `runtime_error_continuation`
- each case maps to an existing query in bundled dataset and records baseline/learned top-1 judged relevance.

## 7. Qualitative Evaluation
From `reports/demo_benchmark/demo_results.json`:

| Metric | Value |
|---|---:|
| Total cases | 6 |
| Pass | 6 |
| Warning | 0 |
| Fail | 0 |
| Pass rate | 1.0 |

From `reports/metrics.json`:

| Model | NDCG@10 | MAP | MRR |
|---|---:|---:|---:|
| BM25 baseline | 0.9877 | 0.9514 | 1.0000 |
| Learned ranker | 0.9565 | 0.8003 | 1.0000 |

Interpretation: the deterministic qualitative demo cases all pass at top-1 relevance, while offline aggregate metrics still show BM25 outperforming the learned ranker on NDCG@10 and MAP.

## 8. Error Handling and Reproducibility
- Missing model/metrics are surfaced in the dashboard with explicit instructions.
- XGBoost fallback path logs failures explicitly.
- Benchmark generation is deterministic and covered by tests.
- Repro steps are documented in `REPRODUCIBILITY.md`.

## 9. Public Demo Mode and Safety Considerations
- Public mode does not require API keys.
- UI displays key presence only, never key value.
- Session-provided key (if used) is in-memory only.
- No external paid calls are triggered by demo benchmark generation.

## 10. Limitations
1. Current repository scope is ranking evaluation, not full LLM red-team execution against external providers.
2. Demo benchmark is illustrative and deterministic.
3. Hugging Face free storage is ephemeral.
4. Docker runtime validation depends on local Docker daemon availability.

## 11. Conclusion
The project now supports a portfolio-grade public deployment with explicit safety defaults, deterministic qualitative reporting, and reproducible commands. It is suitable for demonstrating deployment hygiene, evaluation UX, and artifact-backed benchmarking.

## 12. Reproducibility Checklist
- Install: `python -m pip install -r requirements.txt`
- Tests: `python -m pytest tests -q`
- Train: `python -m src.train_ranker`
- Evaluate: `python -m src.evaluate_ranker`
- Benchmark: `python -m src.generate_demo_benchmark`
- App: `streamlit run app.py --server.headless true --server.port 8501`
- Docker: `docker build -t llm-redteam-hf .`
