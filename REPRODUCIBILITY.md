# Reproducibility Checklist

## Runtime
- Python: 3.12

## Install
```bash
python -m pip install -r requirements.txt
```

## Test
```bash
python -m pytest tests -q
```

## Streamlit
```bash
streamlit run app.py --server.headless true --server.port 8501
```

## Docker
```bash
docker build -t llm-redteam-hf .
docker run --rm -p 8501:8501 -e DEMO_MODE=true -e ALLOW_LIVE_RUNS=false llm-redteam-hf
```

## Hugging Face Spaces Settings
- SDK: Docker
- `DEMO_MODE=true`
- `ALLOW_LIVE_RUNS=false`
- `DEFAULT_CONFIG_PATH=evals/config.yaml`
- `REPORTS_DIR=reports`
- `BENCHMARK_MODE=demo`
- Optional Secret: `OPENAI_API_KEY`

## Benchmark Generation
```bash
python -m src.generate_demo_benchmark
```

## Expected Generated Files
- `models/ranking_pipeline.pkl`
- `reports/metrics.json`
- `reports/evaluation_summary.md`
- `reports/demo_benchmark/demo_results.json`
- `reports/demo_benchmark/demo_summary.csv`
- `reports/demo_benchmark/qualitative_case_gallery.md`

## Environment Variables
See `.env.example`.

## Known Limitations
- Public demo uses deterministic local artifacts.
- No provider/judge API runner exists in this repository yet.
- Docker runtime verification depends on local Docker daemon availability.
