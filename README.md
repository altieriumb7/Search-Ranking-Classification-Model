# Search Ranking Classification Model

Portfolio/demo implementation of a learning-to-rank system for search results.
It compares a classical BM25 keyword baseline with a supervised learning-to-rank
model on a compact judged ranking dataset.

The included dataset is synthetic/demo data. It is designed to make the project
easy to run locally and to demonstrate the full information retrieval workflow;
it is not a production-scale benchmark and does not imply Google-scale quality.

## Project Structure

```text
data/       synthetic queries, documents, and relevance judgments
models/     trained pipeline artifacts
reports/    computed evaluation outputs
src/        ranking pipeline modules and scripts
tests/      unit tests
app.py      Streamlit demo
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The training code uses `xgboost.XGBRanker` when XGBoost is installed. If XGBoost
is not available, it falls back to a deterministic pairwise logistic ranker so
the pipeline can still run on the demo dataset.

## Train

```bash
python -m src.train_ranker
```

This saves:

- `models/ranking_pipeline.pkl`
- `reports/metrics.json`
- `reports/evaluation_summary.md`
- `reports/run_history.jsonl` (append-only history of train/evaluate runs)

## Evaluate

```bash
python -m src.evaluate_ranker
```

Metrics are computed on the held-out synthetic test queries:

- NDCG@10
- MAP
- MRR

Do not edit the reported values by hand. Re-run the command above to refresh
the metrics after changing features, model code, or data.

Latest computed local results:

| Model | NDCG@10 | MAP | MRR |
| --- | ---: | ---: | ---: |
| BM25 baseline | 0.9877 | 0.9514 | 1.0000 |
| Learned ranker | 0.9783 | 0.8958 | 1.0000 |

This run used `pairwise_logistic_fallback (ModuleNotFoundError)` because
XGBoost is not installed in the current environment. Installing requirements
and re-running evaluation may produce different computed values with
`xgboost.XGBRanker`.

## Demo

```bash
streamlit run app.py
```

The demo includes:

- query input box
- at least 3 preloaded example queries
- top-k ranked documents
- BM25 and learned ranker side-by-side
- relevance/ranking scores for judged demo queries
- NDCG@10, MAP, and MRR summary
- bar chart comparing baseline and learned ranker performance

## Tests

```bash
python -m unittest discover tests
```

The tests cover data loading, feature extraction, ranking metrics, and prediction
output format.

## Limitations

- The bundled data is synthetic/demo data, not MS MARCO, LETOR, or production
  search logs.
- The supervised model has few training queries, so results are illustrative.
- Candidate generation is constrained to judged demo documents during offline
  evaluation. For arbitrary demo queries, the app ranks all bundled documents.
- The fallback pairwise ranker is included for reproducibility in minimal
  Python environments; install XGBoost to use the LambdaMART-style `XGBRanker`.
