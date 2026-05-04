# Evaluation Summary

Dataset: synthetic/demo ranking dataset.
Ranker backend: `pairwise_logistic_fallback (ModuleNotFoundError)`.

| Model | NDCG@10 | MAP | MRR |
| --- | ---: | ---: | ---: |
| BM25 baseline | 0.9877 | 0.9514 | 1.0000 |
| Learned ranker | 0.9783 | 0.8958 | 1.0000 |

These metrics are computed by `python -m src.train_ranker` or `python -m src.evaluate_ranker` on the held-out test queries.