# Evaluation Summary

Dataset: synthetic/demo ranking dataset.
Ranker backend: `xgboost_xgbranker`.

| Model | NDCG@10 | MAP | MRR |
| --- | ---: | ---: | ---: |
| BM25 baseline | 0.9877 | 0.9514 | 1.0000 |
| Learned ranker | 0.9565 | 0.8003 | 1.0000 |

These metrics are computed by `python -m src.train_ranker` or `python -m src.evaluate_ranker` on the held-out test queries.