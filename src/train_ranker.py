from __future__ import annotations

import json
from datetime import datetime, timezone

from src.config import METRICS_PATH, MODELS_DIR, PIPELINE_PATH, REPORTS_DIR, SUMMARY_PATH
from src.data_loader import load_dataset
from src.evaluation import evaluate_pipeline
from src.features import FeatureExtractor
from src.model import LearningToRankModel, save_pickle
from src.ranking import RankingPipeline, build_training_matrix


def train() -> dict[str, object]:
    queries, documents, qrels = load_dataset()
    train_queries = [query for query in queries if query.split == "train"]
    test_queries = [query for query in queries if query.split == "test"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_extractor = FeatureExtractor().fit(documents)
    X, y, qids, _doc_ids, groups = build_training_matrix(
        train_queries, documents, qrels, feature_extractor
    )

    ranker = LearningToRankModel(prefer_xgboost=True).fit(X, y, qids, groups)
    pipeline = RankingPipeline(documents, feature_extractor, ranker)
    save_pickle(pipeline, PIPELINE_PATH)

    metrics = evaluate_pipeline(pipeline, test_queries, qrels, top_k=10)
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "synthetic/demo ranking dataset",
        "train_queries": len(train_queries),
        "test_queries": len(test_queries),
        "documents": len(documents),
        "ranker_backend": ranker.backend,
        "metrics": metrics,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    SUMMARY_PATH.write_text(_summary_markdown(output), encoding="utf-8")
    return output


def _summary_markdown(output: dict[str, object]) -> str:
    metrics = output["metrics"]
    baseline = metrics["bm25_baseline"]
    learned = metrics["learned_ranker"]
    return "\n".join(
        [
            "# Evaluation Summary",
            "",
            "Dataset: synthetic/demo ranking dataset.",
            f"Ranker backend: `{output['ranker_backend']}`.",
            "",
            "| Model | NDCG@10 | MAP | MRR |",
            "| --- | ---: | ---: | ---: |",
            (
                f"| BM25 baseline | {baseline['ndcg@10']:.4f} | "
                f"{baseline['map']:.4f} | {baseline['mrr']:.4f} |"
            ),
            (
                f"| Learned ranker | {learned['ndcg@10']:.4f} | "
                f"{learned['map']:.4f} | {learned['mrr']:.4f} |"
            ),
            "",
            "These metrics are computed by `python -m src.train_ranker` or "
            "`python -m src.evaluate_ranker` on the held-out test queries.",
        ]
    )


def main() -> None:
    output = train()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
