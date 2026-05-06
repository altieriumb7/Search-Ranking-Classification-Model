from __future__ import annotations

import json
from datetime import datetime, timezone

from src.config import METRICS_PATH, PIPELINE_PATH, REPORTS_DIR, SUMMARY_PATH
from src.data_loader import load_dataset
from src.evaluation import evaluate_pipeline
from src.model import load_pickle
from src.train_ranker import _summary_markdown, append_run_history, train


def evaluate() -> dict[str, object]:
    if not PIPELINE_PATH.exists():
        return train()

    queries, _documents, qrels = load_dataset()
    test_queries = [query for query in queries if query.split == "test"]
    pipeline = load_pickle(PIPELINE_PATH)
    metrics = evaluate_pipeline(pipeline, test_queries, qrels, top_k=10)
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_type": "evaluate",
        "dataset": "synthetic/demo ranking dataset",
        "train_queries": len([query for query in queries if query.split == "train"]),
        "test_queries": len(test_queries),
        "documents": len(pipeline.documents),
        "ranker_backend": pipeline.ranker.backend,
        "feature_count": len(pipeline.feature_extractor.feature_names),
        "feature_names": pipeline.feature_extractor.feature_names,
        "metrics": metrics,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    append_run_history(output)
    SUMMARY_PATH.write_text(_summary_markdown(output), encoding="utf-8")
    return output


def main() -> None:
    output = evaluate()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
