from __future__ import annotations

from src.data_loader import Query, Qrel, qrels_by_query
from src.metrics import evaluate_rankings
from src.ranking import RankingPipeline


def candidate_ids_for_query(qrels: list[Qrel], query_id: str) -> list[str]:
    return [qrel.doc_id for qrel in qrels if qrel.query_id == query_id]


def evaluate_pipeline(
    pipeline: RankingPipeline,
    queries: list[Query],
    qrels: list[Qrel],
    top_k: int = 10,
) -> dict[str, dict[str, float]]:
    judged = qrels_by_query(qrels)
    relevant_judgments = {
        query.query_id: judged[query.query_id]
        for query in queries
        if query.query_id in judged
    }

    baseline_rankings: dict[str, list[str]] = {}
    learned_rankings: dict[str, list[str]] = {}
    for query in queries:
        candidate_doc_ids = list(judged.get(query.query_id, {}).keys())
        baseline_rankings[query.query_id] = [
            row["doc_id"]
            for row in pipeline.rank_baseline(query.text, candidate_doc_ids, top_k=top_k)
        ]
        learned_rankings[query.query_id] = [
            row["doc_id"]
            for row in pipeline.rank_learned(query.text, candidate_doc_ids, top_k=top_k)
        ]

    return {
        "bm25_baseline": evaluate_rankings(relevant_judgments, baseline_rankings, k=top_k),
        "learned_ranker": evaluate_rankings(relevant_judgments, learned_rankings, k=top_k),
    }
