from __future__ import annotations

import math


def dcg_at_k(relevances: list[int], k: int = 10) -> float:
    return sum(
        ((2**rel - 1) / math.log2(rank + 2))
        for rank, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(ranked_relevances: list[int], k: int = 10) -> float:
    ideal = sorted(ranked_relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(ranked_relevances, k) / ideal_dcg


def average_precision(ranked_relevances: list[int]) -> float:
    hits = 0
    precision_sum = 0.0
    total_relevant = sum(1 for rel in ranked_relevances if rel > 0)
    if total_relevant == 0:
        return 0.0

    for rank, relevance in enumerate(ranked_relevances, start=1):
        if relevance > 0:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / total_relevant


def reciprocal_rank(ranked_relevances: list[int]) -> float:
    for rank, relevance in enumerate(ranked_relevances, start=1):
        if relevance > 0:
            return 1 / rank
    return 0.0


def evaluate_rankings(
    relevance_by_query: dict[str, dict[str, int]],
    ranked_doc_ids_by_query: dict[str, list[str]],
    k: int = 10,
) -> dict[str, float]:
    per_query_ndcg = []
    per_query_ap = []
    per_query_rr = []

    for query_id, qrels in relevance_by_query.items():
        ranked_doc_ids = ranked_doc_ids_by_query.get(query_id, [])
        ranked_relevances = [qrels.get(doc_id, 0) for doc_id in ranked_doc_ids]
        judged_doc_ids = set(qrels)
        missing_judged = [doc_id for doc_id in judged_doc_ids if doc_id not in ranked_doc_ids]
        ranked_relevances.extend(qrels[doc_id] for doc_id in missing_judged)
        per_query_ndcg.append(ndcg_at_k(ranked_relevances, k=k))
        per_query_ap.append(average_precision(ranked_relevances))
        per_query_rr.append(reciprocal_rank(ranked_relevances))

    query_count = max(len(relevance_by_query), 1)
    return {
        f"ndcg@{k}": sum(per_query_ndcg) / query_count,
        "map": sum(per_query_ap) / query_count,
        "mrr": sum(per_query_rr) / query_count,
    }
