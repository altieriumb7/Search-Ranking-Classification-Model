from __future__ import annotations

from dataclasses import dataclass

from src.data_loader import Document, Query, Qrel, documents_by_id, qrels_by_query
from src.features import FeatureExtractor
from src.model import LearningToRankModel


@dataclass
class RankingPipeline:
    documents: list[Document]
    feature_extractor: FeatureExtractor
    ranker: LearningToRankModel

    def rank_baseline(
        self, query: str, candidate_doc_ids: list[str] | None = None, top_k: int = 10
    ) -> list[dict[str, object]]:
        candidates = self._candidate_documents(candidate_doc_ids)
        scored = [
            (self.feature_extractor.bm25.score(query, document.doc_id), document)
            for document in candidates
        ]
        return self._format_ranked(scored, top_k)

    def rank_learned(
        self, query: str, candidate_doc_ids: list[str] | None = None, top_k: int = 10
    ) -> list[dict[str, object]]:
        candidates = self._candidate_documents(candidate_doc_ids)
        X = self.feature_extractor.transform_pairs(query, candidates)
        scores = self.ranker.predict(X)
        scored = list(zip(scores, candidates))
        return self._format_ranked(scored, top_k)

    def _candidate_documents(self, candidate_doc_ids: list[str] | None) -> list[Document]:
        if candidate_doc_ids is None:
            return self.documents
        by_id = documents_by_id(self.documents)
        return [by_id[doc_id] for doc_id in candidate_doc_ids if doc_id in by_id]

    @staticmethod
    def _format_ranked(
        scored: list[tuple[float, Document]], top_k: int
    ) -> list[dict[str, object]]:
        ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
        return [
            {
                "rank": rank,
                "doc_id": document.doc_id,
                "title": document.title,
                "body": document.body,
                "category": document.category,
                "score": float(score),
            }
            for rank, (score, document) in enumerate(ranked, start=1)
        ]


def build_training_matrix(
    queries: list[Query],
    documents: list[Document],
    qrels: list[Qrel],
    feature_extractor: FeatureExtractor,
) -> tuple[list[list[float]], list[int], list[str], list[str], list[int]]:
    by_doc_id = documents_by_id(documents)
    grouped_qrels = qrels_by_query(qrels)
    X: list[list[float]] = []
    y: list[int] = []
    qids: list[str] = []
    doc_ids: list[str] = []

    for query in queries:
        for doc_id, relevance in grouped_qrels.get(query.query_id, {}).items():
            document = by_doc_id[doc_id]
            X.append(feature_extractor.transform_pair(query.text, document))
            y.append(relevance)
            qids.append(query.query_id)
            doc_ids.append(doc_id)

    from src.model import group_sizes

    return X, y, qids, doc_ids, group_sizes(qids)
