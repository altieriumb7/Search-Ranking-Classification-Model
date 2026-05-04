from __future__ import annotations

import math
from collections import Counter

from src.data_loader import Document
from src.preprocessing import normalize_text, tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tokens: dict[str, list[str]] = {}
        self.doc_term_freqs: dict[str, Counter[str]] = {}
        self.doc_lengths: dict[str, int] = {}
        self.doc_freqs: Counter[str] = Counter()
        self.avg_doc_length = 0.0
        self.num_docs = 0

    def fit(self, documents: list[Document]) -> "BM25Index":
        self.doc_tokens.clear()
        self.doc_term_freqs.clear()
        self.doc_lengths.clear()
        self.doc_freqs.clear()
        self.num_docs = len(documents)

        total_length = 0
        for document in documents:
            tokens = tokenize(document.text)
            term_freqs = Counter(tokens)
            self.doc_tokens[document.doc_id] = tokens
            self.doc_term_freqs[document.doc_id] = term_freqs
            self.doc_lengths[document.doc_id] = len(tokens)
            total_length += len(tokens)
            self.doc_freqs.update(term_freqs.keys())

        self.avg_doc_length = total_length / max(self.num_docs, 1)
        return self

    def idf(self, term: str) -> float:
        df = self.doc_freqs.get(term, 0)
        return math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_id: str) -> float:
        query_terms = tokenize(query)
        if not query_terms or doc_id not in self.doc_term_freqs:
            return 0.0

        doc_len = self.doc_lengths[doc_id]
        term_freqs = self.doc_term_freqs[doc_id]
        score = 0.0
        for term in query_terms:
            tf = term_freqs.get(term, 0)
            if tf == 0:
                continue
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_len / max(self.avg_doc_length, 1e-9)
            )
            score += self.idf(term) * ((tf * (self.k1 + 1)) / denominator)
        return score


class FeatureExtractor:
    feature_names = [
        "bm25",
        "term_overlap_count",
        "term_overlap_ratio",
        "query_coverage",
        "title_overlap_ratio",
        "exact_phrase_title",
        "exact_phrase_body",
        "query_length",
        "document_length",
        "average_query_idf",
    ]

    def __init__(self) -> None:
        self.bm25 = BM25Index()
        self._documents_by_id: dict[str, Document] = {}
        self._title_tokens: dict[str, set[str]] = {}
        self._body_norm: dict[str, str] = {}
        self._title_norm: dict[str, str] = {}

    def fit(self, documents: list[Document]) -> "FeatureExtractor":
        self._documents_by_id = {document.doc_id: document for document in documents}
        self._title_tokens = {
            document.doc_id: set(tokenize(document.title)) for document in documents
        }
        self._title_norm = {
            document.doc_id: normalize_text(document.title) for document in documents
        }
        self._body_norm = {
            document.doc_id: normalize_text(document.body) for document in documents
        }
        self.bm25.fit(documents)
        return self

    def transform_pair(self, query: str, document: Document) -> list[float]:
        query_tokens = tokenize(query)
        query_unique = set(query_tokens)
        doc_tokens = set(self.bm25.doc_tokens.get(document.doc_id, []))
        title_tokens = self._title_tokens.get(document.doc_id, set())
        overlap = query_unique & doc_tokens
        title_overlap = query_unique & title_tokens
        normalized_query = normalize_text(query)

        average_idf = (
            sum(self.bm25.idf(term) for term in query_unique) / len(query_unique)
            if query_unique
            else 0.0
        )

        return [
            self.bm25.score(query, document.doc_id),
            float(len(overlap)),
            len(overlap) / max(len(query_unique), 1),
            sum(1 for term in query_tokens if term in doc_tokens) / max(len(query_tokens), 1),
            len(title_overlap) / max(len(query_unique), 1),
            1.0 if normalized_query and normalized_query in self._title_norm[document.doc_id] else 0.0,
            1.0 if normalized_query and normalized_query in self._body_norm[document.doc_id] else 0.0,
            float(len(query_tokens)),
            float(self.bm25.doc_lengths.get(document.doc_id, 0)),
            average_idf,
        ]

    def transform_pairs(
        self, query: str, documents: list[Document]
    ) -> list[list[float]]:
        return [self.transform_pair(query, document) for document in documents]
