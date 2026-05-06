from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.config import DOCUMENTS_PATH, QRELS_PATH, QUERIES_PATH


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    split: str


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    body: str
    category: str

    @property
    def text(self) -> str:
        return f"{self.title} {self.body}"


@dataclass(frozen=True)
class Qrel:
    query_id: str
    doc_id: str
    relevance: int


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. This project ships with synthetic/demo data in data/; "
            "restore the CSV files or provide a compatible public ranking dataset."
        )


def load_queries(path: Path = QUERIES_PATH) -> list[Query]:
    _require_file(path)
    with path.open(newline="", encoding="utf-8") as f:
        return [
            Query(row["query_id"], row["query"], row["split"].lower())
            for row in csv.DictReader(f)
        ]


def load_documents(path: Path = DOCUMENTS_PATH) -> list[Document]:
    _require_file(path)
    with path.open(newline="", encoding="utf-8") as f:
        return [
            Document(row["doc_id"], row["title"], row["body"], row["category"])
            for row in csv.DictReader(f)
        ]


def load_qrels(path: Path = QRELS_PATH) -> list[Qrel]:
    _require_file(path)
    with path.open(newline="", encoding="utf-8") as f:
        return [
            Qrel(row["query_id"], row["doc_id"], int(row["relevance"]))
            for row in csv.DictReader(f)
        ]


def load_dataset() -> tuple[list[Query], list[Document], list[Qrel]]:
    queries, documents, qrels = load_queries(), load_documents(), load_qrels()
    validate_dataset(queries, documents, qrels)
    return queries, documents, qrels


def validate_dataset(
    queries: list[Query], documents: list[Document], qrels: list[Qrel]
) -> None:
    query_ids = {query.query_id for query in queries}
    document_ids = {document.doc_id for document in documents}
    allowed_splits = {"train", "test"}

    invalid_splits = sorted({query.split for query in queries if query.split not in allowed_splits})
    if invalid_splits:
        raise ValueError(f"Invalid query split values: {invalid_splits}. Expected train/test.")

    missing_query_refs = sorted({qrel.query_id for qrel in qrels if qrel.query_id not in query_ids})
    if missing_query_refs:
        raise ValueError(f"Qrels reference unknown query_id values: {missing_query_refs[:5]}.")

    missing_doc_refs = sorted({qrel.doc_id for qrel in qrels if qrel.doc_id not in document_ids})
    if missing_doc_refs:
        raise ValueError(f"Qrels reference unknown doc_id values: {missing_doc_refs[:5]}.")

    invalid_relevance = sorted({qrel.relevance for qrel in qrels if qrel.relevance < 0})
    if invalid_relevance:
        raise ValueError(f"Relevance labels must be non-negative, found: {invalid_relevance}.")


def qrels_by_query(qrels: list[Qrel]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = {}
    for qrel in qrels:
        grouped.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    return grouped


def documents_by_id(documents: list[Document]) -> dict[str, Document]:
    return {document.doc_id: document for document in documents}
