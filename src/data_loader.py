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
    return load_queries(), load_documents(), load_qrels()


def qrels_by_query(qrels: list[Qrel]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = {}
    for qrel in qrels:
        grouped.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    return grouped


def documents_by_id(documents: list[Document]) -> dict[str, Document]:
    return {document.doc_id: document for document in documents}
