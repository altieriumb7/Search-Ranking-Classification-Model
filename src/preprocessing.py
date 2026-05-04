from __future__ import annotations

import re

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "vs",
    "with",
    "your",
}


def normalize_text(text: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(text.lower()))


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    if remove_stopwords:
        return [token for token in tokens if token not in STOPWORDS]
    return tokens
