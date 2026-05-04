from __future__ import annotations

import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from src.config import RANDOM_SEED


class PairwiseLogisticRanker:
    """Small deterministic pairwise LTR fallback used when XGBoost is unavailable."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        epochs: int = 350,
        l2: float = 0.001,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.means: list[float] = []
        self.stds: list[float] = []
        self.weights: list[float] = []

    def fit(self, X: list[list[float]], y: list[int], qids: list[str]) -> "PairwiseLogisticRanker":
        if not X:
            raise ValueError("Cannot fit ranker with no training rows.")

        self._fit_scaler(X)
        X_scaled = [self._scale(row) for row in X]
        grouped: dict[str, list[int]] = defaultdict(list)
        for idx, qid in enumerate(qids):
            grouped[qid].append(idx)

        pairs: list[tuple[int, int]] = []
        for indexes in grouped.values():
            for left in indexes:
                for right in indexes:
                    if y[left] > y[right]:
                        pairs.append((left, right))

        self.weights = [0.0 for _ in X[0]]
        if not pairs:
            return self

        for _ in range(self.epochs):
            gradients = [self.l2 * weight for weight in self.weights]
            for better_idx, worse_idx in pairs:
                diff = [
                    X_scaled[better_idx][feature_idx] - X_scaled[worse_idx][feature_idx]
                    for feature_idx in range(len(self.weights))
                ]
                margin = sum(weight * value for weight, value in zip(self.weights, diff))
                factor = -1.0 / (1.0 + math.exp(min(margin, 35.0)))
                for feature_idx, value in enumerate(diff):
                    gradients[feature_idx] += factor * value / len(pairs)
            for feature_idx, gradient in enumerate(gradients):
                self.weights[feature_idx] -= self.learning_rate * gradient
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        return [
            sum(weight * value for weight, value in zip(self.weights, self._scale(row)))
            for row in X
        ]

    def _fit_scaler(self, X: list[list[float]]) -> None:
        columns = list(zip(*X))
        self.means = [sum(column) / len(column) for column in columns]
        self.stds = []
        for column, mean in zip(columns, self.means):
            variance = sum((value - mean) ** 2 for value in column) / len(column)
            self.stds.append(math.sqrt(variance) or 1.0)

    def _scale(self, row: list[float]) -> list[float]:
        return [
            (value - mean) / std
            for value, mean, std in zip(row, self.means, self.stds)
        ]


class LearningToRankModel:
    def __init__(self, prefer_xgboost: bool = True) -> None:
        self.prefer_xgboost = prefer_xgboost
        self.backend = "untrained"
        self.model = None

    def fit(
        self,
        X: list[list[float]],
        y: list[int],
        qids: list[str],
        groups: list[int],
    ) -> "LearningToRankModel":
        if self.prefer_xgboost:
            try:
                from xgboost import XGBRanker

                self.model = XGBRanker(
                    objective="rank:ndcg",
                    eval_metric="ndcg@10",
                    n_estimators=80,
                    max_depth=3,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=RANDOM_SEED,
                    tree_method="hist",
                )
                self.model.fit(X, y, group=groups, verbose=False)
                self.backend = "xgboost_xgbranker"
                return self
            except Exception as exc:
                self.backend = f"pairwise_logistic_fallback ({exc.__class__.__name__})"

        self.model = PairwiseLogisticRanker().fit(X, y, qids)
        if self.backend == "untrained":
            self.backend = "pairwise_logistic_fallback"
        return self

    def predict(self, X: list[list[float]]) -> list[float]:
        if self.model is None:
            raise ValueError("Ranker has not been trained.")
        raw_scores = self.model.predict(X)
        return [float(score) for score in raw_scores]


def save_pickle(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> object:
    with path.open("rb") as f:
        return pickle.load(f)


def group_sizes(qids: Iterable[str]) -> list[int]:
    sizes: list[int] = []
    previous_qid = None
    count = 0
    for qid in qids:
        if previous_qid is None:
            previous_qid = qid
        if qid != previous_qid:
            sizes.append(count)
            previous_qid = qid
            count = 0
        count += 1
    if count:
        sizes.append(count)
    return sizes
