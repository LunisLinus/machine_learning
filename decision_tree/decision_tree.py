from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class _Node:
    prediction: int
    class_counts: np.ndarray
    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


class CustomDecisionTreeClassifier:

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ) -> None:
        if criterion not in {"gini", "chi2"}:
            raise ValueError("criterion должен быть 'gini' или 'chi2'")
        if min_samples_split < 2:
            raise ValueError("min_samples_split должен быть >= 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf должен быть >= 1")

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.n_features_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.root_: Optional[_Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CustomDecisionTreeClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X должен быть 2D-массивом формы (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y должен быть 1D-массивом")
        if len(X) != len(y):
            raise ValueError("Количество объектов в X и y должно совпадать")
        if len(X) == 0:
            raise ValueError("Пустая выборка")

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_features_ = X.shape[1]
        self.root_ = self._build_tree(X, y_encoded, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root_ is None or self.classes_ is None:
            raise RuntimeError("Сначала вызовите fit")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X должен быть 2D-массивом")

        preds = [self._predict_one(row, self.root_) for row in X]
        return self.classes_[np.asarray(preds, dtype=int)]

    def _predict_one(self, row: np.ndarray, node: _Node) -> int:
        while not node.is_leaf:
            if row[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        class_counts = np.bincount(y, minlength=len(self.classes_))
        prediction = int(np.argmax(class_counts))
        node = _Node(prediction=prediction, class_counts=class_counts)

        if self._should_stop(y, depth):
            return node

        split = self._best_split(X, y)
        if split is None:
            return node

        feature_index, threshold, left_mask = split
        right_mask = ~left_mask
        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _should_stop(self, y: np.ndarray, depth: int) -> bool:
        if len(y) < self.min_samples_split:
            return True
        if np.unique(y).size == 1:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        return False

    def _best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Optional[tuple[int, float, np.ndarray]]:
        n_samples, n_features = X.shape
        best_feature = None
        best_threshold = None
        best_left_mask = None

        if self.criterion == "gini":
            best_score = np.inf
        else:
            best_score = -np.inf

        for feature_idx in range(n_features):
            values = X[:, feature_idx]
            unique_values = np.unique(values)
            if unique_values.size < 2:
                continue

            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = values <= threshold
                left_n = int(left_mask.sum())
                right_n = n_samples - left_n

                if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[~left_mask]

                if self.criterion == "gini":
                    score = self._weighted_gini(y_left, y_right)
                    better = score < best_score
                else:
                    score = self._chi2_score(y_left, y_right)
                    better = score > best_score

                if better:
                    best_score = score
                    best_feature = feature_idx
                    best_threshold = float(threshold)
                    best_left_mask = left_mask

        if best_feature is None:
            return None
        return best_feature, best_threshold, best_left_mask

    def _gini(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        counts = np.bincount(y, minlength=len(self.classes_))
        probs = counts / len(y)
        return 1.0 - np.sum(probs * probs)

    def _weighted_gini(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        n_total = len(y_left) + len(y_right)
        return (len(y_left) / n_total) * self._gini(y_left) + (
            len(y_right) / n_total
        ) * self._gini(y_right)

    def _chi2_score(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        counts_left = np.bincount(y_left, minlength=len(self.classes_)).astype(float)
        counts_right = np.bincount(y_right, minlength=len(self.classes_)).astype(float)
        observed = np.vstack([counts_left, counts_right])

        row_sums = observed.sum(axis=1, keepdims=True)
        col_sums = observed.sum(axis=0, keepdims=True)
        total = observed.sum()
        if total == 0:
            return 0.0

        expected = row_sums @ col_sums / total
        mask = expected > 0.0
        chi2 = np.sum(((observed[mask] - expected[mask]) ** 2) / expected[mask])
        return float(chi2)
