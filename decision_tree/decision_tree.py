from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class _TreeNode:
    prediction: int
    proba: np.ndarray
    feature_index: int | None = None
    threshold: float | None = None
    left: "_TreeNode | None" = None
    right: "_TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ) -> None:
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be >= 1 or None.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2.")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.n_classes_: int | None = None
        self.classes_: np.ndarray | None = None
        self.root_: _TreeNode | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        y = y.reshape(-1)

        if x.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if x.shape[0] == 0:
            raise ValueError("Empty dataset is not allowed.")

        classes, y_encoded = np.unique(y, return_inverse=True)
        self.classes_ = classes
        self.n_classes_ = len(classes)
        self.root_ = self._build_tree(x, y_encoded, depth=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(x)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.root_ is None or self.classes_ is None or self.n_classes_ is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        probs = np.zeros((x.shape[0], self.n_classes_), dtype=float)
        for i, row in enumerate(x):
            node = self.root_
            while not node.is_leaf:
                if row[node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            probs[i] = node.proba
        return probs

    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        counts = np.bincount(y, minlength=self.n_classes_)
        prediction = int(np.argmax(counts))
        proba = counts / counts.sum()
        node = _TreeNode(prediction=prediction, proba=proba)

        if self._should_stop(y, depth):
            return node

        split = self._best_split(x, y)
        if split is None:
            return node

        feature_idx, threshold = split
        left_mask = x[:, feature_idx] <= threshold
        right_mask = ~left_mask

        node.feature_index = feature_idx
        node.threshold = threshold
        node.left = self._build_tree(x[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(x[right_mask], y[right_mask], depth + 1)
        return node

    def _should_stop(self, y: np.ndarray, depth: int) -> bool:
        if np.unique(y).size == 1:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if y.size < self.min_samples_split:
            return True
        return False

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int, float] | None:
        n_samples, n_features = x.shape
        parent_gini = self._gini(y)
        best_gain = 0.0
        best: tuple[int, float] | None = None

        for feature_idx in range(n_features):
            values = x[:, feature_idx]
            thresholds = np.unique(values)
            if thresholds.size <= 1:
                continue

            candidate_thresholds = (thresholds[:-1] + thresholds[1:]) / 2.0
            for threshold in candidate_thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask
                n_left = int(left_mask.sum())
                n_right = n_samples - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                left_gini = self._gini(y[left_mask])
                right_gini = self._gini(y[right_mask])
                weighted_gini = (n_left / n_samples) * left_gini + (n_right / n_samples) * right_gini
                gain = parent_gini - weighted_gini
                if gain > best_gain:
                    best_gain = gain
                    best = (feature_idx, float(threshold))
        return best

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        if y.size == 0:
            return 0.0
        counts = np.bincount(y)
        probs = counts / y.size
        return float(1.0 - np.sum(probs**2))
