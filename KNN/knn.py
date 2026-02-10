from __future__ import annotations

from typing import Any, Sequence

import numpy as np


class KNNClassifier:
    def __init__(self, k: int = 3, metric: str = "euclidean") -> None:
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if metric not in {"euclidean", "manhattan"}:
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        self.k = k
        self.metric = metric
        self._x_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, x_train: Any, y_train: Any) -> None:
        x_arr = np.asarray(x_train, dtype=float)
        y_arr = np.asarray(y_train)

        if x_arr.ndim != 2:
            raise ValueError("x_train must be a 2D array of shape (n_samples, n_features)")
        if y_arr.ndim != 1:
            raise ValueError("y_train must be a 1D array")
        if x_arr.shape[0] == 0:
            raise ValueError("x_train cannot be empty")
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("x_train and y_train must have the same length")
        if self.k > x_arr.shape[0]:
            raise ValueError("k cannot be greater than the number of training samples")

        self._x_train = x_arr
        self._y_train = y_arr

    def predict(self, x_test: Any) -> np.ndarray:
        if self._x_train is None or self._y_train is None:
            raise RuntimeError("Call fit before predict")
        x_arr = np.asarray(x_test, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        if x_arr.ndim != 2:
            raise ValueError("x_test must be a 2D array of shape (n_samples, n_features)")
        if x_arr.shape[1] != self._x_train.shape[1]:
            raise ValueError("All vectors must have the same number of features")

        preds = np.empty(x_arr.shape[0], dtype=self._y_train.dtype)
        for i, sample in enumerate(x_arr):
            preds[i] = self._predict_one(sample)
        return preds

    def _predict_one(self, sample: np.ndarray) -> Any:
        assert self._x_train is not None and self._y_train is not None

        distances = self._distances_to(sample)
        k = self.k
        idx = np.argpartition(distances, kth=k - 1)[:k]
        idx_sorted = idx[np.argsort(distances[idx], kind="mergesort")]

        nearest_labels = self._y_train[idx_sorted]
        labels, counts = np.unique(nearest_labels, return_counts=True)
        max_votes = int(counts.max())
        candidates = set(labels[counts == max_votes].tolist())
        for lbl in nearest_labels.tolist():
            if lbl in candidates:
                return lbl
        raise RuntimeError("Unexpected state during prediction")

    def _distances_to(self, sample: np.ndarray) -> np.ndarray:
        assert self._x_train is not None
        if self.metric == "euclidean":
            diff = self._x_train - sample
            return np.sqrt(np.sum(diff * diff, axis=1))
        return np.sum(np.abs(self._x_train - sample), axis=1)
