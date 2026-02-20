from __future__ import annotations

from typing import Any, Dict, Literal, Optional

import numpy as np


Metric = Literal["euclidean"]


def _as_2d_float_array(X: Any) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
    if X_arr.ndim != 2:
        raise ValueError("X должен быть 2D-массивом формы (n_samples, n_features)")
    if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
        raise ValueError("X не должен быть пустым")
    if not np.isfinite(X_arr).all():
        raise ValueError("X должен содержать только конечные значения")
    return X_arr


def _validate_params(*, eps: float, min_samples: int, metric: Metric) -> None:
    if not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError("eps должен быть числом > 0")
    if not isinstance(min_samples, int) or min_samples < 1:
        raise ValueError("min_samples должен быть целым числом >= 1")
    if metric not in {"euclidean"}:
        raise ValueError("metric должен быть 'euclidean'")


def _pairwise_squared_distances(X: np.ndarray) -> np.ndarray:
    x2 = np.sum(X * X, axis=1, keepdims=True)
    d2 = x2 + x2.T - 2.0 * (X @ X.T)
    return np.maximum(d2, 0.0)


class CustomDBSCAN:
    def __init__(self, eps: float = 0.5, min_samples: int = 5, metric: Metric = "euclidean") -> None:
        _validate_params(eps=eps, min_samples=min_samples, metric=metric)
        self.eps = float(eps)
        self.min_samples = min_samples
        self.metric = metric

        self.labels_: Optional[np.ndarray] = None
        self.core_sample_indices_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        _ = deep
        return {"eps": self.eps, "min_samples": self.min_samples, "metric": self.metric}

    def set_params(self, **params: Any) -> "CustomDBSCAN":
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Неизвестный параметр: {key}")
            setattr(self, key, value)
        return self

    def fit(self, X: Any) -> "CustomDBSCAN":
        X_arr = _as_2d_float_array(X)
        _validate_params(eps=self.eps, min_samples=self.min_samples, metric=self.metric)

        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        eps2 = self.eps * self.eps
        d2 = _pairwise_squared_distances(X_arr)
        neighbors = d2 <= eps2

        neighbor_counts = np.sum(neighbors, axis=1).astype(int)
        is_core = neighbor_counts >= self.min_samples
        self.core_sample_indices_ = np.where(is_core)[0].astype(int)

        labels = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True

            if not is_core[i]:
                continue

            labels[i] = cluster_id
            queue = [i]
            while queue:
                p = queue.pop()
                p_neighbors = np.where(neighbors[p])[0]
                for q in p_neighbors.tolist():
                    if not visited[q]:
                        visited[q] = True
                        if is_core[q]:
                            queue.append(q)
                    if labels[q] == -1:
                        labels[q] = cluster_id

            cluster_id += 1

        self.labels_ = labels
        return self

    def fit_predict(self, X: Any) -> np.ndarray:
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_

