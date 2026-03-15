from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np


InitMethod = Literal["random", "k-means++"]


@dataclass(frozen=True)
class _KMeansRunResult:
    centers: np.ndarray
    labels: np.ndarray
    inertia: float
    n_iter: int


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


def _validate_params(
    *,
    n_clusters: int,
    init: InitMethod,
    n_init: int,
    max_iter: int,
    tol: float,
    random_state: Optional[int],
) -> None:
    if not isinstance(n_clusters, int) or n_clusters < 1:
        raise ValueError("n_clusters должен быть целым числом >= 1")
    if init not in {"random", "k-means++"}:
        raise ValueError("init должен быть 'random' или 'k-means++'")
    if not isinstance(n_init, int) or n_init < 1:
        raise ValueError("n_init должен быть целым числом >= 1")
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter должен быть целым числом >= 1")
    if not isinstance(tol, (int, float)) or tol < 0:
        raise ValueError("tol должен быть числом >= 0")
    if random_state is not None and not isinstance(random_state, int):
        raise ValueError("random_state должен быть None или int")


def _squared_euclidean_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(centers * centers, axis=1, keepdims=True).T
    d2 = x2 + c2 - 2.0 * (X @ centers.T)
    return np.maximum(d2, 0.0)


def _compute_inertia(X: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
    diffs = X - centers[labels]
    return float(np.sum(diffs * diffs))


def _kmeans_plus_plus_init(
    X: np.ndarray, n_clusters: int, rng: np.random.Generator
) -> np.ndarray:
    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features), dtype=float)

    first_idx = int(rng.integers(0, n_samples))
    centers[0] = X[first_idx]

    closest_d2 = _squared_euclidean_distances(X, centers[0:1]).reshape(-1)

    for i in range(1, n_clusters):
        total = float(closest_d2.sum())
        if total == 0.0:
            remaining = rng.choice(n_samples, size=n_clusters - i, replace=False)
            centers[i:] = X[remaining[: n_clusters - i]]
            break

        probs = closest_d2 / total
        next_idx = int(rng.choice(n_samples, p=probs))
        centers[i] = X[next_idx]

        d2_new = _squared_euclidean_distances(X, centers[i : i + 1]).reshape(-1)
        closest_d2 = np.minimum(closest_d2, d2_new)

    return centers


def _random_init(X: np.ndarray, n_clusters: int, rng: np.random.Generator) -> np.ndarray:
    n_samples = X.shape[0]
    idx = rng.choice(n_samples, size=n_clusters, replace=False)
    return X[idx].astype(float, copy=True)


def _handle_empty_clusters(
    X: np.ndarray,
    centers: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n_clusters = centers.shape[0]
    counts = np.bincount(labels, minlength=n_clusters)
    empty = np.where(counts == 0)[0]
    if empty.size == 0:
        return centers, counts

    d2 = _squared_euclidean_distances(X, centers)
    min_d2 = np.min(d2, axis=1)
    candidates = np.argsort(min_d2)[::-1]
    used = set()
    for k in empty.tolist():
        for idx in candidates.tolist():
            if idx not in used:
                centers[k] = X[idx]
                used.add(idx)
                break
        else:
            centers[k] = X[int(rng.integers(0, X.shape[0]))]

    return centers, counts


class CustomKMeans:
    def __init__(
        self,
        n_clusters: int = 8,
        init: InitMethod = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ) -> None:
        _validate_params(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )

        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = float(tol)
        self.random_state = random_state

        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None
        self.n_iter_: Optional[int] = None
        self.n_features_in_: Optional[int] = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        _ = deep
        return {
            "n_clusters": self.n_clusters,
            "init": self.init,
            "n_init": self.n_init,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "CustomKMeans":
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Неизвестный параметр: {key}")
            setattr(self, key, value)
        return self

    def fit(self, X: Any) -> "CustomKMeans":
        X_arr = _as_2d_float_array(X)
        _validate_params(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )

        n_samples, n_features = X_arr.shape
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters не должен превышать число объектов")

        self.n_features_in_ = n_features

        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(
            0, np.iinfo(np.int32).max, size=self.n_init, dtype=np.int64
        )

        best: Optional[_KMeansRunResult] = None
        for seed in seeds:
            run_rng = np.random.default_rng(int(seed))
            result = self._fit_single(X_arr, run_rng)
            if best is None or result.inertia < best.inertia:
                best = result

        assert best is not None
        self.cluster_centers_ = best.centers
        self.labels_ = best.labels
        self.inertia_ = best.inertia
        self.n_iter_ = best.n_iter
        return self

    def fit_predict(self, X: Any) -> np.ndarray:
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_

    def predict(self, X: Any) -> np.ndarray:
        if self.cluster_centers_ is None or self.n_features_in_ is None:
            raise RuntimeError("Сначала вызовите fit")
        X_arr = _as_2d_float_array(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X должен иметь {self.n_features_in_} признаков, получено {X_arr.shape[1]}"
            )
        d2 = _squared_euclidean_distances(X_arr, self.cluster_centers_)
        return np.argmin(d2, axis=1).astype(int)

    def transform(self, X: Any) -> np.ndarray:
        if self.cluster_centers_ is None or self.n_features_in_ is None:
            raise RuntimeError("Сначала вызовите fit")
        X_arr = _as_2d_float_array(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X должен иметь {self.n_features_in_} признаков, получено {X_arr.shape[1]}"
            )
        return np.sqrt(_squared_euclidean_distances(X_arr, self.cluster_centers_))

    def _fit_single(self, X: np.ndarray, rng: np.random.Generator) -> _KMeansRunResult:
        n_samples = X.shape[0]
        if self.init == "k-means++":
            centers = _kmeans_plus_plus_init(X, self.n_clusters, rng)
        else:
            centers = _random_init(X, self.n_clusters, rng)

        labels = np.zeros(n_samples, dtype=int)
        for it in range(1, self.max_iter + 1):
            d2 = _squared_euclidean_distances(X, centers)
            new_labels = np.argmin(d2, axis=1).astype(int)

            if it > 1 and np.array_equal(new_labels, labels):
                labels = new_labels
                break

            labels = new_labels
            centers, _ = _handle_empty_clusters(X, centers, labels, rng)

            new_centers = np.zeros_like(centers)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.any():
                    new_centers[k] = X[mask].mean(axis=0)
                else:
                    new_centers[k] = centers[k]

            shift = float(np.max(np.sqrt(np.sum((new_centers - centers) ** 2, axis=1))))
            centers = new_centers
            if shift <= self.tol:
                break

        inertia = _compute_inertia(X, centers, labels)
        return _KMeansRunResult(centers=centers, labels=labels, inertia=inertia, n_iter=it)
