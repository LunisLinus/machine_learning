from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


MaxFeatures = Union[None, int, float, Literal["sqrt", "log2"]]


@dataclass(frozen=True)
class _FittedTree:
    estimator: Any
    feature_indices: np.ndarray


def _as_2d_float_array(X: Any) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
    if X_arr.ndim != 2:
        raise ValueError("X должен быть 2D-массивом формы (n_samples, n_features)")
    if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
        raise ValueError("X не должен быть пустым")
    return X_arr


def _as_1d_array(y: Any, *, name: str = "y") -> np.ndarray:
    y_arr = np.asarray(y)
    if y_arr.ndim != 1:
        raise ValueError(f"{name} должен быть 1D-массивом")
    if y_arr.shape[0] == 0:
        raise ValueError(f"{name} не должен быть пустым")
    return y_arr


def _validate_basic_params(
    *,
    n_estimators: int,
    max_depth: Optional[int],
    random_state: Optional[int],
) -> None:
    if not isinstance(n_estimators, int) or n_estimators < 1:
        raise ValueError("n_estimators должен быть целым числом >= 1")
    if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 1):
        raise ValueError("max_depth должен быть None или целым числом >= 1")
    if random_state is not None and not isinstance(random_state, int):
        raise ValueError("random_state должен быть None или int")


def _resolve_k_max_features(max_features: MaxFeatures, n_features: int) -> int:
    if max_features is None:
        return n_features

    if isinstance(max_features, str):
        if max_features not in {"sqrt", "log2"}:
            raise ValueError("max_features должен быть None, int, float или 'sqrt'/'log2'")
        if max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        return max(1, int(np.log2(n_features)))

    if isinstance(max_features, int):
        if not (1 <= max_features <= n_features):
            raise ValueError("При int max_features должен быть в диапазоне [1, n_features]")
        return int(max_features)

    if isinstance(max_features, float):
        if not (0.0 < max_features <= 1.0):
            raise ValueError("При float max_features должен быть в диапазоне (0, 1]")
        return max(1, int(max_features * n_features))

    raise ValueError("max_features должен быть None, int, float или 'sqrt'/'log2'")


class CustomRandomForestClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        criterion: Literal["gini", "entropy", "log_loss"] = "gini",
        max_features: MaxFeatures = "sqrt",
        random_state: Optional[int] = None,
    ) -> None:
        _validate_basic_params(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        if criterion not in {"gini", "entropy", "log_loss"}:
            raise ValueError("criterion должен быть 'gini', 'entropy' или 'log_loss'")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.n_features_in_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.estimators_: List[DecisionTreeClassifier] = []
        self._fitted_trees: List[_FittedTree] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        _ = deep
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "criterion": self.criterion,
            "max_features": self.max_features,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "CustomRandomForestClassifier":

        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Неизвестный параметр: {key}")
            setattr(self, key, value)
        return self

    def fit(self, X: Any, y: Any) -> "CustomRandomForestClassifier":
        X_arr = _as_2d_float_array(X)
        y_arr = _as_1d_array(y, name="y")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("Количество объектов в X и y должно совпадать")

        _validate_basic_params(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        if self.criterion not in {"gini", "entropy", "log_loss"}:
            raise ValueError("criterion должен быть 'gini', 'entropy' или 'log_loss'")

        n_samples, n_features = X_arr.shape
        k_features = _resolve_k_max_features(self.max_features, n_features)

        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, np.iinfo(np.int32).max, size=self.n_estimators, dtype=np.int64)

        self.n_features_in_ = n_features
        self.classes_ = np.unique(y_arr)
        self.estimators_ = []
        self._fitted_trees = []

        for seed in seeds:
            sample_idx = rng.integers(0, n_samples, size=n_samples, dtype=np.int64)
            if k_features == n_features:
                feat_idx = np.arange(n_features, dtype=int)
            else:
                feat_idx = rng.choice(n_features, size=k_features, replace=False).astype(int)

            X_boot = X_arr[sample_idx][:, feat_idx]
            y_boot = y_arr[sample_idx]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                criterion=self.criterion,
                random_state=int(seed),
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)
            self._fitted_trees.append(_FittedTree(estimator=tree, feature_indices=feat_idx))

        self.feature_importances_ = self._compute_feature_importances()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.classes_ is None or self.n_features_in_ is None or not self._fitted_trees:
            raise RuntimeError("Сначала вызовите fit")

        X_arr = _as_2d_float_array(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X должен иметь {self.n_features_in_} признаков, получено {X_arr.shape[1]}"
            )

        n_samples = X_arr.shape[0]
        n_classes = int(self.classes_.shape[0])
        proba = np.zeros((n_samples, n_classes), dtype=float)

        class_to_index = {c: i for i, c in enumerate(self.classes_)}

        for fitted in self._fitted_trees:
            tree: DecisionTreeClassifier = fitted.estimator
            X_sub = X_arr[:, fitted.feature_indices]
            tree_proba = tree.predict_proba(X_sub)
            for j, cls in enumerate(tree.classes_):
                proba[:, class_to_index[cls]] += tree_proba[:, j]

        proba /= float(len(self._fitted_trees))
        return proba

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        assert self.classes_ is not None
        return self.classes_[np.argmax(proba, axis=1)]

    def _compute_feature_importances(self) -> np.ndarray:
        assert self.n_features_in_ is not None
        importances = np.zeros(self.n_features_in_, dtype=float)
        if not self._fitted_trees:
            return importances

        for fitted in self._fitted_trees:
            tree: DecisionTreeClassifier = fitted.estimator
            tree_imp = np.asarray(tree.feature_importances_, dtype=float)
            importances[fitted.feature_indices] += tree_imp

        importances /= float(len(self._fitted_trees))
        total = float(importances.sum())
        if total > 0.0:
            importances /= total
        return importances


class CustomRandomForestRegressor:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        criterion: Literal[
            "squared_error", "friedman_mse", "absolute_error", "poisson"
        ] = "squared_error",
        max_features: MaxFeatures = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        _validate_basic_params(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        if criterion not in {"squared_error", "friedman_mse", "absolute_error", "poisson"}:
            raise ValueError(
                "criterion должен быть 'squared_error', 'friedman_mse', 'absolute_error' или 'poisson'"
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state

        self.n_features_in_: Optional[int] = None
        self.estimators_: List[DecisionTreeRegressor] = []
        self._fitted_trees: List[_FittedTree] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        _ = deep
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "criterion": self.criterion,
            "max_features": self.max_features,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "CustomRandomForestRegressor":
        """
        Установить параметры модели в формате sklearn.
        """

        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Неизвестный параметр: {key}")
            setattr(self, key, value)
        return self

    def fit(self, X: Any, y: Any) -> "CustomRandomForestRegressor":
        X_arr = _as_2d_float_array(X)
        y_arr = _as_1d_array(y, name="y").astype(float)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("Количество объектов в X и y должно совпадать")

        _validate_basic_params(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        if self.criterion not in {"squared_error", "friedman_mse", "absolute_error", "poisson"}:
            raise ValueError(
                "criterion должен быть 'squared_error', 'friedman_mse', 'absolute_error' или 'poisson'"
            )

        n_samples, n_features = X_arr.shape
        k_features = _resolve_k_max_features(self.max_features, n_features)

        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, np.iinfo(np.int32).max, size=self.n_estimators, dtype=np.int64)

        self.n_features_in_ = n_features
        self.estimators_ = []
        self._fitted_trees = []

        for seed in seeds:
            sample_idx = rng.integers(0, n_samples, size=n_samples, dtype=np.int64)
            if k_features == n_features:
                feat_idx = np.arange(n_features, dtype=int)
            else:
                feat_idx = rng.choice(n_features, size=k_features, replace=False).astype(int)

            X_boot = X_arr[sample_idx][:, feat_idx]
            y_boot = y_arr[sample_idx]

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                criterion=self.criterion,
                random_state=int(seed),
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)
            self._fitted_trees.append(_FittedTree(estimator=tree, feature_indices=feat_idx))

        self.feature_importances_ = self._compute_feature_importances()
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.n_features_in_ is None or not self._fitted_trees:
            raise RuntimeError("Сначала вызовите fit")

        X_arr = _as_2d_float_array(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X должен иметь {self.n_features_in_} признаков, получено {X_arr.shape[1]}"
            )

        preds = np.zeros(X_arr.shape[0], dtype=float)
        for fitted in self._fitted_trees:
            tree: DecisionTreeRegressor = fitted.estimator
            preds += tree.predict(X_arr[:, fitted.feature_indices])
        preds /= float(len(self._fitted_trees))
        return preds

    def predict_proba(self, X: Any) -> np.ndarray:
        _ = X
        raise AttributeError("predict_proba доступен только для классификации")

    def _compute_feature_importances(self) -> np.ndarray:
        assert self.n_features_in_ is not None
        importances = np.zeros(self.n_features_in_, dtype=float)
        if not self._fitted_trees:
            return importances

        for fitted in self._fitted_trees:
            tree: DecisionTreeRegressor = fitted.estimator
            tree_imp = np.asarray(tree.feature_importances_, dtype=float)
            importances[fitted.feature_indices] += tree_imp

        importances /= float(len(self._fitted_trees))
        total = float(importances.sum())
        if total > 0.0:
            importances /= total
        return importances
