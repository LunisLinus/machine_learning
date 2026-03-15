from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from practice.methods.ensambles.random_forest import _as_1d_array, _as_2d_float_array, _validate_basic_params


@dataclass(frozen=True)
class _FittedEstimator:
    estimator: Any


class CustomBaggingClassifier:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        criterion: Literal["gini", "entropy", "log_loss"] = "gini",
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
        self.random_state = random_state

        self.n_features_in_: Optional[int] = None
        self.classes_: Optional[np.ndarray] = None
        self.estimators_: List[DecisionTreeClassifier] = []
        self._fitted_estimators: List[_FittedEstimator] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        _ = deep
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "criterion": self.criterion,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "CustomBaggingClassifier":
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Неизвестный параметр: {key}")
            setattr(self, key, value)
        return self

    def fit(self, X: Any, y: Any) -> "CustomBaggingClassifier":
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
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(
            0, np.iinfo(np.int32).max, size=self.n_estimators, dtype=np.int64
        )

        self.n_features_in_ = n_features
        self.classes_ = np.unique(y_arr)
        self.estimators_ = []
        self._fitted_estimators = []

        for seed in seeds:
            sample_idx = rng.integers(0, n_samples, size=n_samples, dtype=np.int64)
            X_boot = X_arr[sample_idx]
            y_boot = y_arr[sample_idx]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                criterion=self.criterion,
                random_state=int(seed),
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)
            self._fitted_estimators.append(_FittedEstimator(estimator=tree))

        self.feature_importances_ = self._compute_feature_importances()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.classes_ is None or self.n_features_in_ is None or not self._fitted_estimators:
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

        for fitted in self._fitted_estimators:
            tree: DecisionTreeClassifier = fitted.estimator
            tree_proba = tree.predict_proba(X_arr)
            for j, cls in enumerate(tree.classes_):
                proba[:, class_to_index[cls]] += tree_proba[:, j]

        proba /= float(len(self._fitted_estimators))
        return proba

    def predict(self, X: Any) -> np.ndarray:
        if self.classes_ is None or self.n_features_in_ is None or not self._fitted_estimators:
            raise RuntimeError("Сначала вызовите fit")

        X_arr = _as_2d_float_array(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X должен иметь {self.n_features_in_} признаков, получено {X_arr.shape[1]}"
            )

        n_samples = X_arr.shape[0]
        n_classes = int(self.classes_.shape[0])
        votes = np.zeros((n_samples, n_classes), dtype=int)
        class_to_index = {c: i for i, c in enumerate(self.classes_)}

        for fitted in self._fitted_estimators:
            tree: DecisionTreeClassifier = fitted.estimator
            pred = tree.predict(X_arr)
            idx = np.fromiter((class_to_index[p] for p in pred), dtype=int, count=n_samples)
            votes[np.arange(n_samples), idx] += 1

        chosen = np.argmax(votes, axis=1)
        return self.classes_[chosen]

    def _compute_feature_importances(self) -> np.ndarray:
        assert self.n_features_in_ is not None
        importances = np.zeros(self.n_features_in_, dtype=float)
        if not self._fitted_estimators:
            return importances

        for fitted in self._fitted_estimators:
            tree: DecisionTreeClassifier = fitted.estimator
            importances += np.asarray(tree.feature_importances_, dtype=float)

        importances /= float(len(self._fitted_estimators))
        total = float(importances.sum())
        if total > 0.0:
            importances /= total
        return importances


class CustomBaggingRegressor:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        criterion: Literal[
            "squared_error", "friedman_mse", "absolute_error", "poisson"
        ] = "squared_error",
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
        self.random_state = random_state

        self.n_features_in_: Optional[int] = None
        self.estimators_: List[DecisionTreeRegressor] = []
        self._fitted_estimators: List[_FittedEstimator] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        _ = deep
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "criterion": self.criterion,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "CustomBaggingRegressor":
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Неизвестный параметр: {key}")
            setattr(self, key, value)
        return self

    def fit(self, X: Any, y: Any) -> "CustomBaggingRegressor":
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
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(
            0, np.iinfo(np.int32).max, size=self.n_estimators, dtype=np.int64
        )

        self.n_features_in_ = n_features
        self.estimators_ = []
        self._fitted_estimators = []

        for seed in seeds:
            sample_idx = rng.integers(0, n_samples, size=n_samples, dtype=np.int64)
            X_boot = X_arr[sample_idx]
            y_boot = y_arr[sample_idx]

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                criterion=self.criterion,
                random_state=int(seed),
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)
            self._fitted_estimators.append(_FittedEstimator(estimator=tree))

        self.feature_importances_ = self._compute_feature_importances()
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.n_features_in_ is None or not self._fitted_estimators:
            raise RuntimeError("Сначала вызовите fit")

        X_arr = _as_2d_float_array(X)
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X должен иметь {self.n_features_in_} признаков, получено {X_arr.shape[1]}"
            )

        preds = np.zeros(X_arr.shape[0], dtype=float)
        for fitted in self._fitted_estimators:
            tree: DecisionTreeRegressor = fitted.estimator
            preds += tree.predict(X_arr)
        preds /= float(len(self._fitted_estimators))
        return preds

    def predict_proba(self, X: Any) -> np.ndarray:
        _ = X
        raise AttributeError("predict_proba доступен только для классификации")

    def _compute_feature_importances(self) -> np.ndarray:
        assert self.n_features_in_ is not None
        importances = np.zeros(self.n_features_in_, dtype=float)
        if not self._fitted_estimators:
            return importances

        for fitted in self._fitted_estimators:
            tree: DecisionTreeRegressor = fitted.estimator
            importances += np.asarray(tree.feature_importances_, dtype=float)

        importances /= float(len(self._fitted_estimators))
        total = float(importances.sum())
        if total > 0.0:
            importances /= total
        return importances
