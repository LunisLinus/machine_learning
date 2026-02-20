from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np


class LinearRegression:
    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def _to_2d(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return x.reshape(-1, 1)
        return x

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearRegression":
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        x = self._to_2d(x)

        if x.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        if self.fit_intercept:
            x_design = np.c_[np.ones(x.shape[0]), x]
        else:
            x_design = x

        params = np.linalg.pinv(x_design.T @ x_design) @ x_design.T @ y

        if self.fit_intercept:
            self.intercept_ = float(params[0])
            self.coef_ = params[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = params

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model is not fitted yet. Call fit() first.")

        x = np.asarray(x, dtype=float)
        x = self._to_2d(x)

        return x @ self.coef_ + self.intercept_

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(x)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return 1 - (ss_res / ss_tot)


class PolynomialRegression:
    def __init__(self, degree: int = 2, fit_intercept: bool = True) -> None:
        if degree < 1:
            raise ValueError("degree must be >= 1.")

        self.degree = degree
        self.fit_intercept = fit_intercept
        self._linear_model = LinearRegression(fit_intercept=fit_intercept)
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def _to_2d(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return x.reshape(-1, 1)
        return x

    def _polynomial_features(self, x: np.ndarray) -> np.ndarray:
        x = self._to_2d(np.asarray(x, dtype=float))
        n_samples, n_features = x.shape

        columns: list[np.ndarray] = []
        for degree in range(1, self.degree + 1):
            for feature_ids in combinations_with_replacement(range(n_features), degree):
                col = np.ones(n_samples, dtype=float)
                for idx in feature_ids:
                    col *= x[:, idx]
                columns.append(col)

        return np.column_stack(columns)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PolynomialRegression":
        x_poly = self._polynomial_features(x)
        self._linear_model.fit(x_poly, y)
        self.coef_ = self._linear_model.coef_
        self.intercept_ = self._linear_model.intercept_
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_poly = self._polynomial_features(x)
        return self._linear_model.predict(x_poly)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        x_poly = self._polynomial_features(x)
        return self._linear_model.score(x_poly, y)
