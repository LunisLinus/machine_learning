from collections import Counter
from math import sqrt
from typing import Iterable, List, Sequence

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:
    def __init__(self, k: int = 3, metric: str = "euclidean") -> None:
        if k <= 0:
            raise ValueError("k must be a positive integer")
        if metric not in {"euclidean", "manhattan"}:
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        self.k = k
        self.metric = metric
        self._x_train: List[List[float]] = []
        self._y_train: List[int] = []

    def fit(self, x_train: Sequence[Sequence[float]], y_train: Sequence[int]) -> None:
        if len(x_train) == 0:
            raise ValueError("x_train cannot be empty")
        if len(x_train) != len(y_train):
            raise ValueError("x_train and y_train must have the same length")

        self._x_train = [list(map(float, row)) for row in x_train]
        self._y_train = list(y_train)

    def predict(self, x_test: Sequence[Sequence[float]]) -> List[int]:
        if not self._x_train:
            raise RuntimeError("Call fit before predict")
        return [self._predict_one(sample) for sample in x_test]

    def _predict_one(self, sample: Iterable[float]) -> int:
        sample = list(map(float, sample))
        distances = []

        for train_vector, label in zip(self._x_train, self._y_train):
            if len(train_vector) != len(sample):
                raise ValueError("All vectors must have the same number of features")
            dist = self._distance(train_vector, sample)
            distances.append((dist, label))

        distances.sort(key=lambda item: item[0])
        nearest_labels = [label for _, label in distances[: self.k]]
        counter = Counter(nearest_labels)
        max_votes = max(counter.values())

        candidates = {label for label, votes in counter.items() if votes == max_votes}
        for _, label in distances:
            if label in candidates:
                return label

        raise RuntimeError("Unexpected state during prediction")

    @staticmethod
    def _euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
        return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def _manhattan_distance(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(abs(x - y) for x, y in zip(a, b))

    def _distance(self, a: Sequence[float], b: Sequence[float]) -> float:
        if self.metric == "euclidean":
            return self._euclidean_distance(a, b)
        return self._manhattan_distance(a, b)
