"""kNN 분류기 구현

scikit-learn KNeighborsClassifier 기반의 분류기.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from subject.classifiers.base_classifier import BaseClassifier


class KNNClassifier(BaseClassifier):
    """kNN 분류기

    cosine 거리 기반 k-최근접 이웃 분류기.
    """

    def __init__(
        self,
        n_neighbors: int,
        metric: str,
        weights: str,
    ):
        """kNN 분류기 초기화

        Args:
            n_neighbors: 이웃 수
            metric: 거리 메트릭 ("cosine", "euclidean" 등)
            weights: 가중치 방식 ("uniform", "distance")
        """
        self._classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            weights=weights,
            n_jobs=-1,
        )
        self._label_encoder = LabelEncoder()
        self._is_fitted = False

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        """분류기 학습

        Args:
            embeddings: 임베딩 벡터 (shape: [n_samples, dimension])
            labels: 라벨 배열 (문자열)

        Returns:
            학습 결과 메트릭 딕셔너리
        """
        # 라벨 인코딩
        encoded_labels = self._label_encoder.fit_transform(labels)

        # 분류기 학습
        self._classifier.fit(embeddings, encoded_labels)
        self._is_fitted = True

        return {
            "n_samples": len(labels),
            "n_classes": len(self._label_encoder.classes_),
            "classes": self._label_encoder.classes_.tolist(),
        }

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """분류 예측

        Args:
            embeddings: 임베딩 벡터

        Returns:
            예측 라벨 배열 (문자열)
        """
        if not self._is_fitted:
            raise RuntimeError("분류기가 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        encoded_preds = self._classifier.predict(embeddings)
        return self._label_encoder.inverse_transform(encoded_preds)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """분류 확률 예측

        Args:
            embeddings: 임베딩 벡터

        Returns:
            클래스별 확률 배열
        """
        if not self._is_fitted:
            raise RuntimeError("분류기가 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        return self._classifier.predict_proba(embeddings)

    def save(self, path: str) -> None:
        """분류기 저장"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "classifier": self._classifier,
            "label_encoder": self._label_encoder,
            "is_fitted": self._is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "KNNClassifier":
        """분류기 로드"""
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls.__new__(cls)
        instance._classifier = data["classifier"]
        instance._label_encoder = data["label_encoder"]
        instance._is_fitted = data["is_fitted"]
        return instance

    @property
    def classes(self) -> np.ndarray:
        """클래스 라벨 목록 반환"""
        if not self._is_fitted:
            raise RuntimeError("분류기가 학습되지 않았습니다.")
        return self._label_encoder.classes_
