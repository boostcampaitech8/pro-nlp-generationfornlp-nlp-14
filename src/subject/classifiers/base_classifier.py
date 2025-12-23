"""분류기 추상 기본 클래스

모든 분류기가 구현해야 하는 공통 인터페이스를 정의.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseClassifier(ABC):
    """분류기 추상 기본 클래스"""

    @abstractmethod
    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        """분류기 학습

        Args:
            embeddings: 임베딩 벡터 (shape: [n_samples, dimension])
            labels: 라벨 배열

        Returns:
            학습 결과 메트릭 딕셔너리
        """
        pass

    @abstractmethod
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """분류 예측

        Args:
            embeddings: 임베딩 벡터 (shape: [n_samples, dimension])

        Returns:
            예측 라벨 배열
        """
        pass

    @abstractmethod
    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """분류 확률 예측

        Args:
            embeddings: 임베딩 벡터 (shape: [n_samples, dimension])

        Returns:
            클래스별 확률 배열 (shape: [n_samples, n_classes])
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """분류기 저장

        Args:
            path: 저장 경로
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseClassifier":
        """분류기 로드

        Args:
            path: 로드 경로

        Returns:
            로드된 분류기
        """
        pass

    @property
    @abstractmethod
    def classes(self) -> np.ndarray:
        """클래스 라벨 목록 반환"""
        pass
