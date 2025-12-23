"""임베딩 저장소 추상 기본 클래스

향후 벡터 DB 등 다른 저장소로 확장 가능하도록 추상화.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbeddingStore(ABC):
    """임베딩 저장소 추상 기본 클래스"""

    @abstractmethod
    def get(self, text: str, model_name: str) -> np.ndarray | None:
        """캐시에서 임베딩 조회

        Args:
            text: 원본 텍스트
            model_name: 임베딩 모델명

        Returns:
            캐시된 임베딩 벡터 또는 None
        """
        pass

    @abstractmethod
    def set(self, text: str, model_name: str, embedding: np.ndarray) -> None:
        """임베딩을 캐시에 저장

        Args:
            text: 원본 텍스트
            model_name: 임베딩 모델명
            embedding: 임베딩 벡터
        """
        pass

    @abstractmethod
    def save_bulk(
        self,
        embeddings: np.ndarray,
        ids: list[str],
        model_name: str,
        filename: str,
    ) -> None:
        """대량 임베딩 저장

        Args:
            embeddings: 임베딩 벡터 배열 (shape: [n_samples, dimension])
            ids: 샘플 ID 리스트
            model_name: 임베딩 모델명
            filename: 저장할 파일명/컬렉션명
        """
        pass

    @abstractmethod
    def load_bulk(self, filename: str) -> dict | None:
        """대량 임베딩 로드

        Args:
            filename: 로드할 파일명/컬렉션명

        Returns:
            임베딩 데이터 딕셔너리 또는 None
            - embeddings: 임베딩 벡터 배열
            - ids: 샘플 ID 배열
            - model_name: 임베딩 모델명
        """
        pass

    @abstractmethod
    def exists_bulk(self, filename: str) -> bool:
        """대량 임베딩 존재 여부 확인

        Args:
            filename: 확인할 파일명/컬렉션명

        Returns:
            존재 여부
        """
        pass
