"""임베딩 모델 추상 기본 클래스

모든 임베딩 모델이 구현해야 하는 공통 인터페이스를 정의.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """임베딩 모델 추상 기본 클래스"""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """임베딩 벡터의 차원 반환"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """모델 이름 반환 (캐싱 키로 사용)"""
        pass

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """단일 텍스트 임베딩

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터 (shape: [dimension])
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """배치 텍스트 임베딩

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 배열 (shape: [num_texts, dimension])
        """
        pass

    def combine_paragraph_question(self, paragraph: str, question: str) -> str:
        """지문과 질문을 결합하여 임베딩용 텍스트 생성

        Args:
            paragraph: 지문
            question: 질문

        Returns:
            결합된 텍스트
        """
        return f"[지문]\n{paragraph}\n\n[질문]\n{question}"
