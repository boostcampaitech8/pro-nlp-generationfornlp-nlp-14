"""OpenAI Embeddings API 구현체

OpenAI의 text-embedding 모델을 사용하여 텍스트를 임베딩.
"""

import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from embedding.base_embedder import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI Embeddings API 구현체"""

    # 모델별 차원 정의
    MODEL_DIMENSIONS: dict[str, int] = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str,
        batch_size: int,
        api_key: str | None = None,
    ):
        """OpenAI 임베더 초기화

        Args:
            model: 사용할 OpenAI 임베딩 모델명
            batch_size: API 호출 시 배치 크기
            api_key: OpenAI API 키 (None이면 환경변수에서 로드)
        """
        load_dotenv()
        self._model = model
        self._batch_size = batch_size

        if model not in self.MODEL_DIMENSIONS:
            raise ValueError(f"지원하지 않는 모델입니다: {model}")
        self._dimension = self.MODEL_DIMENSIONS[model]

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API 키가 필요합니다. "
                "OPENAI_API_KEY 환경변수를 설정하거나 api_key 인자를 전달하세요."
            )
        self._client = OpenAI(api_key=api_key)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """단일 텍스트 임베딩

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터 (shape: [dimension])
        """
        response = self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        return np.array(response.data[0].embedding)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """배치 텍스트 임베딩

        rate limit을 고려하여 batch_size 단위로 나누어 처리.

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 배열 (shape: [num_texts, dimension])
        """
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            batch_embeddings = [d.embedding for d in response.data]
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)
