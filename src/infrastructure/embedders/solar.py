"""
Solar Embedder 구현.

Upstage Solar Pro2 Embedding API를 사용하는 EmbedderProtocol 구현체.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI

from core.protocols import EmbedderProtocol


@dataclass
class SolarEmbedderConfig:
    """Solar Embedder 설정."""

    api_key: str | None = None
    base_url: str = "https://api.upstage.ai/v1"
    model: str = "embedding-passage"
    max_chars: int = 2500  # 토큰 제한을 피하기 위한 최대 글자수
    dimensions: int = 4096


class SolarEmbedder:
    """
    Upstage Solar Pro2 Embedding API를 사용하는 Embedder.

    EmbedderProtocol을 구현합니다.
    """

    def __init__(self, config: SolarEmbedderConfig | None = None) -> None:
        self._config = config or SolarEmbedderConfig()

        api_key = self._config.api_key or os.getenv("SOLAR_PRO2_API_KEY")
        if not api_key:
            raise ValueError(
                "SOLAR_PRO2_API_KEY가 필요합니다. "
                "환경 변수 또는 SolarEmbedderConfig.api_key로 설정하세요."
            )

        self._client = OpenAI(api_key=api_key, base_url=self._config.base_url)

    def embed(self, text: str) -> list[float]:
        """
        단일 텍스트 임베딩 생성.

        Args:
            text: 임베딩할 텍스트

        Returns:
            임베딩 벡터 (4096차원)
        """
        # 전처리: 줄바꿈 제거, 길이 제한
        clean_text = text.replace("\n", " ")[: self._config.max_chars]

        try:
            response = self._client.embeddings.create(
                input=[clean_text],
                model=self._config.model,
            )
            return response.data[0].embedding
        except Exception as e:
            # 에러 시 0벡터 반환 (로깅 추가 권장)
            print(f"임베딩 API 오류: {e}")
            return [0.0] * self._config.dimensions

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        배치 텍스트 임베딩 생성.

        Args:
            texts: 임베딩할 텍스트 리스트

        Returns:
            임베딩 벡터 리스트
        """
        if not texts:
            return []

        clean_texts = [t.replace("\n", " ")[: self._config.max_chars] for t in texts]

        try:
            response = self._client.embeddings.create(
                input=clean_texts,
                model=self._config.model,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"임베딩 API 오류: {e}")
            return [[0.0] * self._config.dimensions for _ in texts]


# Protocol 호환성 검증 (타입 체커용)
def _verify_protocol() -> None:
    """SolarEmbedder가 EmbedderProtocol을 구현하는지 타입 체커가 확인."""
    embedder: EmbedderProtocol = SolarEmbedder(SolarEmbedderConfig(api_key="test"))
    _ = embedder
