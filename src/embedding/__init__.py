"""임베딩 모듈

텍스트 임베딩 생성 및 저장을 위한 모듈.
"""

from embedding.base_embedder import BaseEmbedder
from embedding.base_store import BaseEmbeddingStore
from embedding.file_store import FileEmbeddingStore
from embedding.openai_embedder import OpenAIEmbedder


def create_embedder(config: dict) -> BaseEmbedder:
    """설정에 따라 적절한 임베더 생성

    Args:
        config: 임베딩 설정 딕셔너리
            - provider: 임베딩 제공자 ("openai")
            - model: 모델명
            - batch_size: 배치 크기

    Returns:
        BaseEmbedder 구현체

    Raises:
        ValueError: 알 수 없는 provider이거나 필수 설정이 누락된 경우
        KeyError: 필수 설정 키가 없는 경우
    """
    provider = config["provider"]

    if provider == "openai":
        return OpenAIEmbedder(
            model=config["model"],
            batch_size=config["batch_size"],
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def create_store(store_type: str, cache_dir: str) -> BaseEmbeddingStore:
    """임베딩 저장소 팩토리 함수

    Args:
        store_type: 저장소 타입 ("file")
        cache_dir: 캐시 저장 디렉토리

    Returns:
        BaseEmbeddingStore 구현체

    Raises:
        ValueError: 알 수 없는 store_type인 경우
    """
    if store_type == "file":
        return FileEmbeddingStore(cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown store type: {store_type}")


__all__ = [
    "BaseEmbedder",
    "BaseEmbeddingStore",
    "FileEmbeddingStore",
    "OpenAIEmbedder",
    "create_embedder",
    "create_store",
]
