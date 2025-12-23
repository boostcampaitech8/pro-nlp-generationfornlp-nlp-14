"""파일 기반 임베딩 저장소 (NPZ/NPY)

임베딩 결과를 파일 시스템에 저장하고 로드하는 기능 제공.
"""

import hashlib
from pathlib import Path

import numpy as np

from embedding.base_store import BaseEmbeddingStore


class FileEmbeddingStore(BaseEmbeddingStore):
    """파일 기반 임베딩 저장소 (NPZ/NPY)"""

    def __init__(self, cache_dir: str = "data/embeddings"):
        """파일 저장소 초기화

        Args:
            cache_dir: 캐시 저장 디렉토리
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, text: str, model_name: str) -> str:
        """텍스트와 모델명으로 캐시 키 생성

        Args:
            text: 원본 텍스트
            model_name: 임베딩 모델명

        Returns:
            MD5 해시 기반 캐시 키
        """
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, model_name: str) -> np.ndarray | None:
        """캐시에서 임베딩 조회"""
        cache_key = self._get_cache_key(text, model_name)
        cache_path = self.cache_dir / f"{cache_key}.npy"

        if cache_path.exists():
            return np.load(cache_path)
        return None

    def set(self, text: str, model_name: str, embedding: np.ndarray) -> None:
        """임베딩을 캐시에 저장"""
        cache_key = self._get_cache_key(text, model_name)
        cache_path = self.cache_dir / f"{cache_key}.npy"
        np.save(cache_path, embedding)

    def save_bulk(
        self,
        embeddings: np.ndarray,
        ids: list[str],
        model_name: str,
        filename: str = "embeddings.npz",
    ) -> None:
        """대량 임베딩 저장 (학습 데이터용)"""
        save_path = self.cache_dir / filename
        np.savez_compressed(
            save_path,
            embeddings=embeddings,
            ids=np.array(ids),
            model_name=np.array(model_name),
        )

    def load_bulk(self, filename: str = "embeddings.npz") -> dict | None:
        """대량 임베딩 로드"""
        load_path = self.cache_dir / filename
        if load_path.exists():
            data = np.load(load_path, allow_pickle=True)
            return {
                "embeddings": data["embeddings"],
                "ids": data["ids"],
                "model_name": str(data["model_name"]),
            }
        return None

    def exists_bulk(self, filename: str = "embeddings.npz") -> bool:
        """대량 임베딩 파일 존재 여부 확인"""
        return (self.cache_dir / filename).exists()
