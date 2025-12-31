"""Elasticsearch 인덱스 매핑 정의.

이 모듈은 src/vectorstore와 독립적으로 ES 인덱스 스키마만 정의합니다.
인프라 설정 목적이므로 비즈니스 로직 의존성이 없습니다.
"""

from __future__ import annotations

from typing import Any


def _ko_nori_settings() -> dict[str, Any]:
    """한국어 Nori 분석기 설정.

    Note:
        analysis-nori 플러그인이 필요합니다.
    """
    return {
        "analysis": {
            "analyzer": {
                "ko_nori": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "filter": ["lowercase"],
                }
            }
        }
    }


def parents_index_mapping(dims: int) -> dict[str, Any]:
    """Parent 인덱스 매핑.

    Args:
        dims: 임베딩 벡터 차원수

    Fields:
        - doc_id: 고유 식별자
        - subject: 과목 (keyword)
        - topic: 토픽/단원명 (text + keyword)
        - parent_text: 지문/문단 전문 (text)
        - topic_vector: 토픽 임베딩 (dense_vector)
        - parent_vector: 지문 임베딩 (dense_vector, 선택)
        - version: 스키마 버전
        - created_at: 생성 시각
    """
    return {
        "settings": _ko_nori_settings(),
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "subject": {"type": "keyword"},
                "topic": {
                    "type": "text",
                    "analyzer": "ko_nori",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                },
                "parent_text": {"type": "text", "analyzer": "ko_nori"},
                "topic_vector": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                },
                "parent_vector": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                },
                "version": {"type": "keyword"},
                "created_at": {"type": "date"},
            }
        },
    }


def chunks_index_mapping(dims: int) -> dict[str, Any]:
    """Chunk 인덱스 매핑.

    Args:
        dims: 임베딩 벡터 차원수

    Fields:
        - chunk_id: 고유 식별자
        - doc_id: Parent 문서 ID (외래키)
        - subject: 과목 (keyword)
        - topic: 토픽/단원명 (text + keyword)
        - chunk_idx: 청크 순서 인덱스
        - chunk_text: 청크 텍스트 (text)
        - chunk_vector: 청크 임베딩 (dense_vector)
        - start_char, end_char: 원문 내 위치
        - version: 스키마 버전
        - created_at: 생성 시각
    """
    return {
        "settings": _ko_nori_settings(),
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "subject": {"type": "keyword"},
                "topic": {
                    "type": "text",
                    "analyzer": "ko_nori",
                    "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                },
                "chunk_idx": {"type": "integer"},
                "chunk_text": {"type": "text", "analyzer": "ko_nori"},
                "chunk_vector": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                },
                "start_char": {"type": "integer"},
                "end_char": {"type": "integer"},
                "version": {"type": "keyword"},
                "created_at": {"type": "date"},
            }
        },
    }


# 인덱스 타입 -> 매핑 함수 레지스트리
INDEX_MAPPINGS = {
    "parents": parents_index_mapping,
    "chunks": chunks_index_mapping,
}

__all__ = [
    "parents_index_mapping",
    "chunks_index_mapping",
    "INDEX_MAPPINGS",
]
