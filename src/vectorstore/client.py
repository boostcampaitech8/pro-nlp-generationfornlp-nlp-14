"""Elasticsearch 클라이언트 팩토리.

로컬 서버 환경용 클라이언트를 생성합니다.
"""

from __future__ import annotations

from elasticsearch import Elasticsearch

from .config import ESConfig


def create_es_client(cfg: ESConfig | None = None) -> Elasticsearch:
    """Elasticsearch 클라이언트 생성.

    Args:
        cfg: ES 설정. None이면 기본 설정 사용.

    Returns:
        Elasticsearch 클라이언트 인스턴스.

    Raises:
        ValueError: ES_URL이 설정되지 않은 경우.
    """
    if cfg is None:
        cfg = ESConfig()

    if not cfg.es_url:
        raise ValueError("ES_URL 환경변수를 설정하세요.")

    # Basic Auth 사용
    if cfg.es_username and cfg.es_password:
        return Elasticsearch(
            hosts=[cfg.es_url],
            basic_auth=(cfg.es_username, cfg.es_password),
            verify_certs=cfg.verify_certs,
            request_timeout=cfg.request_timeout_s,
        )

    # No Auth (로컬 개발용)
    return Elasticsearch(
        hosts=[cfg.es_url],
        verify_certs=cfg.verify_certs,
        request_timeout=cfg.request_timeout_s,
    )


def check_connection(es: Elasticsearch) -> bool:
    """ES 연결 상태 확인.

    Returns:
        연결 성공 여부.
    """
    try:
        return bool(es.ping())
    except Exception:
        return False
