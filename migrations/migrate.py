"""Elasticsearch 인덱스 마이그레이션 관리.

이 모듈은 src/vectorstore와 독립적으로 ES 인덱스 DDL 작업을 수행합니다.

Usage:
    make migrate-create     # 인덱스 생성
    make migrate-status     # 상태 확인
    make migrate-drop       # 삭제
    make migrate-recreate   # 재생성

환경변수:
    ES_URL: Elasticsearch URL (기본: http://localhost:9200)
    ES_USERNAME: Basic Auth 사용자명 (선택)
    ES_PASSWORD: Basic Auth 비밀번호 (선택)
    ES_PARENTS_INDEX: Parents 인덱스명 (기본: kb_parents_v1)
    ES_CHUNKS_INDEX: Chunks 인덱스명 (기본: kb_chunks_v1)
    EMBEDDING_DIMS: 임베딩 벡터 차원수 (기본: 4096)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from .mappings import chunks_index_mapping, parents_index_mapping

load_dotenv()


# =============================================================================
# Config (migrations 전용, 최소한의 설정만)
# =============================================================================


@dataclass(frozen=True)
class MigrationConfig:
    """마이그레이션 전용 설정.

    src/vectorstore/config.py와 독립적입니다.
    """

    es_url: str
    es_username: str | None
    es_password: str | None
    parents_index: str
    chunks_index: str
    embedding_dims: int

    @classmethod
    def from_env(cls) -> MigrationConfig:
        """환경변수에서 설정 로드."""
        return cls(
            es_url=os.environ["ES_URL"],
            es_username=os.getenv("ES_USERNAME"),
            es_password=os.getenv("ES_PASSWORD"),
            parents_index=os.getenv("ES_PARENTS_INDEX", "kb_parents_v1"),
            chunks_index=os.getenv("ES_CHUNKS_INDEX", "kb_chunks_v1"),
            embedding_dims=int(os.getenv("EMBEDDING_DIMS", "4096")),
        )


def create_es_client(cfg: MigrationConfig) -> Elasticsearch:
    """마이그레이션용 ES 클라이언트 생성."""
    if cfg.es_username and cfg.es_password:
        return Elasticsearch(
            hosts=[cfg.es_url],
            basic_auth=(cfg.es_username, cfg.es_password),
            request_timeout=30,
        )
    return Elasticsearch(hosts=[cfg.es_url], request_timeout=30)


# =============================================================================
# Migrator
# =============================================================================


@dataclass
class IndexInfo:
    """인덱스 정보."""

    name: str
    exists: bool
    doc_count: int = 0
    size_bytes: int = 0


class Migrator:
    """Elasticsearch 인덱스 마이그레이션 관리자."""

    def __init__(self, es: Elasticsearch, cfg: MigrationConfig):
        self.es = es
        self.cfg = cfg

    def get_index_info(self, index_name: str) -> IndexInfo:
        """인덱스 정보 조회."""
        exists = self.es.indices.exists(index=index_name)
        if not exists:
            return IndexInfo(name=index_name, exists=False)

        stats = self.es.indices.stats(index=index_name)
        index_stats = stats["indices"].get(index_name, {}).get("primaries", {})
        doc_count = index_stats.get("docs", {}).get("count", 0)
        size_bytes = index_stats.get("store", {}).get("size_in_bytes", 0)

        return IndexInfo(
            name=index_name,
            exists=True,
            doc_count=doc_count,
            size_bytes=size_bytes,
        )

    def status(self) -> dict[str, IndexInfo]:
        """모든 관리 인덱스 상태 조회."""
        return {
            "parents": self.get_index_info(self.cfg.parents_index),
            "chunks": self.get_index_info(self.cfg.chunks_index),
        }

    def create_index(
        self,
        index_type: Literal["parents", "chunks"],
        *,
        skip_existing: bool = True,
    ) -> bool:
        """단일 인덱스 생성."""
        if index_type == "parents":
            index_name = self.cfg.parents_index
            mapping = parents_index_mapping(self.cfg.embedding_dims)
        else:
            index_name = self.cfg.chunks_index
            mapping = chunks_index_mapping(self.cfg.embedding_dims)

        if self.es.indices.exists(index=index_name):
            if skip_existing:
                return True
            raise ValueError(f"인덱스 '{index_name}'이 이미 존재합니다.")

        self.es.indices.create(index=index_name, body=mapping)
        return True

    def create_all(self, *, skip_existing: bool = True) -> dict[str, bool]:
        """모든 인덱스 생성."""
        return {
            "parents": self.create_index("parents", skip_existing=skip_existing),
            "chunks": self.create_index("chunks", skip_existing=skip_existing),
        }

    def drop_index(self, index_type: Literal["parents", "chunks"]) -> bool:
        """단일 인덱스 삭제."""
        index_name = self.cfg.parents_index if index_type == "parents" else self.cfg.chunks_index

        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)
        return True

    def drop_all(self) -> dict[str, bool]:
        """모든 인덱스 삭제."""
        return {
            "parents": self.drop_index("parents"),
            "chunks": self.drop_index("chunks"),
        }

    def recreate_all(self) -> dict[str, bool]:
        """모든 인덱스 재생성 (drop + create)."""
        self.drop_all()
        return self.create_all(skip_existing=False)


# =============================================================================
# CLI
# =============================================================================


def _format_bytes(size_bytes: int | float) -> str:
    """바이트를 읽기 쉬운 형식으로 변환."""
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def cmd_status(migrator: Migrator) -> int:
    """인덱스 상태 출력."""
    print("\n�� Elasticsearch Index Status")
    print("=" * 50)

    status = migrator.status()
    for idx_type, info in status.items():
        emoji = "✅" if info.exists else "❌"
        print(f"\n{emoji} {idx_type}: {info.name}")
        if info.exists:
            print(f"   Documents: {info.doc_count:,}")
            print(f"   Size: {_format_bytes(info.size_bytes)}")

    print()
    return 0


def cmd_create(migrator: Migrator) -> int:
    """인덱스 생성."""
    print("\n🔧 Creating indices...")
    results = migrator.create_all(skip_existing=True)

    for idx_type, success in results.items():
        emoji = "✅" if success else "❌"
        print(f"   {emoji} {idx_type}")

    print("\n✨ Done!")
    return 0 if all(results.values()) else 1


def cmd_drop(migrator: Migrator, confirm: bool) -> int:
    """인덱스 삭제."""
    if not confirm:
        print("\n⚠️  --confirm 플래그를 추가해야 삭제됩니다.")
        print("   이 작업은 모든 데이터를 삭제합니다!")
        return 1

    print("\n🗑️  Dropping indices...")
    results = migrator.drop_all()

    for idx_type, success in results.items():
        emoji = "✅" if success else "❌"
        print(f"   {emoji} {idx_type}")

    print("\n✨ Done!")
    return 0


def cmd_recreate(migrator: Migrator, confirm: bool) -> int:
    """인덱스 재생성."""
    if not confirm:
        print("\n⚠️  --confirm 플래그를 추가해야 재생성됩니다.")
        print("   이 작업은 모든 데이터를 삭제합니다!")
        return 1

    print("\n♻️  Recreating indices...")
    results = migrator.recreate_all()

    for idx_type, success in results.items():
        emoji = "✅" if success else "❌"
        print(f"   {emoji} {idx_type}")

    print("\n✨ Done!")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(
        description="Elasticsearch 인덱스 마이그레이션 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
환경변수:
  ES_URL             Elasticsearch URL (기본: http://localhost:9200)
  ES_USERNAME        Basic Auth 사용자명
  ES_PASSWORD        Basic Auth 비밀번호
  ES_PARENTS_INDEX   Parents 인덱스명 (기본: kb_parents_v1)
  ES_CHUNKS_INDEX    Chunks 인덱스명 (기본: kb_chunks_v1)
  EMBEDDING_DIMS     임베딩 벡터 차원수 (기본: 4096)
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="명령어")

    subparsers.add_parser("status", help="인덱스 상태 확인")
    subparsers.add_parser("create", help="인덱스 생성")

    drop_parser = subparsers.add_parser("drop", help="인덱스 삭제")
    drop_parser.add_argument("--confirm", action="store_true", help="삭제 확인 (필수)")

    recreate_parser = subparsers.add_parser("recreate", help="인덱스 재생성")
    recreate_parser.add_argument("--confirm", action="store_true", help="재생성 확인 (필수)")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    # ES 연결
    cfg = MigrationConfig.from_env()
    try:
        es = create_es_client(cfg)
        if not es.ping():
            print(f"\n❌ Elasticsearch 연결 실패: {cfg.es_url}")
            return 1
        print(f"\n🔗 Connected to: {cfg.es_url}")
    except Exception as e:
        print(f"\n❌ Elasticsearch 연결 오류: {e}")
        return 1

    migrator = Migrator(es, cfg)

    if args.command == "status":
        return cmd_status(migrator)
    elif args.command == "create":
        return cmd_create(migrator)
    elif args.command == "drop":
        return cmd_drop(migrator, args.confirm)
    elif args.command == "recreate":
        return cmd_recreate(migrator, args.confirm)

    return 1


if __name__ == "__main__":
    sys.exit(main())
