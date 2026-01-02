"""인덱스 전체 다운로드 스크립트.

Elasticsearch의 모든 Parent/Chunk 데이터를 JSONL 파일로 백업합니다.
스트리밍 방식으로 저장하여 메모리 사용량을 최소화합니다.
기본 저장 경로: ./data/index_backup/

Usage:
    PYTHONPATH=src uv run python src/indexing/download_index.py
"""

import argparse
import logging
from pathlib import Path

from vectorstore.backup import VectorStoreBackup
from vectorstore.client import create_es_client
from vectorstore.config import ESConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ES 인덱스 전체 다운로드 (JSONL)")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./data/index_backup",
        help="백업 디렉토리 경로 (기본: ./data/index_backup)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    logger.info(f"백업 시작: {output_dir}")

    try:
        cfg = ESConfig()
        es = create_es_client(cfg)
        backup = VectorStoreBackup(es, cfg)

        parents_count, chunks_count = backup.backup_to_jsonl(output_dir)

        logger.info(f"백업 완료: Parents {parents_count}개, Chunks {chunks_count}개")
        logger.info(f"파일 위치: {output_dir.absolute()}")

    except Exception as e:
        logger.error(f"백업 실패: {e}")
        raise


if __name__ == "__main__":
    main()
