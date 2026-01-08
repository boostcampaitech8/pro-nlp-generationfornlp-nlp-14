"""인덱스 전체 업로드 스크립트.

다운로드된 JSONL 백업을 Elasticsearch로 복원합니다.
깨끗한 DB에 데이터를 올리는 것을 가정합니다.
기본 읽기 경로: ./data/index_backup/

Usage:
    PYTHONPATH=src uv run python src/indexing/upload_index.py           # 업로드
    PYTHONPATH=src uv run python src/indexing/upload_index.py --preview # 미리보기
"""

import argparse
import logging
from pathlib import Path

from infrastructure.vectorstore.backup import VectorStoreBackup
from infrastructure.vectorstore.client import create_es_client
from infrastructure.vectorstore.config import ESConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ES 인덱스 전체 업로드 (JSONL)")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="./data/index_backup",
        help="백업 디렉토리 경로 (기본: ./data/index_backup)",
    )
    parser.add_argument(
        "--preview",
        "-p",
        action="store_true",
        help="백업 정보만 확인하고 복원하지 않음",
    )
    args = parser.parse_args()

    backup_dir = Path(args.input)

    if not backup_dir.exists():
        logger.error(f"백업 디렉토리를 찾을 수 없습니다: {backup_dir}")
        return

    logger.info(f"백업 디렉토리: {backup_dir.absolute()}")

    try:
        cfg = ESConfig()
        es = create_es_client(cfg)
        backup = VectorStoreBackup(es, cfg)

        # 백업 정보 확인
        info = backup.get_backup_info(backup_dir)
        logger.info(f"백업 생성 시간: {info.get('created_at', 'N/A')}")
        logger.info(f"Parents: {info.get('total_parents', 0)}개")
        logger.info(f"Chunks: {info.get('total_chunks', 0)}개")

        if args.preview:
            logger.info("프리뷰 모드: 백업 정보 확인 완료")
            return

        # 복원 실행
        logger.info("업로드 시작...")
        parents_count, chunks_count = backup.restore_from_jsonl(backup_dir)

        logger.info(f"업로드 완료: Parents {parents_count}개, Chunks {chunks_count}개")

    except FileNotFoundError as e:
        logger.error(f"{e}")
    except Exception as e:
        logger.error(f"업로드 실패: {e}")
        raise


if __name__ == "__main__":
    main()
