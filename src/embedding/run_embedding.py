"""임베딩 생성 스크립트

학습 데이터의 임베딩을 생성하고 캐시에 저장합니다.

사용법:
    uv run python src/embedding/run_embedding.py configs/config.yaml
"""

import sys
from pathlib import Path

from rich.console import Console

from embedding import create_embedder, create_store
from subject.data_loader import load_labeled_data, prepare_embedding_texts
from utils.config_loader import SubjectClassifierConfig

console = Console()


def main(config_path: str) -> None:
    """임베딩 생성 메인 함수

    Args:
        config_path: 설정 파일 경로
    """
    config = SubjectClassifierConfig.from_yaml(config_path)

    console.print("=" * 50, style="bold blue")
    console.print("임베딩 생성 시작", style="bold blue")
    console.print("=" * 50, style="bold blue")

    # 1. 데이터 로드
    console.print("\n[1/3] 학습 데이터 로드 중...", style="bold")
    df = load_labeled_data(config.train_data)
    console.print(f"  - 총 {len(df)}개 샘플 로드")

    # 2. 임베딩 저장소 설정
    console.print("\n[2/3] 임베딩 생성 중...", style="bold")
    store = create_store(store_type="file", cache_dir=config.embeddings_cache)

    # 캐시 확인
    cached = store.load_bulk("train_embeddings.npz")
    if cached and cached["model_name"] == config.embedding_model:
        console.print(f"  - 이미 캐시된 임베딩이 존재합니다: {config.embeddings_cache}", style="yellow")
        console.print(f"  - 모델: {cached['model_name']}")
        console.print(f"  - Shape: {cached['embeddings'].shape}")
        console.print("  - 새로 생성하려면 캐시 디렉토리를 삭제하세요.", style="yellow")
        return

    # 3. 임베딩 생성
    embedding_config = {
        "provider": config.embedding_provider,
        "model": config.embedding_model,
        "batch_size": config.embedding_batch_size,
    }

    console.print(f"  - Provider: {config.embedding_provider}")
    console.print(f"  - Model: {config.embedding_model}")
    console.print(f"  - Batch size: {config.embedding_batch_size}")

    embedder = create_embedder(embedding_config)
    texts = prepare_embedding_texts(df)

    console.print(f"  - {len(texts)}개 텍스트 임베딩 생성 중...")
    embeddings = embedder.embed_batch(texts)

    # 캐시 저장
    console.print("\n[3/3] 임베딩 저장 중...", style="bold")
    Path(config.embeddings_cache).mkdir(parents=True, exist_ok=True)
    store.save_bulk(
        embeddings=embeddings,
        ids=df["id"].tolist(),
        model_name=config.embedding_model,
        filename="train_embeddings.npz",
    )

    console.print(f"  - 저장 경로: {config.embeddings_cache}/train_embeddings.npz")
    console.print(f"  - 임베딩 shape: {embeddings.shape}")

    console.print("\n" + "=" * 50, style="bold blue")
    console.print("임베딩 생성 완료!", style="bold blue")
    console.print("=" * 50, style="bold blue")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("Usage: python run_embedding.py <config_path>", style="red")
        sys.exit(1)

    main(sys.argv[1])
