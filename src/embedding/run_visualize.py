"""임베딩 시각화 스크립트

생성된 임베딩을 시각화하여 subject 분포를 확인합니다.

사용법:
    uv run python src/embedding/run_visualize.py configs/config.yaml
    uv run python src/embedding/run_visualize.py configs/config.yaml --method umap
    uv run python src/embedding/run_visualize.py configs/config.yaml --method tsne
    uv run python src/embedding/run_visualize.py configs/config.yaml --method pca
"""

import argparse
import sys
from typing import Literal

from rich.console import Console
from rich.table import Table

from embedding import create_store
from embedding.visualize import print_cluster_stats, visualize_embeddings
from subject.data_loader import load_labeled_data
from utils.config_loader import SubjectClassifierConfig

console = Console()


def main(
    config_path: str,
    method: Literal["umap", "tsne", "pca"],
) -> None:
    """임베딩 시각화 메인 함수

    Args:
        config_path: 설정 파일 경로
        method: 차원 축소 방법
    """
    config = SubjectClassifierConfig.from_yaml(config_path)

    console.print("=" * 50, style="bold blue")
    console.print("임베딩 시각화 시작", style="bold blue")
    console.print("=" * 50, style="bold blue")

    # 1. 임베딩 로드
    console.print("\n[1/3] 임베딩 로드 중...", style="bold")
    store = create_store(store_type="file", cache_dir=config.embeddings_cache)
    cached = store.load_bulk("train_embeddings.npz")

    if not cached:
        console.print(
            "  [ERROR] 임베딩 캐시가 없습니다. 먼저 'make embedding'을 실행하세요.",
            style="red bold",
        )
        sys.exit(1)

    embeddings = cached["embeddings"]
    console.print(f"  - 임베딩 shape: {embeddings.shape}")
    console.print(f"  - 모델: {cached['model_name']}")

    # 2. 라벨 로드
    console.print("\n[2/3] Subject 라벨 로드 중...", style="bold")
    df = load_labeled_data(config.train_data)
    labels = df["subject"].values
    console.print(f"  - 샘플 수: {len(labels)}")

    # Subject 분포 테이블 출력
    stats = print_cluster_stats(labels)
    console.print(f"  - Subject 종류: {len(stats)}개")

    table = Table(title="Subject 분포", show_header=True)
    table.add_column("Subject", style="cyan")
    table.add_column("Count", justify="right", style="green")
    table.add_column("Ratio", justify="right", style="yellow")

    total = len(labels)
    for subject, count in stats.items():
        ratio = f"{count / total * 100:.1f}%"
        table.add_row(subject, str(count), ratio)

    console.print(table)

    # 3. 시각화 생성
    console.print(f"\n[3/3] {method.upper()} 시각화 생성 중...", style="bold")
    output_dir = f"{config.embeddings_cache}/visualizations"

    saved_files = visualize_embeddings(
        embeddings=embeddings,
        labels=labels,
        output_dir=output_dir,
        method=method,
        random_state=42,
    )

    console.print("\n생성된 파일:", style="bold green")
    for dim, path in saved_files.items():
        console.print(f"  - {dim}: {path}")

    console.print("\n" + "=" * 50, style="bold blue")
    console.print("임베딩 시각화 완료!", style="bold blue")
    console.print("=" * 50, style="bold blue")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="임베딩 시각화")
    parser.add_argument("config_path", help="설정 파일 경로")
    parser.add_argument(
        "--method",
        choices=["umap", "tsne", "pca"],
        default="umap",
        help="차원 축소 방법 (기본값: umap)",
    )

    args = parser.parse_args()
    main(config_path=args.config_path, method=args.method)
