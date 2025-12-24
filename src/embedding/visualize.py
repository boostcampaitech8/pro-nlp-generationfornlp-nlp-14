"""임베딩 시각화 모듈

고차원 임베딩을 2D/3D로 축소하여 subject별 분포를 시각화.
UMAP, t-SNE, PCA 지원.
"""

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def reduce_dimensions(
    embeddings: np.ndarray,
    method: Literal["umap", "tsne", "pca"],
    n_components: int,
    random_state: int,
) -> np.ndarray:
    """고차원 임베딩을 저차원으로 축소

    Args:
        embeddings: 임베딩 배열 (n_samples, embedding_dim)
        method: 차원 축소 방법 (umap, tsne, pca)
        n_components: 목표 차원 수 (2 또는 3)
        random_state: 랜덤 시드

    Returns:
        축소된 임베딩 배열 (n_samples, n_components)
    """
    if method == "umap":
        try:
            import umap
        except ImportError as e:
            raise ImportError(
                "UMAP을 사용하려면 umap-learn 패키지를 설치하세요: "
                "uv add umap-learn"
            ) from e

        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
        )
        return reducer.fit_transform(embeddings)

    elif method == "tsne":
        # t-SNE는 perplexity가 샘플 수보다 작아야 함
        perplexity = min(30, len(embeddings) - 1)
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=perplexity,
            n_iter=1000,
            metric="cosine",
        )
        return reducer.fit_transform(embeddings)

    elif method == "pca":
        reducer = PCA(
            n_components=n_components,
            random_state=random_state,
        )
        return reducer.fit_transform(embeddings)

    else:
        raise ValueError(f"지원하지 않는 method: {method}")


def plot_embeddings_2d(
    reduced: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str,
    figsize: tuple[int, int],
) -> None:
    """2D 임베딩 산점도 시각화

    Args:
        reduced: 2D 축소된 임베딩 (n_samples, 2)
        labels: subject 라벨 배열
        title: 플롯 제목
        save_path: 저장 경로
        figsize: 그림 크기
    """
    plt.figure(figsize=figsize)

    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            c=[colors[i]],
            label=label,
            alpha=0.7,
            s=50,
        )

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=10,
        title="Subject",
    )
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_embeddings_3d(
    reduced: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str,
    figsize: tuple[int, int],
) -> None:
    """3D 임베딩 산점도 시각화

    Args:
        reduced: 3D 축소된 임베딩 (n_samples, 3)
        labels: subject 라벨 배열
        title: 플롯 제목
        save_path: 저장 경로
        figsize: 그림 크기
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            reduced[mask, 2],
            c=[colors[i]],
            label=label,
            alpha=0.7,
            s=50,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.legend(
        bbox_to_anchor=(1.15, 1),
        loc="upper left",
        fontsize=9,
        title="Subject",
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    method: Literal["umap", "tsne", "pca"],
    random_state: int,
) -> dict[str, str]:
    """임베딩 시각화 메인 함수

    2D와 3D 시각화를 모두 생성합니다.

    Args:
        embeddings: 임베딩 배열 (n_samples, embedding_dim)
        labels: subject 라벨 배열
        output_dir: 출력 디렉토리
        method: 차원 축소 방법 (umap, tsne, pca)
        random_state: 랜덤 시드

    Returns:
        생성된 파일 경로 딕셔너리
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    method_upper = method.upper()
    saved_files = {}

    # 2D 시각화
    reduced_2d = reduce_dimensions(
        embeddings=embeddings,
        method=method,
        n_components=2,
        random_state=random_state,
    )

    save_path_2d = str(output_path / f"embedding_{method}_2d.png")
    plot_embeddings_2d(
        reduced=reduced_2d,
        labels=labels,
        title=f"Subject Embedding Distribution ({method_upper} 2D)",
        save_path=save_path_2d,
        figsize=(12, 8),
    )
    saved_files["2d"] = save_path_2d

    # 3D 시각화 (t-SNE는 3D 지원하지 않으므로 스킵)
    if method != "tsne":
        reduced_3d = reduce_dimensions(
            embeddings=embeddings,
            method=method,
            n_components=3,
            random_state=random_state,
        )

        save_path_3d = str(output_path / f"embedding_{method}_3d.png")
        plot_embeddings_3d(
            reduced=reduced_3d,
            labels=labels,
            title=f"Subject Embedding Distribution ({method_upper} 3D)",
            save_path=save_path_3d,
            figsize=(12, 10),
        )
        saved_files["3d"] = save_path_3d

    return saved_files


def print_cluster_stats(labels: np.ndarray) -> dict[str, int]:
    """Subject별 샘플 수 통계 출력

    Args:
        labels: subject 라벨 배열

    Returns:
        subject별 샘플 수 딕셔너리
    """
    unique, counts = np.unique(labels, return_counts=True)
    stats = dict(zip(unique, counts))
    return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))
