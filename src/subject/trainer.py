"""Subject 분류기 학습 스크립트

사용법:
    uv run python src/subject/trainer.py configs/config.yaml
"""

import sys
from pathlib import Path

from rich.console import Console
from sklearn.metrics import classification_report, f1_score

from embedding import create_embedder, create_store
from subject.classifiers import create_classifier
from subject.data_loader import load_labeled_data, prepare_embedding_texts, split_train_val
from utils.config_loader import SubjectClassifierConfig

console = Console()


def main(config_path: str) -> None:
    """분류기 학습 메인 함수

    Args:
        config_path: 설정 파일 경로
    """
    config = SubjectClassifierConfig.from_yaml(config_path)

    console.print("=" * 50, style="bold blue")
    console.print("Subject 분류기 학습 시작", style="bold blue")
    console.print("=" * 50, style="bold blue")

    # 1. 데이터 로드
    console.print("\n[1/5] 학습 데이터 로드 중...", style="bold")
    df = load_labeled_data(config.train_data)
    console.print(f"  - 총 {len(df)}개 샘플 로드")
    console.print(f"  - Subject 카테고리: {df['subject'].nunique()}개")

    # 2. 임베딩 생성 또는 캐시 로드
    console.print("\n[2/5] 임베딩 생성 중...", style="bold")
    store = create_store(store_type="file", cache_dir=config.embeddings_cache)
    cached = store.load_bulk("train_embeddings.npz")

    embedding_config = {
        "provider": config.embedding_provider,
        "model": config.embedding_model,
        "batch_size": config.embedding_batch_size,
    }

    if cached and cached["model_name"] == config.embedding_model:
        console.print("  - 캐시된 임베딩 사용", style="green")
        embeddings = cached["embeddings"]
    else:
        console.print(f"  - {config.embedding_provider} 모델로 새로 임베딩 생성")
        embedder = create_embedder(embedding_config)
        texts = prepare_embedding_texts(df)
        embeddings = embedder.embed_batch(texts)

        # 캐시 저장
        store.save_bulk(
            embeddings=embeddings,
            ids=df["id"].tolist(),
            model_name=config.embedding_model,
            filename="train_embeddings.npz",
        )
        console.print("  - 임베딩 캐시 저장 완료", style="green")

    console.print(f"  - 임베딩 shape: {embeddings.shape}")

    # 3. 학습/검증 분리
    console.print("\n[3/5] 데이터 분리 중...", style="bold")
    labels = df["subject"].values
    X_train, X_val, y_train, y_val = split_train_val(
        embeddings=embeddings,
        labels=labels,
        val_ratio=0.2,
        seed=42,
    )
    console.print(f"  - 학습 데이터: {len(X_train)}개")
    console.print(f"  - 검증 데이터: {len(X_val)}개")

    # 4. 분류기 학습
    console.print("\n[4/5] 분류기 학습 중...", style="bold")
    classifier_config = {
        "type": config.classifier_type,
        "knn": {
            "n_neighbors": config.knn_n_neighbors,
            "metric": config.knn_metric,
            "weights": config.knn_weights,
        },
        "svm": {
            "kernel": config.svm_kernel,
            "C": config.svm_c,
            "probability": config.svm_probability,
        },
    }
    classifier = create_classifier(classifier_config)
    train_result = classifier.fit(X_train, y_train)
    console.print(f"  - 분류기 타입: {config.classifier_type}")
    console.print(f"  - 클래스 수: {train_result['n_classes']}")

    # 5. 검증 평가
    console.print("\n[5/5] 검증 평가 중...", style="bold")
    y_pred = classifier.predict(X_val)

    f1_macro = f1_score(y_val, y_pred, average="macro")
    f1_weighted = f1_score(y_val, y_pred, average="weighted")
    accuracy = (y_val == y_pred).mean()

    console.print(f"  - Accuracy: {accuracy:.4f}")
    console.print(f"  - F1 (macro): {f1_macro:.4f}")
    console.print(f"  - F1 (weighted): {f1_weighted:.4f}")

    # 상세 리포트
    console.print("\n[Classification Report]", style="bold")
    report = classification_report(y_val, y_pred, zero_division=0)
    console.print(report)

    # 모델 저장
    Path(config.classifier_model).parent.mkdir(parents=True, exist_ok=True)
    classifier.save(config.classifier_model)
    console.print(f"\n분류기 저장 완료: {config.classifier_model}", style="green bold")

    console.print("\n" + "=" * 50, style="bold blue")
    console.print("Subject 분류기 학습 완료!", style="bold blue")
    console.print("=" * 50, style="bold blue")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("Usage: python trainer.py <config_path>", style="red")
        sys.exit(1)

    main(sys.argv[1])
