import sys
from pathlib import Path
from typing import Any

import yaml

from data.preprocess.sgkfold import create_folds
from data.preprocess.source_tagger import add_source_labels


def load_config(config_path: str | Path) -> dict[str, Any]:
    """설정 파일 로드"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    return config


def run_preprocessing(config_path: str | Path) -> None:
    """
    데이터 전처리 파이프라인 실행
    1. 소스 태깅: train.csv -> train_source_labeled.csv
    2. Fold 분할: train_source_labeled.csv -> train_with_folds.csv

    Args:
        config_path: preprocess.yaml 경로
    """
    # 설정 로드
    config = load_config(config_path)
    preprocess_config = config["preprocess"]

    # 경로 설정
    project_root = Path(__file__).parent.parent.parent.parent
    train_file = project_root / preprocess_config["input"]["train_file"]
    source_labeled_file = project_root / preprocess_config["output"]["source_labeled_file"]
    with_folds_file = project_root / preprocess_config["output"]["with_folds_file"]
    train_split_file = project_root / preprocess_config["output"]["train_split_file"]
    eval_split_file = project_root / preprocess_config["output"]["eval_split_file"]

    # SGKFold 설정
    sgkfold_config = preprocess_config["sgkfold"]
    n_splits = sgkfold_config["n_splits"]
    eval_fold = sgkfold_config["eval_fold"]
    random_seed = sgkfold_config["random_seed"]

    print("=" * 70)
    print("데이터 전처리 파이프라인 시작")
    print("=" * 70)
    print(f"입력 파일: {train_file}")
    print(f"소스 라벨링 출력: {source_labeled_file}")
    print(f"최종 출력: {with_folds_file}")
    print(f"Fold 설정: n_splits={n_splits}, eval_fold={eval_fold}, seed={random_seed}")
    print("=" * 70)

    # 1단계: 소스 태깅
    print("\n[1/2] 소스 태깅 수행 중...")
    df_labeled = add_source_labels(input_path=train_file, output_path=source_labeled_file)

    # 2단계: Stratified Group K-Fold 분할
    print("\n[2/2] Stratified Group K-Fold 분할 수행 중...")
    df_with_folds = create_folds(
        df=df_labeled, n_splits=n_splits, eval_fold_idx=eval_fold, random_seed=random_seed
    )

    # 최종 결과 저장
    # 작업용 컬럼 제거
    export_df = df_with_folds.drop(
        columns=["strat_label", "para_len_bin", "para_len", "answer"], errors="ignore"
    )

    with_folds_file.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(with_folds_file, index=False)

    # train/eval split 파일 생성
    train_df = export_df[export_df["fold"] != eval_fold].copy()
    eval_df = export_df[export_df["fold"] == eval_fold].copy()

    train_df.to_csv(train_split_file, index=False)
    eval_df.to_csv(eval_split_file, index=False)

    print("\n" + "=" * 70)
    print("데이터 전처리 완료!")
    print("=" * 70)
    print(f"✅ 소스 라벨링 파일: {source_labeled_file}")
    print(f"✅ Fold 분할 파일: {with_folds_file}")
    print(f"✅ Train Split 파일: {train_split_file} ({len(train_df)}개)")
    print(f"✅ Eval Split 파일: {eval_split_file} ({len(eval_df)}개)")
    print(f"   - 총 샘플 수: {len(export_df)}")
    print(f"   - Fold 개수: {n_splits}")
    print(f"   - 평가 Fold: {eval_fold}")
    print("=" * 70)


def main():
    """CLI 진입점"""
    if len(sys.argv) < 2:
        print("사용법: make preprocess")
        print("예시: make preprocess")
        sys.exit(1)

    config_path = sys.argv[1]

    try:
        run_preprocessing(config_path)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
