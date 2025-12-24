import sys
from pathlib import Path

from data.preprocess.sgkfold import create_folds
from data.preprocess.source_tagger import add_source_labels
from utils.config_loader import PreprocessConfig


def main(config: PreprocessConfig) -> None:
    """
    데이터 전처리 파이프라인 실행
    1. 소스 태깅: train.csv -> train_source_labeled.csv
    2. Fold 분할: train_source_labeled.csv -> train_with_folds.csv

    Args:
        config: 전처리 설정 객체
    """

    print("=" * 70)
    print("데이터 전처리 파이프라인 시작")
    print("=" * 70)
    print(f"입력 파일: {config.train_file}")
    print(f"소스 라벨링 출력: {config.source_labeled_file}")
    print(f"최종 출력: {config.with_folds_file}")
    print(
        f"Fold 설정: n_splits={config.n_splits}, eval_fold={config.eval_fold}, seed={config.random_seed}"
    )
    print("=" * 70)

    # 1단계: 소스 태깅
    print("\n[1/2] 소스 태깅 수행 중...")
    df_labeled = add_source_labels(
        input_path=config.train_file, output_path=config.source_labeled_file
    )

    # 2단계: Stratified Group K-Fold 분할
    print("\n[2/2] Stratified Group K-Fold 분할 수행 중...")
    df_with_folds = create_folds(
        df=df_labeled,
        n_splits=config.n_splits,
        eval_fold_idx=config.eval_fold,
        random_seed=config.random_seed,
    )

    # 최종 결과 저장
    # 작업용 컬럼 제거
    export_df = df_with_folds.drop(
        columns=["strat_label", "para_len_bin", "para_len", "answer"], errors="ignore"
    )

    Path(config.with_folds_file).parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(config.with_folds_file, index=False)

    # train/eval split 파일 생성
    train_df = export_df[export_df["fold"] != config.eval_fold].copy()
    eval_df = export_df[export_df["fold"] == config.eval_fold].copy()

    train_df.to_csv(config.train_split_file, index=False)
    eval_df.to_csv(config.eval_split_file, index=False)

    print("\n" + "=" * 70)
    print("데이터 전처리 완료!")
    print("=" * 70)
    print(f"✅ 소스 라벨링 파일: {config.source_labeled_file}")
    print(f"✅ Fold 분할 파일: {config.with_folds_file}")
    print(f"✅ Train Split 파일: {config.train_split_file} ({len(train_df)}개)")
    print(f"✅ Eval Split 파일: {config.eval_split_file} ({len(eval_df)}개)")
    print(f"   - 총 샘플 수: {len(export_df)}")
    print(f"   - Fold 개수: {config.n_splits}")
    print(f"   - 평가 Fold: {config.eval_fold}")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <config_path>")
        print("Example: python preprocess.py configs/preprocess.yaml")
        sys.exit(1)

    config_path = sys.argv[1]
    config = PreprocessConfig.from_yaml(config_path)
    main(config)
