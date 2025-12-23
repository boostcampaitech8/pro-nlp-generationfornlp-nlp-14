"""분류기 모듈

Subject 분류를 위한 분류기 모듈.
"""

from subject.classifiers.base_classifier import BaseClassifier
from subject.classifiers.knn_classifier import KNNClassifier
from subject.classifiers.svm_classifier import SVMClassifier


def create_classifier(config: dict) -> BaseClassifier:
    """설정에 따라 적절한 분류기 생성

    Args:
        config: 분류기 설정 딕셔너리
            - type: 분류기 타입 ("knn", "svm")
            - knn: kNN 설정 (n_neighbors, metric, weights)
            - svm: SVM 설정 (kernel, C, probability)

    Returns:
        BaseClassifier 구현체

    Raises:
        ValueError: 알 수 없는 classifier type인 경우
        KeyError: 필수 설정 키가 없는 경우
    """
    classifier_type = config["type"]

    if classifier_type == "knn":
        knn_config = config["knn"]
        return KNNClassifier(
            n_neighbors=knn_config["n_neighbors"],
            metric=knn_config["metric"],
            weights=knn_config["weights"],
        )
    elif classifier_type == "svm":
        svm_config = config["svm"]
        return SVMClassifier(
            kernel=svm_config["kernel"],
            C=svm_config["C"],
            probability=svm_config["probability"],
        )
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


__all__ = [
    "BaseClassifier",
    "KNNClassifier",
    "SVMClassifier",
    "create_classifier",
]
