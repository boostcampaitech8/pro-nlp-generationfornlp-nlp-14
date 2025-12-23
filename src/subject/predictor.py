"""Subject 런타임 예측기

학습된 분류기를 사용하여 새 입력에 대해 Subject를 예측.
"""

import numpy as np

from embedding import create_embedder, BaseEmbedder
from subject.classifiers import BaseClassifier, KNNClassifier, SVMClassifier
from utils.config_loader import SubjectClassifierConfig


class SubjectPredictor:
    """런타임 Subject 예측기

    추론 시 새 입력을 임베딩하고 분류하는 파이프라인.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        classifier: BaseClassifier,
    ):
        """예측기 초기화

        Args:
            embedder: 임베딩 모델
            classifier: 학습된 분류기
        """
        self._embedder = embedder
        self._classifier = classifier

    @classmethod
    def from_config(cls, config: SubjectClassifierConfig) -> "SubjectPredictor":
        """설정에서 예측기 생성

        Args:
            config: SubjectClassifierConfig 객체

        Returns:
            SubjectPredictor 인스턴스
        """
        # 임베더 생성
        embedding_config = {
            "provider": config.embedding_provider,
            "model": config.embedding_model,
            "batch_size": config.embedding_batch_size,
        }
        embedder = create_embedder(embedding_config)

        # 분류기 로드
        classifier_type = config.classifier_type
        if classifier_type == "knn":
            classifier = KNNClassifier.load(config.classifier_model)
        elif classifier_type == "svm":
            classifier = SVMClassifier.load(config.classifier_model)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        return cls(embedder=embedder, classifier=classifier)

    def predict(
        self,
        paragraph: str,
        question: str,
        return_proba: bool = False,
    ) -> dict:
        """단일 입력에 대한 Subject 예측

        Args:
            paragraph: 지문
            question: 질문
            return_proba: 확률값 반환 여부

        Returns:
            예측 결과 딕셔너리
            - subject: 예측된 Subject
            - confidence: 신뢰도 (return_proba=True일 때)
            - probabilities: 클래스별 확률 (return_proba=True일 때)
        """
        # 텍스트 결합 및 임베딩
        combined_text = self._embedder.combine_paragraph_question(paragraph, question)
        embedding = self._embedder.embed(combined_text).reshape(1, -1)

        # 예측
        subject = self._classifier.predict(embedding)[0]

        result = {"subject": subject}

        if return_proba:
            proba = self._classifier.predict_proba(embedding)[0]
            classes = self._classifier.classes
            result["confidence"] = float(proba.max())
            result["probabilities"] = {
                c: float(p) for c, p in zip(classes, proba)
            }

        return result

    def predict_batch(
        self,
        paragraphs: list[str],
        questions: list[str],
    ) -> list[str]:
        """배치 입력에 대한 Subject 예측

        Args:
            paragraphs: 지문 리스트
            questions: 질문 리스트

        Returns:
            예측된 Subject 라벨 리스트
        """
        texts = [
            self._embedder.combine_paragraph_question(p, q)
            for p, q in zip(paragraphs, questions)
        ]
        embeddings = self._embedder.embed_batch(texts)
        return self._classifier.predict(embeddings).tolist()
