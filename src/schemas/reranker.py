from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class DocScore:
    """
    리랭커 API의 개별 문서 점수 응답 스키마.

    Attributes:
        index: 원본 문서 리스트에서의 인덱스.
        score: 모델이 계산한 질문과의 관련성 점수 (0~1).
    """

    index: int
    score: float


@dataclass
class RerankResponse:
    """
    리랭킹 체인의 중간 결과 스키마.

    Attributes:
        results: 각 쿼리 그룹별로 정렬 및 점수가 부여된 문서 리스트들의 리스트.
                 (list[list[Document]] 구조를 캡슐화)
    """

    results: list[list[Document]]

    @property
    def total_count(self) -> int:
        """모든 그룹에 포함된 전체 문서의 개수를 반환합니다."""
        return sum(len(group) for group in self.results)


@dataclass
class RetrievalResponse:
    """
    리전 및 리랭킹 파이프라인의 최종 출력 규격 스키마.

    QA 체인으로 넘어가기 직전의 '약속된' 형태입니다.

    Attributes:
        question: 리랭킹에 사용된 (또는 원본) 질문 문자열.
        context: 평탄화 전략(Merging)을 통해 하나로 합쳐진 최종 지문 텍스트.
    """

    question: str
    context: str
