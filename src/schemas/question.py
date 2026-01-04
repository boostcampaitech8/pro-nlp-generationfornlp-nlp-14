"""전처리된 MCQ 질문 데이터 스키마."""

from typing import TypedDict

from typing_extensions import NotRequired


class PreprocessedQuestion(TypedDict):
    """
    전처리된 MCQ 질문 데이터 스키마.

    Pipeline의 입력 데이터이며, PipelineState.data 필드로 전달됩니다.

    Fields:
        id: 샘플 식별자
        paragraph: 지문(본문) 텍스트
        question: 질문 텍스트
        choices: 선택지 리스트
        question_plus: 보조 문구 (옵션)
        len_choices: 선택지 개수
    """

    id: str
    paragraph: str
    question: str
    choices: list[str]
    question_plus: NotRequired[str | None]
    len_choices: int
