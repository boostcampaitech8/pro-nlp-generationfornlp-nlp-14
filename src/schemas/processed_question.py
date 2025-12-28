from dataclasses import dataclass

from .question_base import QuestionBase


@dataclass
class ProcessedQuestion(QuestionBase):
    """
    전처리/가공 단계 이후의 MCQ 데이터 스키마.

    - len_choices: 추가
    """

    len_choices: int = 0

    def __post_init__(self):
        if self.len_choices == 0:
            self.len_choices = len(self.choices)
        elif self.len_choices != len(self.choices):
            raise ValueError("len_choices must match len(choices)")
