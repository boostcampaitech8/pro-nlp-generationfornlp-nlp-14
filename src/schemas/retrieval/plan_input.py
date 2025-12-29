from dataclasses import dataclass

from schemas.processed_question import ProcessedQuestion


@dataclass
class PlanInput(ProcessedQuestion):
    """
    RetrievalPlan을 생성(planner LLM 호출)하기 위한 입력 스키마.

    ProcessedData와 필드가 동일하더라도 planner 단계의 '입력 계약'을 별도 타입으로 두면:
    - planner 프롬프트가 요구하는 필드를 기억하기 쉬워지고,
    - planner 전용 정책/전처리(예: 텍스트 truncate) 등을 추가하기 용이하다.
    """

    def __post_init__(self):
        if self.len_choices == 0:
            self.len_choices = len(self.choices)
        elif self.len_choices != len(self.choices):
            raise ValueError("len_choices must match len(choices)")
