from dataclasses import dataclass


@dataclass
class QuestionBase:
    """
    MCQ 파이프라인에서 공통으로 사용하는 '원본(raw) 필수 필드' 베이스 스키마.

    이 클래스는 다음을 목적으로 한다:
    - 데이터가 최소한 어떤 필드를 반드시 가져야 하는지(계약)를 고정한다.
    - choices는 list[str]로 보관하되, 프롬프트에 바로 넣기 좋은 문자열 형태를
    @property로 제공해 prompt builder가 필드/포맷을 기억하지 않아도 되게 한다.

    Fields:
        id: 샘플 식별자.
        paragraph: 지문(본문) 텍스트.
        question: 질문 텍스트.
        choices: 선택지 리스트. 인덱스 순서가 정답 번호(1..N)와 대응한다고 가정한다.
        question_plus: 질문과 선택지 사이에 추가로 삽입되는 보조 문구(옵션).
    """

    id: str
    paragraph: str
    question: str
    choices: list[str]
    question_plus: str | None = None

    @property
    def choices_text(self) -> str:
        """
        프롬프트에 넣기 좋은 선택지 문자열 표현.

        Returns:
            예)
            1. ...
            2. ...
            3. ...
        """
        # 프롬프트에 넣기 좋은 형태
        return "\n".join([f"{i + 1}. {c}" for i, c in enumerate(self.choices)])

    @property
    def question_plus_block(self) -> str:
        """
        question_plus를 프롬프트에 삽입하기 위한 블록 문자열.

        - question_plus가 존재하면 QUESTION과 CHOICES 사이에 넣을 수 있도록
            구분 태그를 포함한 문자열을 반환한다.
        - 없으면 빈 문자열을 반환한다.

        Returns:
            question_plus가 있을 때:
                "\\n[QUESTION_PLUS]\\n...\\n"
            없을 때:
                ""
        """
        qp = (self.question_plus or "").strip()
        return f"\n[QUESTION_PLUS]\n{qp}\n" if qp else ""
