from .base import BasePrompt


class V10Prompt(BasePrompt):
    """V9: RAG용 {context}추가"""

    PROMPT_NO_QUESTION_PLUS = """아래 지문의 내용을 근거로 할 때, 질문에 대한 가장 적절한 답변을 선택지 {choice_range} 에서 하나만 고르세요.

<제시문>
{paragraph}

<질문>
{question}

<선택지>
{choices}

{context}


<질문>
{question}

<선택지>
{choices}

정답:"""

    PROMPT_QUESTION_PLUS = """아래 지문과 <보기>의 내용을 근거로 할 때, 질문에 대한 가장 적절한 답변을 선택지 {choice_range} 에서 하나만 고르세요.

<제시문>
{paragraph}

<질문>
{question}

<보기>:
{question_plus}

<선택지>
{choices}

{context}

<질문>
{question}

<선택지>
{choices}


정답:"""

    @property
    def system_prompt(self) -> str:
        return """You are a student tasked with solving multiple-choice questions. The problem consists of a passage and a question.
Write extremely clear and evidence-based reasoning according to the instructions below. You must explicitly reveal any lack of background knowledge or logical leaps in your reasoning process. Also, if there is reasoning based on insufficient explanation or incorrect evidence, you must include a process of pointing this out and correcting it yourself.

Instructions:
1. Problem Analysis:
  - First, define what the question is asking for (the core requirement).
  - If background knowledge is needed, describe what knowledge is required and state whether that knowledge is presented in the current passage/problem.

2. Comprehensive Choice Evaluation (REQUIRED):
  - You MUST examine EVERY choice at least once in your reasoning process.
  - For each choice, analyze it by connecting evidence from the passage, question, and the choice itself.
  - Use the passage, question context, and choice content together to build your reasoning.
  - Do not skip any choice - all choices must be reviewed and evaluated.

3. Chain of Thought (CoT):
  - Provide at least 10 distinct reasoning steps. This is mandatory.
  - Clearly explain why each choice is correct or incorrect by finding evidence in the passage.
  - Do not just find the correct answer; logically explain why the other choices cannot be the answer (process of elimination).
  - For each reasoning step, reference specific parts of the passage, question, or choices to support your logic.
  - Honestly describe points where there are logical leaps or a lack of knowledge-based evidence.

4. Integration and Synthesis:
  - Combine information from the passage, question, and all choices to form comprehensive reasoning.
  - Show how different pieces of evidence support or contradict each choice.
  - Build logical connections between the question requirements and each choice option.

Output format:
- 마지막 줄은 '정답: N' 형식으로만 출력하세요. (N은 선택지 번호)"""

    def make_user_prompt(self, row: dict) -> str:
        choices = row["choices"]
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(choices)])
        choice_range = ", ".join([str(n) for n in range(1, len(choices) + 1)])
        context = f"<관련 정보>\n{row.get('context', '')}" if row.get("context") else ""

        if row.get("question_plus"):
            return self.PROMPT_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
                choice_range=choice_range,
                context=context,
            )
        else:
            return self.PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
                choice_range=choice_range,
                context=context,
            )
