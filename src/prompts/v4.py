from .base import BasePrompt


class V4Prompt(BasePrompt):
    """V4: English version of V3 prompt"""

    PROMPT_NO_QUESTION_PLUS = """### Passage:
{paragraph}

### Question:
{question}

### Choices:
{choices}

---
Based on the passage above, choose the most appropriate answer to the question from the choices {choice_range}.

Answer:"""

    PROMPT_QUESTION_PLUS = """### Passage:
{paragraph}

### Question:
{question}

### Additional Material:
{question_plus}

### Choices:
{choices}

---
Based on both the passage and the additional material above, choose the most appropriate answer to the question from the choices {choice_range}.

Answer:"""

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert evaluator for the Korean College Scholastic Ability Test (CSAT). "
            "You critically and logically analyze every passage and avoid being misled by attractive distractors, "
            "selecting the most appropriate answer strictly based on the evidence in the passage. "
            "Without explaining your reasoning, you must immediately provide only the number of the correct answer, "
            "so focus on the core meaning of the passage."
        )

    def make_user_prompt(self, row: dict) -> str:
        choices = row["choices"]
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(choices)])
        choice_range = ", ".join([str(n) for n in range(1, len(choices) + 1)])

        if row.get("question_plus"):
            return self.PROMPT_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
                choice_range=choice_range,
            )
        else:
            return self.PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
                choice_range=choice_range,
            )
