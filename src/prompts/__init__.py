from .base import BasePrompt
from .v1 import V1Prompt
from .v2 import V2Prompt
from .v3 import V3Prompt
from .v4 import V4Prompt
from .v5 import V5Prompt
from .v6 import V6Prompt
from .v7 import V7Prompt

PROMPT_REGISTRY = {
    "v1": V1Prompt(),
    "v2": V2Prompt(),
    "v3": V3Prompt(),
    "v4": V4Prompt(),
    "v5": V5Prompt(),
    "v6": V6Prompt(),
    "v7": V7Prompt(),
}


def get_prompt_manager(name: str) -> BasePrompt:
    """이름으로 프롬프트 매니저 반환"""
    return PROMPT_REGISTRY.get(name, V1Prompt())
