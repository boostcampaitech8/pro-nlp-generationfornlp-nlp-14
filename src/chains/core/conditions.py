"""
Condition 팩토리 함수.

Lambda 조건문을 대체하여 state 기반 조건 체크를 수행하는
재사용 가능한 chain을 생성합니다.
"""

from langchain_core.runnables import chain


def is_shorter_than(*keys, max_chars: int):
    """
    State에서 값을 가져와 길이를 확인하는 팩토리 함수.

    Args:
        *keys: state에서 값을 가져올 키들 (중첩 가능)
        max_chars: 최대 문자 수

    Returns:
        Chain function that returns bool

    Example:
        >>> is_shorter_than("paragraph", max_chars=100)
        >>> is_shorter_than("data", "paragraph", max_chars=100)  # nested
    """

    @chain
    def _check(state) -> bool:
        value = state
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                return True
        return len((value or "").strip()) < max_chars

    return _check
