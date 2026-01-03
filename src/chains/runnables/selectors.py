"""
Selector 및 Constant 팩토리 함수.

Lambda 함수를 대체하여 state에서 값을 선택하거나 상수를 반환하는
재사용 가능한 chain을 생성합니다.
"""

from langchain_core.runnables import chain


def selector(*keys):
    """
    State에서 중첩된 키를 선택하는 팩토리 함수.

    Args:
        *keys: 선택할 키들 (중첩 가능)

    Returns:
        Chain function

    Example:
        >>> selector("data")  # lambda x: x["data"]
        >>> selector("data", "paragraph")  # lambda x: x["data"]["paragraph"]
    """

    @chain
    def _selector(state):
        value = state
        for key in keys:
            value = value[key]
        return value

    return _selector


def constant(value):
    """
    항상 같은 값을 반환하는 팩토리 함수.

    Args:
        value: 반환할 상수 값

    Returns:
        Chain function

    Example:
        >>> constant("")  # lambda _: ""
        >>> constant(0)   # lambda _: 0
    """

    @chain
    def _constant(_):
        return value

    return _constant
