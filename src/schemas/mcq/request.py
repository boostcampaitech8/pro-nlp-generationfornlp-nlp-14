from typing import Literal, TypedDict

from langchain_core.messages.base import BaseMessage


class LocalModelMessage(TypedDict):
    """loacl model의 이 원하는 메시지 배열의 dict형태에 초점을 두고 맞춥니다.

    llamacpp를 사용할 경우,
    """

    role: Literal["system", "user", "assistant"]
    content: str


class McqRequest(TypedDict):
    id: str
    messages: list[BaseMessage]
    len_choices: int
