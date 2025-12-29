from abc import ABC, abstractmethod


class BasePrompt(ABC):
    """프롬프트 생성 전략의 기본 클래스"""

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System Prompt 반환 (기본값)"""
        pass

    def make_system_prompt(self, row: dict) -> str:
        """System Prompt 생성 (행 데이터 기반) - 기본적으로 system_prompt 프로퍼티 반환"""
        return self.system_prompt

    @abstractmethod
    def make_user_prompt(self, row: dict) -> str:
        """User Prompt 생성 (행 데이터 기반)"""
        pass
