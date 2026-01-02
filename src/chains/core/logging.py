"""
Pipeline 전반에서 사용되는 로깅 유틸리티.

기존 chains/nodes/tap.py를 개선하여 범용적인 tap logger를 제공합니다.
구체적인 로깅 로직은 사용처에서 lambda나 transform으로 처리합니다.
"""

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from langchain_core.runnables import chain

logger = logging.getLogger(__name__)


def _to_jsonable(x: Any) -> Any:
    """
    Dataclass, dict, list를 JSON 직렬화 가능한 형태로 변환.
    """
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def tap(path: str | Path):
    """
    JSON Line logger (tap pattern).

    입력 데이터를 로그 파일에 기록하면서 그대로 통과시킵니다.
    기존 silent fail 대신 warning log를 남깁니다.

    Args:
        path: 로그 파일 경로

    Returns:
        Runnable (via @chain decorator)

    Example:
        >>> # State 전체 로깅
        >>> chain = planner | tap("log/plans.jsonl")
        >>>
        >>> # 특정 필드만 로깅
        >>> extract = RunnableLambda(lambda s: {"id": s["data"]["id"], "plan": s["plan"]})
        >>> chain = planner | extract | tap("log/plans.jsonl")
    """
    log_file = Path(path)

    @chain
    def _tap(x: Any) -> Any:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_to_jsonable(x), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log to {path}: {e}")
        return x

    return _tap
