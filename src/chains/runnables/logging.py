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


def tap(path: str | Path, transform=None):
    """
    JSON Line logger (tap pattern).

    입력 데이터를 로그 파일에 기록하면서 그대로 통과시킵니다.
    기존 silent fail 대신 warning log를 남깁니다.

    Args:
        path: 로그 파일 경로
        transform: 로깅 전에 state를 변환하는 함수 (optional)

    Returns:
        Runnable (via @chain decorator)

    Example:
        >>> # State 전체 로깅
        >>> chain = planner | tap("log/plans.jsonl")
        >>>
        >>> # 특정 필드만 로깅 (transform 사용)
        >>> chain = planner | tap(
        ...     "log/plans.jsonl",
        ...     transform=lambda s: {"id": s["data"]["id"], "plan": s["plan"]}
        ... )
    """
    log_file = Path(path)

    @chain
    def _tap(x: Any) -> Any:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            # Transform 적용 (있는 경우)
            log_data = transform(x) if transform else x
            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_to_jsonable(log_data), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log to {path}: {e}")
        return x  # 원본 state 그대로 반환

    return _tap
