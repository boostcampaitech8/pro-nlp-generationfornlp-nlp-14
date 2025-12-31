import json
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableLambda


def _to_jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return asdict(x)
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def tap(path: str, build_record: Callable[[Any], dict]) -> RunnableLambda:
    p = Path(path)

    def _tap(x: Any) -> Any:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            rec = _to_jsonable(build_record(x))
            rec = {**rec}
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return x

    return RunnableLambda(_tap)
