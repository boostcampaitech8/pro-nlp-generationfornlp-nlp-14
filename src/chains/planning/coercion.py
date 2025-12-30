from schemas.retrieval.plan import RetrievalPlan


def _coerce_plan(x) -> RetrievalPlan:
    if isinstance(x, RetrievalPlan):
        return x
    if isinstance(x, dict):
        return RetrievalPlan(**x)
    raise TypeError(f"Unexpected plan output type: {type(x)}")
