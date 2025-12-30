from langchain_core.runnables import chain

from schemas.mcq.context import DecodedContext
from schemas.mcq.rows import PredRow, ScoreRow


@chain
def format_rows(ctx: DecodedContext) -> tuple[PredRow, ScoreRow]:
    data = ctx["data"]
    pred_row: PredRow = {"id": data["id"], "answer": ctx["pred"]}
    score_row: ScoreRow = {"id": data["id"], "score": ctx["score"]}
    return (pred_row, score_row)
