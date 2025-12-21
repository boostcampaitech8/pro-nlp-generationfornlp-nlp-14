import torch
from langchain_core.runnables import chain

from utils import get_choice_token_ids, logits_to_prediction


def create_forward(model, tokenizer):
    device = next(model.parameters()).device

    @chain
    @torch.inference_mode()
    def forward(data: dict):
        messages = data["messages"]
        outputs = model(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(device)
        )
        return {"data": data, "outputs": outputs}

    return forward


def create_choice_scorer(tokenizer):
    @chain
    def choice_scorer(ctx):
        data = ctx["data"]
        outputs = ctx["outputs"]
        len_choices = data["len_choices"]
        logits = outputs.logits[:, -1].flatten().cpu()
        choice_ids = get_choice_token_ids(tokenizer, len_choices)
        target_logits = logits[choice_ids]
        return {"data": data, "score": target_logits}

    return choice_scorer


@chain
def decode_prediction(ctx: dict) -> dict:
    data = ctx["data"]
    pred = logits_to_prediction(ctx["score"], data["len_choices"])
    return {"data": data, "pred": pred}


@chain
def format_row(ctx: dict):
    data = ctx["data"]
    pred_row = {"id": data["id"], "answer": ctx["pred"]}
    return pred_row
