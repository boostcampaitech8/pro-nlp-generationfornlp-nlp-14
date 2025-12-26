import torch
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI

from utils import get_choice_token_ids, logits_to_prediction


def create_local_forward(model, tokenizer):
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
        len_choices = data["len_choices"]
        logits = outputs.logits[:, -1].flatten().cpu()
        choice_ids = get_choice_token_ids(tokenizer, len_choices)
        target_logits = logits[choice_ids]
        return {"data": data, "outputs": target_logits}

    return forward


def create_llamacpp_forward():
    import os

    import dotenv

    dotenv.load_dotenv()
    LLAMA_CPP_SERVER_URL = os.getenv("LLAMA_CPP_SERVER_URL")
    NEG_INF = -1e30

    model = ChatOpenAI(
        base_url=LLAMA_CPP_SERVER_URL,
        api_key="NOT_NEED",
        model_name="NOTE_NEED",
        temperature=0,
        logprobs=True,
        extra_body={
            "max_tokens": 1,
        },
    )

    def _build_choice_logprobs(top_pairs, len_choices):
        choice_logprobs = torch.tensor([NEG_INF] * len_choices, dtype=torch.float)

        for tok, lp in top_pairs:
            s = tok.strip()
            if s.isdigit():
                k = int(s)
                if 1 <= k <= len_choices:
                    choice_logprobs[k - 1] = lp

        return choice_logprobs

    @chain
    def foward(data):
        _chain = model
        output = _chain.invoke(data["messages"])
        top = output.response_metadata["logprobs"]["content"][0]["top_logprobs"]
        top_pairs = [(token["token"], token["logprob"]) for token in top]
        choice_logprobs = _build_choice_logprobs(top_pairs, data["len_choices"])

        return {"data": data, "score": choice_logprobs}

    return foward


@chain
def decode_prediction(ctx: dict) -> dict:
    data = ctx["data"]
    pred = logits_to_prediction(ctx["score"], data["len_choices"])
    return {"data": data, "score": ctx["score"], "pred": pred}


@chain
def format_rows(ctx: dict):
    data = ctx["data"]
    pred_row = {"id": data["id"], "answer": ctx["pred"]}
    score_row = {"id": data["id"], "score": ctx["score"].tolist()}
    return (pred_row, score_row)
