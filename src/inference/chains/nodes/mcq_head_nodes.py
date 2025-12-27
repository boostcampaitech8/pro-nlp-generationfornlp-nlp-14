import torch
from inference_utils import load_model
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI

from utils import get_choice_token_ids, logits_to_prediction
from utils.constants import CHOICE_TOKENS


def create_local_forward(config):
    model, tokenizer = load_model(config.checkpoint_path, config.max_seq_length)
    device = next(model.parameters()).device

    @chain
    @torch.inference_mode()
    def forward(data: dict):
        messages = data["messages"]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        outputs = model(input_ids)
        len_choices = data["len_choices"]
        log_probs = torch.log_softmax(outputs.logits[:, -1].flatten(), dim=0)
        choice_ids = get_choice_token_ids(tokenizer, len_choices)
        target = log_probs[choice_ids].cpu()  # LogS(V)filter choice_ids
        return {"data": data, "score": target}

    return forward


def create_llamacpp_forward():
    import os

    import dotenv

    dotenv.load_dotenv()
    LLAMA_CPP_SERVER_URL = os.getenv("LLAMA_CPP_SERVER_URL")

    model = ChatOpenAI(
        base_url=LLAMA_CPP_SERVER_URL,
        api_key="NOT_NEED",
        model_name="NOTE_NEED",
        temperature=0,
        extra_body={
            "max_tokens": 1,
            "grammar": 'root ::= ("1" | "2" | "3" | "4" | "5")',
            "n_probs": 50,
            "min_keep": 5,
        },
    )

    @chain
    def foward(data):
        output = model.invoke(data["messages"])
        len_choices = data["len_choices"]
        top = output.response_metadata["logprobs"]["content"][0]["top_logprobs"]
        top_dict = {tok["token"]: tok["logprob"] for tok in top if tok["token"] in CHOICE_TOKENS}
        target = [top_dict.get(str(i), -torch.inf) for i in range(1, len_choices + 1)]
        target_tensor = torch.tensor(target, dtype=torch.float)

        return {"data": data, "score": target_tensor}

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
