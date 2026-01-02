from langchain_core.runnables import chain

from utils.choice_utils import get_choice_token_ids


def build_local_forward(config):
    # Lazy import: use_remote=False일 때만 필요
    import torch

    from inference.inference_utils import load_model

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
        target = log_probs[choice_ids].cpu().tolist()  # LogS(V)filter choice_ids
        return {"data": data, "score": target}

    return forward
