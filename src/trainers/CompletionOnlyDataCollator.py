
import torch


class CompletionOnlyDataCollator:
    def __init__(self, tokenizer, response_template: str | None = None, label_pad_token_id: int = -100):
        self.tokenizer = tokenizer

        self.response_template_ids = tokenizer.encode(response_template, add_special_tokens=False) if response_template else []
        self.label_pad_token_id = label_pad_token_id

    def _find_response_start(self, ids: list[int]) -> int:
        seq = self.response_template_ids
        if not seq:
            return 0
        L = len(seq)
        for i in range(max(0, len(ids) - L + 1)):
            if ids[i : i + L] == seq:
                return i + L
        return 0

    def __call__(self, features):
        # features: list of dicts with 'input_ids' (list[int]) or tensors
        input_seqs = []
        for f in features:
            ids = f["input_ids"]
            if not isinstance(ids, list):
                ids = ids.tolist()
            input_seqs.append(ids)

        batch = self.tokenizer.pad({"input_ids": input_seqs}, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        labels = torch.full(batch["input_ids"].shape, self.label_pad_token_id, dtype=torch.long)

        for i, ids in enumerate(input_seqs):
            start = self._find_response_start(ids)
            # clip/truncate if needed
            truncated = ids[:max_len]
            if start < len(truncated):
                labels[i, : len(truncated)] = torch.tensor(
                    [ (tok if idx >= start else self.label_pad_token_id) for idx, tok in enumerate(truncated) ],
                    dtype=torch.long,
                )

        batch["labels"] = labels
        return batch
