"""Trainers module for Generation for NLP."""

from trainers.train import (
    compute_metrics,
    get_peft_config,
    main,
    preprocess_logits_for_metrics,
)
from trainers.CompletionOnlyDataCollator import (CompletionOnlyDataCollator)

__all__ = [
    "compute_metrics",
    "get_peft_config",
    "main",
    "preprocess_logits_for_metrics",
    "CompletionOnlyDataCollator"
]
