"""Trainers module for Generation for NLP."""

from .train import (
    CHAT_TEMPLATE,
    compute_metrics,
    get_peft_config,
    main,
    preprocess_logits_for_metrics,
)

__all__ = [
    "CHAT_TEMPLATE",
    "compute_metrics",
    "get_peft_config",
    "main",
    "preprocess_logits_for_metrics",
]
