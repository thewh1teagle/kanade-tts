"""Optimizer and LR schedule for TTS training."""

from __future__ import annotations

import torch
from transformers import get_cosine_schedule_with_warmup


def build_optimizer(model, encoder_lr: float, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW with discriminative LRs: lower for encoder, higher for the rest."""
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"}

    def is_no_decay(name: str) -> bool:
        return any(term in name for term in no_decay)

    encoder_params_decay = [p for n, p in model.encoder.named_parameters() if not is_no_decay(n)]
    encoder_params_nodecay = [p for n, p in model.encoder.named_parameters() if is_no_decay(n)]
    rest_params_decay = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and not is_no_decay(n)]
    rest_params_nodecay = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and is_no_decay(n)]

    return torch.optim.AdamW([
        {"params": encoder_params_decay, "lr": encoder_lr, "weight_decay": weight_decay},
        {"params": encoder_params_nodecay, "lr": encoder_lr, "weight_decay": 0.0},
        {"params": rest_params_decay, "lr": lr, "weight_decay": weight_decay},
        {"params": rest_params_nodecay, "lr": lr, "weight_decay": 0.0},
    ])


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
