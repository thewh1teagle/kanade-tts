"""Evaluation helpers for autoregressive Kanade TTS training."""

from __future__ import annotations

import torch

from data import IGNORE_INDEX


def compute_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    mask = labels != IGNORE_INDEX
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    return (preds[mask] == labels[mask]).float().mean().item()


def evaluate(model, val_loader, device, fp16: bool) -> dict:
    model.eval()
    total_loss = tok_acc_sum = 0.0
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", enabled=fp16):
                out = model(**batch)

            total_loss += out.loss.item()
            tok_acc_sum += compute_token_accuracy(out.logits, batch["labels"])
            n += 1

    model.train()
    return {
        "val_loss": total_loss / n,
        "token_acc": tok_acc_sum / n,
    }
