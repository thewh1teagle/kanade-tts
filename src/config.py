"""CLI argument parsing for autoregressive Kanade TTS training."""

from __future__ import annotations

import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Train the autoregressive Kanade TTS model")

    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset.jsonl")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--val-split", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--epochs", type=float, default=100.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dataloader-workers", type=int, default=4)

    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)

    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=5)
    parser.add_argument("--save-epochs", action="store_true", default=False)

    parser.add_argument("--hidden-size", type=int, default=320)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=8)
    parser.add_argument("--num-attention-heads", type=int, default=5)
    parser.add_argument("--ffn-dim", type=int, default=1280)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-text-positions", type=int, default=512)
    parser.add_argument("--max-audio-positions", type=int, default=512)

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--reset-steps", action="store_true", default=False)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
    )

    return parser.parse_args()
