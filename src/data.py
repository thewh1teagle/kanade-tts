"""Dataset loading and collation for autoregressive Kanade TTS training."""

from __future__ import annotations

import json
import random

import torch
from torch.utils.data import DataLoader, Dataset

from model import AUDIO_EOS_TOKEN_ID


IGNORE_INDEX = -100


class TTSDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "ipa_tokens": r["ipa_tokens"],
            "audio_tokens": r["audio_tokens"],
        }


class TTSCollator:
    text_pad_id: int = 0

    def __call__(self, features: list[dict]) -> dict:
        max_text = max(len(f["ipa_tokens"]) for f in features)
        max_audio = max(len(f["audio_tokens"]) for f in features) + 1

        input_ids, attention_mask = [], []
        labels = []

        for f in features:
            text = f["ipa_tokens"]
            audio = f["audio_tokens"]

            text_pad = max_text - len(text)
            audio_pad = max_audio - (len(audio) + 1)

            input_ids.append(text + [self.text_pad_id] * text_pad)
            attention_mask.append([1] * len(text) + [0] * text_pad)

            label = audio + [AUDIO_EOS_TOKEN_ID]

            labels.append(label + [IGNORE_INDEX] * audio_pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def make_dataloaders(args) -> tuple[DataLoader, DataLoader]:
    records = load_dataset(args.dataset)
    rng = random.Random(args.seed)
    rng.shuffle(records)

    n_val = max(1, int(len(records) * args.val_split))
    val_records = records[:n_val]
    train_records = records[n_val:]

    collator = TTSCollator()
    train_loader = DataLoader(
        TTSDataset(train_records),
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.dataloader_workers,
    )
    val_loader = DataLoader(
        TTSDataset(val_records),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.dataloader_workers,
    )
    return train_loader, val_loader
