"""Checkpoint saving and resuming for training."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from safetensors.torch import load_file


def save_checkpoint(model, output_dir: Path, step: int, val_loss: float, save_total_limit: int) -> None:
    ckpt_dir = output_dir / f"step-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir), safe_serialization=True)
    (ckpt_dir / "train_state.json").write_text(json.dumps({"step": step, "val_loss": val_loss}))

    checkpoints = sorted(output_dir.glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    while len(checkpoints) > save_total_limit:
        shutil.rmtree(checkpoints.pop(0))


def save_epoch_checkpoint(model, output_dir: Path, epoch: int, step: int, val_loss: float) -> None:
    ckpt_dir = output_dir / f"epoch-{epoch}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir), safe_serialization=True)
    (ckpt_dir / "train_state.json").write_text(json.dumps({"step": step, "epoch": epoch, "val_loss": val_loss}))


def resume_step(checkpoint: str, scheduler) -> int:
    state_path = Path(checkpoint) / "train_state.json"
    if not state_path.exists():
        return 0
    step = json.loads(state_path.read_text())["step"]
    for _ in range(step):
        scheduler.step()
    return step


def load_weights(model, checkpoint: str) -> None:
    state = load_file(str(Path(checkpoint) / "model.safetensors"), device="cpu")
    model.load_state_dict(state)
