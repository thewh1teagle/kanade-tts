"""Metric logging helpers for training."""

from __future__ import annotations


def log_train_metrics(loss, enc_lr, lr, writer, step):
    print(f"[step {step}] loss={loss:.4f} lr_enc={enc_lr:.2e} lr={lr:.2e}")
    writer.add_scalar("train/loss", loss, step)
    writer.add_scalar("train/lr_encoder", enc_lr, step)
    writer.add_scalar("train/lr", lr, step)


def log_eval_metrics(metrics, writer, step, label):
    print(f"[{label}] val_loss={metrics['val_loss']:.4f} tok_acc={metrics['token_acc']:.1%}")
    writer.add_scalar("eval/loss", metrics["val_loss"], step)
    writer.add_scalar("eval/token_acc", metrics["token_acc"], step)
