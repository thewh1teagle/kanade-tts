"""Train the autoregressive Kanade TTS model.

Example:
    uv run src/train.py \
        --dataset dataset/dataset.jsonl \
        --output-dir outputs/kanade-ar
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from checkpoint import load_weights, resume_step, save_checkpoint, save_epoch_checkpoint
from config import parse_args
from data import make_dataloaders
from eval import evaluate
from metrics import log_eval_metrics, log_train_metrics
from model import build_model
from optimizer import build_optimizer, build_scheduler
from tokenization import load_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision="fp16" if args.fp16 else "no")
    device = accelerator.device

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard")) if accelerator.is_main_process else None

    tokenizer = load_tokenizer()
    train_loader, val_loader = make_dataloaders(args)
    model = build_model(args, tokenizer)

    if args.resume:
        load_weights(model, args.resume)
        if accelerator.is_main_process:
            print(f"Loaded weights from {args.resume}")

    total_opt_steps = math.ceil(len(train_loader) * args.epochs / args.gradient_accumulation_steps)
    optimizer = build_optimizer(model, args.encoder_lr, args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.warmup_steps, total_opt_steps)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    opt_step = 0
    if args.resume and not args.reset_steps:
        opt_step = resume_step(args.resume, scheduler)
        if accelerator.is_main_process:
            print(f"Resumed from step {opt_step}")

    global_step = opt_step * args.gradient_accumulation_steps
    optimizer.zero_grad()

    for epoch in range(math.ceil(args.epochs)):
        epoch_loss = epoch_steps = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}", dynamic_ncols=True, disable=not accelerator.is_main_process)

        for batch in pbar:
            if opt_step >= total_opt_steps:
                break

            with accelerator.autocast():
                out = model(**batch)

            scaled_loss = out.loss / args.gradient_accumulation_steps
            accelerator.backward(scaled_loss)
            epoch_loss += out.loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % args.gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                opt_step += 1

                avg_loss = epoch_loss / epoch_steps
                pbar.set_postfix(step=opt_step, loss=f"{avg_loss:.4f}")

                if accelerator.is_main_process and opt_step % args.logging_steps == 0:
                    log_train_metrics(avg_loss, optimizer.param_groups[0]["lr"], optimizer.param_groups[2]["lr"], writer, opt_step)

                if accelerator.is_main_process and opt_step % args.save_steps == 0:
                    metrics = evaluate(accelerator.unwrap_model(model), val_loader, device, args.fp16)
                    log_eval_metrics(metrics, writer, opt_step, f"step {opt_step}")
                    save_checkpoint(accelerator.unwrap_model(model), output_dir, opt_step, metrics["val_loss"], args.save_total_limit)

        if args.save_epochs and accelerator.is_main_process:
            metrics = evaluate(accelerator.unwrap_model(model), val_loader, device, args.fp16)
            log_eval_metrics(metrics, writer, opt_step, f"epoch {epoch + 1}")
            save_epoch_checkpoint(accelerator.unwrap_model(model), output_dir, epoch + 1, opt_step, metrics["val_loss"])

    if accelerator.is_main_process:
        metrics = evaluate(accelerator.unwrap_model(model), val_loader, device, args.fp16)
        log_eval_metrics(metrics, writer, opt_step, "final")
        save_checkpoint(accelerator.unwrap_model(model), output_dir, opt_step, metrics["val_loss"], args.save_total_limit)
        writer.close()


if __name__ == "__main__":
    main()
