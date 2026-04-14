set -e

TRAIN_DIR="outputs/kanade-ar"
RESUME="${1:-outputs/kanade-ar/step-86400}"

echo "=== Resuming autoregressive Kanade TTS training ==="
uv run accelerate launch src/train.py \
    --dataset dataset/dataset.jsonl \
    --output-dir "$TRAIN_DIR" \
    --epochs 300 \
    --train-batch-size 16 \
    --eval-batch-size 16 \
    --encoder-lr 1e-4 \
    --lr 3e-4 \
    --warmup-steps 1000 \
    --logging-steps 50 \
    --save-steps 1000 \
    --save-total-limit 5 \
    --dataloader-workers 4 \
    --dropout 0.4 \
    --weight-decay 0.1 \
    --resume "$RESUME" \
    --reset-steps
