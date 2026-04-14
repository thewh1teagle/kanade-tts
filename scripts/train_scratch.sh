set -e

TRAIN_DIR="outputs/kanade-ar"

echo "=== Training autoregressive Kanade TTS ==="
uv run accelerate launch src/train.py \
    --dataset dataset/dataset.jsonl \
    --output-dir "$TRAIN_DIR" \
    --epochs 100 \
    --train-batch-size 16 \
    --eval-batch-size 16 \
    --encoder-lr 1e-4 \
    --lr 3e-4 \
    --warmup-steps 1000 \
    --logging-steps 50 \
    --save-steps 1000 \
    --save-total-limit 5 \
    --dataloader-workers 4
