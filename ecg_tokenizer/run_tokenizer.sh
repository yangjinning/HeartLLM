#!/usr/bin/env bash
set -e

source config.env

python tokenizer.py \
  --root_ecg "$ROOT_ECG" \
  --train_path "$TRAIN_PATH" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --log_root "$LOG_ROOT" \
  --val_path "$VAL_PATH" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --val_every "$VAL_EVERY" \
  --patience "$PATIENCE"
