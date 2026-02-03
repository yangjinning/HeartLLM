#!/usr/bin/env bash
set -e

source env/ft_report_ptbxl.env

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes 2 --main_process_port 29501  run_main.py \
  --batch_size "$BATCH_SIZE" \
  --stage "$STAGE" \
  --dataset "$DATASET" \
  --tasktype "$TASKTYPE" \
  --root_path_ecg "$ROOT_PATH_ECG" \
  --root_path_json "$ROOT_PATH_JSON" \
  --root_report_json "$ROOT_REPORT_JSON" \
  --tokenizer_path "$TOKENIZER_PATH" \
  --pretrain_path "$PRETRAIN_PATH" \
  --local_llm_path "$LOCAL_LLM_PATH" \