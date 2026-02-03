#!/usr/bin/env bash
set -e

source env/test_qa_ptbxl.env

python run_eval.py \
  --stage "$STAGE" \
  --dataset "$DATASET" \
  --tasktype "$TASKTYPE" \
  --ckp_path "$CKP_PATH" \
  --batch_size "$BATCH_SIZE" \
  --pretrain_path "$PRETRAIN_PATH" \
  --root_path_ecg "$ROOT_PATH_ECG" \
  --root_path_json "$ROOT_PATH_JSON" \
  --root_report_json "$ROOT_REPORT_JSON" \
  --local_llm_path "$LOCAL_LLM_PATH" \
  --tokenizer_path "$TOKENIZER_PATH" \