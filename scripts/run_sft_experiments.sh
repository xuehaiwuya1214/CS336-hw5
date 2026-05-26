#!/usr/bin/env bash
set -euo pipefail

# 4.3 SFT 实验批处理脚本。
#
# 用法示例：
#   bash scripts/run_sft_experiments.sh smoke
#   bash scripts/run_sft_experiments.sh unfiltered_128
#   bash scripts/run_sft_experiments.sh unfiltered_full
#   bash scripts/run_sft_experiments.sh filtered_full
#
# 服务器上可通过环境变量覆盖默认路径：
#   MODEL_PATH=/path/to/model DATA_DIR=/data/math bash scripts/run_sft_experiments.sh smoke

EXPERIMENT="${1:-smoke}"

MODEL_PATH="${MODEL_PATH:-/data/a5-alignment/models/Qwen2.5-Math-1.5B}"
DATA_DIR="${DATA_DIR:-/data/math}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/sft}"

UNFILTERED_PATH="${UNFILTERED_PATH:-${DATA_DIR}/sft_gpt-oss-120b.jsonl}"
FILTERED_PATH="${FILTERED_PATH:-${DATA_DIR}/sft_gpt-oss-120b_filtered.jsonl}"
VAL_PATH="${VAL_PATH:-${DATA_DIR}/val.json}"

COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --val-path "${VAL_PATH}"
  --train-device "${TRAIN_DEVICE:-cuda:0}"
  --eval-device "${EVAL_DEVICE:-cuda:1}"
  --eval-backend "${EVAL_BACKEND:-vllm}"
  --vllm-sync-mode "${VLLM_SYNC_MODE:-checkpoint}"
  --per-device-train-batch-size "${BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${GRAD_ACCUM:-8}"
  --learning-rate "${LR:-1e-5}"
  --max-seq-len "${MAX_SEQ_LEN:-2048}"
  --normalize-constant "${NORMALIZE_CONSTANT:-2048}"
  --eval-every "${EVAL_EVERY:-100}"
  --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS:-1024}"
  --save-every "${SAVE_EVERY:-500}"
  --gradient-checkpointing
)

run_unfiltered_size() {
  local size="$1"
  python scripts/train_sft.py \
    "${COMMON_ARGS[@]}" \
    --train-path "${UNFILTERED_PATH}" \
    --output-dir "${OUTPUT_ROOT}/unfiltered_${size}" \
    --run-name "unfiltered_${size}" \
    --num-train-examples "${size}"
}

case "${EXPERIMENT}" in
  smoke)
    python scripts/train_sft.py \
      --model-path "${MODEL_PATH}" \
      --train-path "${UNFILTERED_PATH}" \
      --val-path "${VAL_PATH}" \
      --output-dir "${OUTPUT_ROOT}/smoke" \
      --run-name "smoke" \
      --train-device "${TRAIN_DEVICE:-cuda:0}" \
      --eval-backend "${SMOKE_EVAL_BACKEND:-transformers}" \
      --num-train-examples 16 \
      --max-steps 5 \
      --eval-every 5 \
      --eval-limit 16 \
      --per-device-train-batch-size 1 \
      --gradient-accumulation-steps 2 \
      --max-seq-len 1024 \
      --normalize-constant 1024 \
      --gradient-checkpointing
    ;;
  unfiltered_128)
    run_unfiltered_size 128
    ;;
  unfiltered_256)
    run_unfiltered_size 256
    ;;
  unfiltered_512)
    run_unfiltered_size 512
    ;;
  unfiltered_1024)
    run_unfiltered_size 1024
    ;;
  unfiltered_full)
    python scripts/train_sft.py \
      "${COMMON_ARGS[@]}" \
      --train-path "${UNFILTERED_PATH}" \
      --output-dir "${OUTPUT_ROOT}/unfiltered_full" \
      --run-name "unfiltered_full" \
      --num-train-examples -1
    ;;
  filtered_full)
    python scripts/train_sft.py \
      "${COMMON_ARGS[@]}" \
      --train-path "${FILTERED_PATH}" \
      --output-dir "${OUTPUT_ROOT}/filtered_full" \
      --run-name "filtered_full" \
      --num-train-examples -1
    ;;
  *)
    echo "Unknown experiment: ${EXPERIMENT}" >&2
    echo "Valid: smoke, unfiltered_128, unfiltered_256, unfiltered_512, unfiltered_1024, unfiltered_full, filtered_full" >&2
    exit 1
    ;;
esac
