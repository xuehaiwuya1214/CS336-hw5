#!/usr/bin/env bash
set -euo pipefail

# 第 5 节 Expert Iteration 实验入口。
#
# 用法：
#   bash scripts/run_expert_iteration_experiments.sh smoke
#   bash scripts/run_expert_iteration_experiments.sh g4_e1_b512
#   bash scripts/run_expert_iteration_experiments.sh g8_e1_b512
#   bash scripts/run_expert_iteration_experiments.sh g4_e2_b512
#   bash scripts/run_expert_iteration_experiments.sh g4_e1_b1024
#
# 可通过环境变量覆盖路径和资源设置：
#   MODEL_PATH=/path/to/model DATA_DIR=/data/math bash scripts/run_expert_iteration_experiments.sh smoke

EXPERIMENT="${1:-smoke}"

MODEL_PATH="${MODEL_PATH:-/data/a5-alignment/models/Qwen2.5-Math-1.5B}"
DATA_DIR="${DATA_DIR:-/data/math}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/expert_iteration}"
TRAIN_PATH="${TRAIN_PATH:-${DATA_DIR}/train.jsonl}"
VAL_PATH="${VAL_PATH:-${DATA_DIR}/val.jsonl}"

COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --train-path "${TRAIN_PATH}"
  --val-path "${VAL_PATH}"
  --train-device "${TRAIN_DEVICE:-cuda:0}"
  --eval-device "${EVAL_DEVICE:-cuda:1}"
  --rollout-backend "${ROLLOUT_BACKEND:-vllm}"
  --eval-backend "${EVAL_BACKEND:-vllm}"
  --sft-batch-size "${SFT_BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${GRAD_ACCUM:-8}"
  --learning-rate "${LR:-1e-5}"
  --max-seq-len "${MAX_SEQ_LEN:-2048}"
  --normalize-constant "${NORMALIZE_CONSTANT:-2048}"
  --rollout-max-tokens "${ROLLOUT_MAX_TOKENS:-1024}"
  --rollout-min-tokens "${ROLLOUT_MIN_TOKENS:-4}"
  --eval-max-tokens "${EVAL_MAX_TOKENS:-1024}"
  --gradient-checkpointing
)

run_config() {
  local name="$1"
  local g="$2"
  local epochs="$3"
  local batch_size="$4"
  python scripts/train_expert_iteration.py \
    "${COMMON_ARGS[@]}" \
    --output-dir "${OUTPUT_ROOT}/${name}" \
    --run-name "${name}" \
    --rollouts-per-problem "${g}" \
    --sft-epochs-per-ei-step "${epochs}" \
    --ei-batch-size "${batch_size}" \
    --num-ei-steps "${NUM_EI_STEPS:-5}"
}

case "${EXPERIMENT}" in
  smoke)
    python scripts/train_expert_iteration.py \
      --model-path "${MODEL_PATH}" \
      --train-path "${TRAIN_PATH}" \
      --val-path "${VAL_PATH}" \
      --output-dir "${OUTPUT_ROOT}/smoke" \
      --run-name "ei_smoke" \
      --train-device "${TRAIN_DEVICE:-cuda:0}" \
      --rollout-backend "${SMOKE_BACKEND:-transformers}" \
      --eval-backend "${SMOKE_EVAL_BACKEND:-transformers}" \
      --num-ei-steps 1 \
      --ei-batch-size 8 \
      --rollouts-per-problem 2 \
      --sft-epochs-per-ei-step 1 \
      --sft-batch-size 1 \
      --gradient-accumulation-steps 2 \
      --eval-limit 16 \
      --rollout-max-tokens 256 \
      --eval-max-tokens 256 \
      --max-seq-len 1024 \
      --normalize-constant 1024 \
      --gradient-checkpointing
    ;;
  g4_e1_b512)
    run_config "g4_e1_b512" 4 1 512
    ;;
  g8_e1_b512)
    run_config "g8_e1_b512" 8 1 512
    ;;
  g4_e2_b512)
    run_config "g4_e2_b512" 4 2 512
    ;;
  g4_e1_b1024)
    run_config "g4_e1_b1024" 4 1 1024
    ;;
  g4_e1_b2048)
    run_config "g4_e1_b2048" 4 1 2048
    ;;
  *)
    echo "Unknown experiment: ${EXPERIMENT}" >&2
    echo "Valid: smoke, g4_e1_b512, g8_e1_b512, g4_e2_b512, g4_e1_b1024, g4_e1_b2048" >&2
    exit 1
    ;;
esac
