#!/bin/bash
set -x

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export USER_CACHE_PATH="${USER_CACHE_PATH:-${PROJECT_ROOT}/outputs}"
export TRAIN_CKPT_PATH="${TRAIN_CKPT_PATH:-${PROJECT_ROOT}/outputs/gr/checkpoints}"

cd "${PROJECT_ROOT}"

EVAL_ARGS=(
  --config gr/gr_train.json
  --generate_num "${GR_EVAL_GENERATE_NUM:-100}"
  --candidate_top_k "${GR_EVAL_CANDIDATE_TOP_K:-20}"
  --metric_k "${GR_EVAL_METRIC_K:-20}"
  --output_path "${GR_EVAL_OUTPUT_PATH:-${PROJECT_ROOT}/outputs/gr/eval_result.json}"
)

if [[ "${GR_EVAL_NPROC_PER_NODE:-1}" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node "${GR_EVAL_NPROC_PER_NODE}" gr/eval_gr.py "${EVAL_ARGS[@]}"
else
  python gr/eval_gr.py "${EVAL_ARGS[@]}"
fi
