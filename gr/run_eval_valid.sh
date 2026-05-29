#!/bin/bash
set -x

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export GR_EVAL_TEST_FILE="${GR_EVAL_TEST_FILE:-gr_valid_split.jsonl}"
export GR_EVAL_CANDIDATE_FILE="${GR_EVAL_CANDIDATE_FILE:-candidate_items.txt}"
export GR_EVAL_OUTPUT_PATH="${GR_EVAL_OUTPUT_PATH:-${PROJECT_ROOT}/outputs/gr/eval_valid_result.json}"

bash "$(dirname "$0")/run_eval_fast.sh"
