#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RQVAE_DATA_PATH="${RQVAE_DATA_PATH:-${PROJECT_ROOT}/outputs/sasrec/cache/emb/emb/emb}"
export RQVAE_OUTPUT_PATH="${RQVAE_OUTPUT_PATH:-${PROJECT_ROOT}/outputs/rqvae}"
export RQVAE_CKPT_PATH="${RQVAE_CKPT_PATH:-${RQVAE_OUTPUT_PATH}/ckpt}"
export RQVAE_TF_EVENTS_PATH="${RQVAE_TF_EVENTS_PATH:-${RQVAE_OUTPUT_PATH}/tf_events}"
export USER_CACHE_PATH="${USER_CACHE_PATH:-${RQVAE_OUTPUT_PATH}/cache}"

mkdir -p "$RQVAE_CKPT_PATH" "$RQVAE_TF_EVENTS_PATH" "$USER_CACHE_PATH"

cd "${PROJECT_ROOT}/rqvae/train"

python -u rqvae_train.py \
  --data_path "$RQVAE_DATA_PATH" \
  --ckpt_dir "$RQVAE_CKPT_PATH" \
  --device "${RQVAE_DEVICE:-cuda:0}" \
  --batch_size "${RQVAE_BATCH_SIZE:-8192}" \
  --epochs "${RQVAE_EPOCHS:-1000}" \
  --eval_step "${RQVAE_EVAL_STEP:-20}" \
  --lr "${RQVAE_LR:-3e-4}" \
  --num_workers "${RQVAE_NUM_WORKERS:-4}" \
  --num_emb_list ${RQVAE_NUM_EMB_LIST:-2048 2048 1024} \
  --e_dim "${RQVAE_E_DIM:-32}" \
  --layers ${RQVAE_LAYERS:-256 128 64} \
  --sk_epsilons ${RQVAE_SK_EPSILONS:-0.0 0.0 0.0} \
  --kmeans_init "${RQVAE_KMEANS_INIT:-true}"
