#!/bin/bash
set -x

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

export USER_CACHE_PATH="${USER_CACHE_PATH:-${PROJECT_ROOT}/outputs}"
export TRAIN_CKPT_PATH="${TRAIN_CKPT_PATH:-${PROJECT_ROOT}/outputs/gr/checkpoints}"
mkdir -p "${TRAIN_CKPT_PATH}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export DS_TRANSPORT_TCP="${DS_TRANSPORT_TCP:-1}"
export FI_SHM_DISABLE="${FI_SHM_DISABLE:-1}"
export PYTORCH_DISTRIBUTED_BACKEND="${PYTORCH_DISTRIBUTED_BACKEND:-gloo}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export GLOO_DEBUG="${GLOO_DEBUG:-1}"

cd "${PROJECT_ROOT}"
python gr/train_gr.py --config gr/gr_train.json
