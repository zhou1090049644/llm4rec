#!/usr/bin/env bash
set -e

export TRAIN_DATA_PATH=/data1/zhouchen109/LLM4Rec/data
export TRAIN_LOG_PATH=/data1/zhouchen109/LLM4Rec/sasrec/logs
export TRAIN_TF_EVENTS_PATH=/data1/zhouchen109/LLM4Rec/sasrec/tf_events
export TRAIN_CKPT_PATH=/data1/zhouchen109/LLM4Rec/sasrec/checkpoints
export USER_CACHE_PATH=/data1/zhouchen109/LLM4Rec/sasrec/cache

mkdir -p "$TRAIN_LOG_PATH" "$TRAIN_TF_EVENTS_PATH" "$TRAIN_CKPT_PATH" "$USER_CACHE_PATH"

cd sasrec

python main.py --mm_emb_id 81 --batch_size 16 --num_epochs 1 --device cuda
