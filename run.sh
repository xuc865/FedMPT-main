#!/usr/bin/env bash
# Federated multi-label experiment launcher.
#
# Usage:
#   bash run.sh [dataset] [lr] [model] [epochs] [gpu] [num_clusters]
#
# Examples:
#   bash run.sh voc 0.001 fedmpt 50 0 2
#   bash run.sh coco 0.001 dualcoop 100 0 8
#
# Environment:
#   DATA_ROOT   dataset root (default: ./data)
#   OUTPUT_DIR  logs & checkpoints (default: ./outputs)

set -euo pipefail

DATASET="${1:-voc}"
LR="${2:-0.001}"
MODEL="${3:-fedmpt}"
EPOCH="${4:-50}"
GPU="${5:-0}"
CLUSTERS="${6:-2}"

DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"

export CUDA_VISIBLE_DEVICES="${GPU}"

python Launch_FL.py \
  --root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --exp_name cross_cls \
  --model_name "${MODEL}" \
  --dataset "${DATASET}" \
  --num_cls_per_client 1 \
  --num_clusters "${CLUSTERS}" \
  --num_epoch "${EPOCH}" \
  --avail_percent 1 \
  --lr "${LR}" \
  --cond 5 \
  --cls 4 \
  --temp 4
