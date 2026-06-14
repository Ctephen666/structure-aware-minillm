#!/usr/bin/env bash
set -euo pipefail

mkdir -p checkpoints/sft_200m_120k

python train/train_struct.py \
  --config configs/struct_sft_200m_4090.yaml \
  --device cuda 2>&1 | tee checkpoints/sft_200m_120k/train.log
