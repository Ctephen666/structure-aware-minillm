#!/usr/bin/env bash
set -euo pipefail

mkdir -p checkpoints/sft_struct_200m

python train/train_struct.py \
  --config configs/struct_sft_format_200m_4090.yaml \
  --device cuda 2>&1 | tee checkpoints/sft_struct_200m/train.log
