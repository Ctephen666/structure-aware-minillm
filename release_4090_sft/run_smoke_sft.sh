#!/usr/bin/env bash
set -euo pipefail

mkdir -p checkpoints/sft_200m_smoke

python - <<'PY'
from pathlib import Path

src = Path("data/sft/final/sft_train_200m.jsonl")
dst = Path("data/sft/final/sft_train_smoke.jsonl")
limit = 2000

with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8", newline="\n") as fout:
    for idx, line in enumerate(fin):
        if idx >= limit:
            break
        fout.write(line)
print(f"wrote {dst} with up to {limit} rows")
PY

python train/train_struct.py \
  --config configs/struct_sft_200m_4090.yaml \
  --train-path data/sft/final/sft_train_smoke.jsonl \
  --valid-path data/sft/final/sft_val_200m.jsonl \
  --max-steps 20 \
  --batch-size 2 \
  --eval-iters 5 \
  --checkpoint-path checkpoints/sft_200m_smoke/struct_sft_200m_smoke.pt \
  --device cuda 2>&1 | tee checkpoints/sft_200m_smoke/train.log
