#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-checkpoints/sft_200m_120k/struct_sft_200m_120k.pt}"
TEST_PATH="data/sft/final/sft_test_200m.jsonl"

echo "Model: ${MODEL_PATH}"
echo "Test data: ${TEST_PATH}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Missing model checkpoint: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -f "${TEST_PATH}" ]]; then
  echo "Missing test data: ${TEST_PATH}" >&2
  exit 1
fi

echo "TODO: plug in a dedicated SFT evaluator for JSON Parse Rate, Markdown Close Rate, and Trap Defense Rate."
echo "Basic generation sanity check:"

python decode/generate_pretrain.py \
  --model "${MODEL_PATH}" \
  --prompt "请输出一个 JSON 对象，包含姓名、年龄和技能列表。" \
  --max-new-tokens 160 \
  --temperature 0.6 \
  --top-k 40 \
  --repetition-penalty 1.12 \
  --no-repeat-ngram-size 4 \
  --device cuda
