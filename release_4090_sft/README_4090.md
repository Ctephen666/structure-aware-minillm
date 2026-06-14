# 4090 SFT Release

This release contains the code, checkpoint, and final SFT dataset needed to continue SFT training for a 200M Chinese structure-aware decoder-only LLM.

## Project

- Model: 200M Chinese structure-aware LLM
- Current stage: SFT on 120k formatted/instruction samples
- Split: train 114000, val 3600, test 2400
- Base checkpoint: `checkpoints/struct_pretrain_200m_zh_best.pt`
- SFT config: `configs/struct_sft_200m_4090.yaml`

The model uses token, position, depth, and state embeddings, plus LM/depth/state heads. The SFT dataset uses `instruction/input/output` rows, and the packaged dataset loader supports that schema.

## Recommended 4090 Environment

- Ubuntu 22.04
- Python 3.10
- CUDA 12.x
- PyTorch GPU build matching your CUDA runtime
- NVIDIA RTX 4090 24GB

Check the GPU environment:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

Final SFT files:

```text
data/sft/final/sft_train_200m.jsonl
data/sft/final/sft_val_200m.jsonl
data/sft/final/sft_test_200m.jsonl
data/sft/final/sft_dataset_stats.json
```

## Smoke Test

Run a small SFT smoke test first. It takes the first 2000 train rows, verifies checkpoint/tokenizer/data loading, runs a few steps, and saves into `checkpoints/sft_200m_smoke/`.

```bash
bash run_smoke_sft.sh
```

## Full SFT

Recommended hyperparameters:

```text
seq_len: 1024
precision: bf16
micro_batch_size: 16
gradient_accumulation_steps: 4
learning_rate: 5e-5
epochs: 1
warmup_ratio: 0.03
weight_decay: 0.01
grad_clip: 1.0
save_every: 1000
eval_every: 500
```

Start full SFT:

```bash
bash run_full_sft_4090.sh
```

The packaged training entrypoint is:

```bash
python train/train_struct.py --config configs/struct_sft_200m_4090.yaml
```

## OOM Handling

If 4090 24GB runs out of memory:

- Reduce `batch_size` from 16 to 8.
- Increase `gradient_accumulation_steps` from 4 to 8.
- Reduce `block_size` from 1024 to 512.
- Keep `gradient_checkpointing: true`.

Note: the base pretraining checkpoint was trained with `block_size=512`. The packaged loader can initialize a `block_size=1024` model by copying the old position embeddings into the first 512 positions and leaving the extra positions randomly initialized.

## Evaluation

After training, use:

```bash
bash run_eval_sft.sh
```

This project does not currently include a dedicated SFT metric evaluator for JSON Parse Rate, Markdown Close Rate, and Trap Defense Rate. `run_eval_sft.sh` contains a basic command framework and TODO markers for plugging in such an evaluator.

## Security

Do not put API keys in this release. If a script needs a key in the future, use environment variables such as:

```bash
export DEEPSEEK_API_KEY="PLEASE_SET_YOUR_API_KEY"
```

Never commit or upload real `sk-...` keys.

During packaging, release text files were scanned for common API key patterns. Any detected key-like value was replaced with `PLEASE_SET_YOUR_API_KEY`.
