# Struct-LM 50M Pretraining

This stage trains Struct-LM from scratch with a pretrained Chinese tokenizer.
It does not load pretrained model weights.

## Dataset Sampling

Run all three sampling stages:

```powershell
python data_gen\sample_skypile_stages.py
```

Run only one stage:

```powershell
python data_gen\sample_skypile_stages.py --only 100mb
```

The script skips existing files by default. Use `--force` to resample.

Manual commands are still available. Start small:

```powershell
python data_gen\sample_skypile.py --target-bytes 100MB --out-prefix data/pretrain/skypile_100mb
```

Then scale:

```powershell
python data_gen\sample_skypile.py --target-bytes 1GB --out-prefix data/pretrain/skypile_1gb
python data_gen\sample_skypile.py --target-bytes 5GB --out-prefix data/pretrain/skypile_5gb
```

For 1GB or 5GB, update `train_path` and `valid_path` in
`configs/struct_pretrain_50m.yaml`.

## Pretraining

Smoke test:

```powershell
python train\train_struct.py --config configs\struct_pretrain_50m.yaml --max-steps 10 --batch-size 1 --block-size 128 --eval-iters 1 --device cpu
```

Real run:

```powershell
python train\train_struct.py --config configs\struct_pretrain_50m.yaml
```

Pretraining gives continuation ability and Chinese language statistics. Basic
chat behavior usually requires a later SFT stage with instruction/dialogue data.
