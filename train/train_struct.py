"""Train the structure-aware decoder-only Transformer.

Examples:
    python train/train_struct.py --config configs/struct_pretrain_80m.yaml --device cuda
    python train/train_struct.py --config configs/struct_pretrain_80m.yaml --resume
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.struct_transformer import StructTransformerConfig, StructTransformerModel
from tokenizer.tokenizer_factory import build_tokenizer
from train.dataset import read_text_samples
from train.structure_dataset import (
    StreamingStructureLanguageModelingDataset,
    StructureLanguageModelingDataset,
    WeightedStreamingStructureLanguageModelingDataset,
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device: torch.device):
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


def resolve_amp_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported amp_dtype: {name}")


def resolve_weighted_sources(sources: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    resolved_sources = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"{key}[{index}] must be a mapping with path/weight fields.")
        if "path" not in source:
            raise ValueError(f"{key}[{index}] is missing required field: path")
        resolved = dict(source)
        resolved["path"] = resolve_path(resolved["path"])
        if not resolved["path"].exists():
            raise FileNotFoundError(f"Missing {key}[{index}] file: {resolved['path']}")
        resolved_sources.append(resolved)
    return resolved_sources


def should_stream(cfg: dict[str, Any], key: str, path: Path | None) -> bool:
    if key in cfg:
        return bool(cfg[key])
    if path is None or not path.exists():
        return False
    threshold = int(cfg.get("auto_streaming_min_bytes", 256 * 1024 * 1024))
    return path.stat().st_size >= threshold


@torch.no_grad()
def estimate_loss(
    model: StructTransformerModel,
    loader: DataLoader,
    device: torch.device,
    eval_iters: int,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "lm_loss": 0.0, "depth_loss": 0.0, "state_loss": 0.0}
    count = 0
    for step, batch in enumerate(loader):
        if step >= eval_iters:
            break
        input_ids, labels, depth_ids, state_ids, depth_targets, state_targets = move_batch(batch, device)
        out = model(input_ids, depth_ids, state_ids, labels, depth_targets, state_targets)
        for key in totals:
            value = out[key]
            totals[key] += float(value.item()) if value is not None else 0.0
        count += 1
    model.train()
    return {key: value / max(count, 1) for key, value in totals.items()}


def save_checkpoint(
    path: str | Path,
    model: StructTransformerModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    config: dict[str, Any],
    step: int,
    best_val_loss: float,
) -> None:
    checkpoint_path = resolve_path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    val_loss = None if math.isinf(best_val_loss) else best_val_loss
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "step": step,
            "val_loss": val_loss,
            "best_val_loss": val_loss,
        },
        checkpoint_path,
    )


def load_checkpoint(
    path: str | Path,
    model: StructTransformerModel,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = resolve_path(path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"]
    model_state = model.state_dict()
    patched_state = {}
    skipped = []
    for name, value in state_dict.items():
        if name in model_state and model_state[name].shape != value.shape:
            if name == "position_embedding.weight" and model_state[name].ndim == 2 and value.ndim == 2:
                resized = model_state[name].clone()
                rows = min(resized.shape[0], value.shape[0])
                cols = min(resized.shape[1], value.shape[1])
                resized[:rows, :cols] = value[:rows, :cols]
                patched_state[name] = resized
                print(
                    f"Resized checkpoint tensor {name}: "
                    f"{tuple(value.shape)} -> {tuple(resized.shape)}"
                )
            else:
                skipped.append((name, tuple(value.shape), tuple(model_state[name].shape)))
            continue
        patched_state[name] = value
    if skipped:
        for name, old_shape, new_shape in skipped:
            print(f"Skipped incompatible checkpoint tensor {name}: {old_shape} -> {new_shape}")
    model.load_state_dict(patched_state, strict=False)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def build_model_config(cfg: dict[str, Any], tokenizer) -> StructTransformerConfig:
    tokenizer_vocab_size = int(tokenizer.vocab_size)
    configured_vocab_size = cfg.get("vocab_size")
    if configured_vocab_size is not None and int(configured_vocab_size) != tokenizer_vocab_size:
        print(
            f"Warning: config vocab_size={configured_vocab_size}, "
            f"tokenizer vocab_size={tokenizer_vocab_size}; using tokenizer size."
        )
    return StructTransformerConfig(
        vocab_size=tokenizer_vocab_size,
        block_size=int(cfg["block_size"]),
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        n_embd=int(cfg["n_embd"]),
        dropout=float(cfg["dropout"]),
        max_depth=int(cfg.get("max_depth", 32)),
        num_states=int(cfg.get("num_states", 9)),
        lambda_depth=float(cfg.get("lambda_depth", 0.1)),
        lambda_state=float(cfg.get("lambda_state", 0.2)),
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def learning_rate_for_step(cfg: dict[str, Any], step: int) -> float:
    max_lr = float(cfg["learning_rate"])
    min_lr = float(cfg.get("min_lr", 0.0))
    warmup_steps = int(cfg.get("warmup_steps", 0))
    max_steps = int(cfg["max_steps"])

    if warmup_steps > 0 and step <= warmup_steps:
        return max_lr * step / warmup_steps
    if max_steps <= warmup_steps:
        return max_lr

    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + cosine_coeff * (max_lr - min_lr)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def build_datasets(cfg: dict[str, Any], tokenizer):
    block_size = int(cfg["block_size"])
    streaming_stride = cfg.get("streaming_stride")
    stride = int(streaming_stride) if streaming_stride is not None else None

    train_sources = cfg.get("train_sources")
    valid_sources = cfg.get("valid_sources")
    train_path = resolve_path(cfg["train_path"]) if "train_path" in cfg else None
    valid_path = resolve_path(cfg["valid_path"]) if "valid_path" in cfg else None

    streaming_train = bool(train_sources) or should_stream(cfg, "streaming_train", train_path)
    streaming_valid = bool(valid_sources) or should_stream(cfg, "streaming_valid", valid_path)

    if train_sources is not None:
        train_dataset = WeightedStreamingStructureLanguageModelingDataset(
            resolve_weighted_sources(train_sources, "train_sources"),
            tokenizer,
            block_size,
            stride=stride,
            repeat=True,
        )
    elif streaming_train:
        if train_path is None:
            raise ValueError("streaming_train requires train_path or train_sources.")
        train_dataset = StreamingStructureLanguageModelingDataset(train_path, tokenizer, block_size, stride=stride)
    else:
        if train_path is None:
            raise ValueError("Training requires train_path or train_sources.")
        train_dataset = StructureLanguageModelingDataset(train_path, tokenizer, block_size)

    if valid_sources is not None:
        valid_dataset = WeightedStreamingStructureLanguageModelingDataset(
            resolve_weighted_sources(valid_sources, "valid_sources"),
            tokenizer,
            block_size,
            stride=stride,
            repeat=True,
        )
    else:
        if valid_path is None or not valid_path.exists():
            if train_path is None:
                raise ValueError("Validation requires valid_path, train_path, or valid_sources.")
            valid_source = train_path
        else:
            valid_source = valid_path
        if streaming_valid:
            valid_dataset = StreamingStructureLanguageModelingDataset(valid_source, tokenizer, block_size, stride=stride)
        else:
            valid_dataset = StructureLanguageModelingDataset(valid_source, tokenizer, block_size)

    return train_dataset, valid_dataset, train_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/struct.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--valid-path", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--init-from-checkpoint", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--streaming-train", action="store_true")
    parser.add_argument("--streaming-valid", action="store_true")
    parser.add_argument("--streaming-stride", type=int, default=None)
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(cfg)
    for arg_name, cfg_name in [
        ("max_steps", "max_steps"),
        ("batch_size", "batch_size"),
        ("block_size", "block_size"),
        ("eval_iters", "eval_iters"),
        ("train_path", "train_path"),
        ("valid_path", "valid_path"),
        ("checkpoint_path", "checkpoint_path"),
        ("init_from_checkpoint", "init_from_checkpoint"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            cfg[cfg_name] = value
    if args.resume_from is not None:
        cfg["resume_from_checkpoint"] = args.resume_from
    if args.streaming_train:
        cfg["streaming_train"] = True
    if args.streaming_valid:
        cfg["streaming_valid"] = True
    if args.streaming_stride is not None:
        cfg["streaming_stride"] = args.streaming_stride
    if args.max_hours is not None:
        cfg["max_train_seconds"] = float(args.max_hours) * 3600.0
    return cfg


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)

    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.set_float32_matmul_precision(str(cfg.get("float32_matmul_precision", "high")))

    train_texts = None
    train_path_for_tokenizer = resolve_path(cfg["train_path"]) if "train_path" in cfg else None
    if str(cfg.get("tokenizer_type", "regex")).lower() != "hf" and not bool(cfg.get("streaming_train", False)):
        if train_path_for_tokenizer is None:
            raise ValueError("Regex tokenizer training requires train_path when streaming_train is false.")
        train_texts = read_text_samples(train_path_for_tokenizer)
    tokenizer = build_tokenizer(cfg, PROJECT_ROOT, train_texts)

    train_dataset, valid_dataset, _ = build_datasets(cfg, tokenizer)
    train_is_iterable = isinstance(train_dataset, IterableDataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=not train_is_iterable,
        drop_last=True,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    model = StructTransformerModel(build_model_config(cfg, tokenizer)).to(device)
    model.gradient_checkpointing = bool(cfg.get("gradient_checkpointing", False))
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Model parameters: {count_parameters(model) / 1_000_000:.2f}M")
    print(f"Gradient checkpointing: {model.gradient_checkpointing}")
    if args.dry_run:
        print("Dry run finished before optimizer/training.")
        return

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.1)),
        betas=(float(cfg.get("beta1", 0.9)), float(cfg.get("beta2", 0.95))),
    )

    amp_enabled = bool(cfg.get("amp", False)) and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(str(cfg.get("amp_dtype", "float16"))) if amp_enabled else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    if amp_enabled:
        print(f"AMP enabled: {amp_dtype}")

    checkpoint_path = cfg["checkpoint_path"]
    start_step = 0
    best_val_loss = math.inf

    resume_path = args.resume_from or cfg.get("resume_from_checkpoint")
    if args.resume and resume_path is None:
        resume_path = checkpoint_path
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, model, optimizer, scaler, device)
        start_step = int(checkpoint.get("step", 0))
        checkpoint_best = checkpoint.get("best_val_loss", checkpoint.get("val_loss"))
        if checkpoint_best is not None:
            best_val_loss = float(checkpoint_best)
        print(f"Resumed checkpoint: {resume_path} (step={start_step})")
    else:
        init_from_checkpoint = args.init_from_checkpoint or cfg.get("init_from_checkpoint")
        resume_optimizer = args.resume_optimizer or bool(cfg.get("resume_optimizer", False))
        if init_from_checkpoint is not None:
            checkpoint = load_checkpoint(
                init_from_checkpoint,
                model,
                optimizer if resume_optimizer else None,
                scaler if resume_optimizer else None,
                device,
            )
            print(f"Loaded checkpoint: {init_from_checkpoint} (step={checkpoint.get('step', '?')})")

    max_steps = int(cfg["max_steps"])
    max_train_seconds = cfg.get("max_train_seconds")
    max_train_seconds = float(max_train_seconds) if max_train_seconds is not None else None
    gradient_accumulation_steps = int(cfg.get("gradient_accumulation_steps", 1))
    if gradient_accumulation_steps <= 0:
        raise ValueError(f"gradient_accumulation_steps must be positive, got {gradient_accumulation_steps}.")
    eval_interval = int(cfg["eval_interval"])
    save_interval = int(cfg["save_interval"])
    eval_iters = int(cfg.get("eval_iters", 20))
    grad_clip = float(cfg.get("grad_clip", 1.0))

    if max_train_seconds is not None:
        print(f"Max train time: {max_train_seconds / 3600.0:.2f} hours")

    model.train()
    data_iter = iter(train_loader)
    start_time = time.time()
    last_step = start_step
    progress = tqdm(range(start_step + 1, max_steps + 1), initial=start_step, total=max_steps, desc="struct training")

    for step in progress:
        last_step = step
        lr = learning_rate_for_step(cfg, step)
        set_optimizer_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)
        metric_sums = {"loss": 0.0, "lm_loss": 0.0, "depth_loss": 0.0, "state_loss": 0.0}

        for _ in range(gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration as exc:
                data_iter = iter(train_loader)
                try:
                    batch = next(data_iter)
                except StopIteration as retry_exc:
                    raise RuntimeError(
                        "Training dataset produced no batches. Check that train_path exists, is non-empty, "
                        "contains JSONL rows like {\"text\": \"...\"}, and has enough tokens for block_size "
                        f"and batch_size. Current train_path={cfg.get('train_path')!r}, "
                        f"block_size={cfg.get('block_size')}, batch_size={cfg.get('batch_size')}."
                    ) from retry_exc

            input_ids, labels, depth_ids, state_ids, depth_targets, state_targets = move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                out = model(input_ids, depth_ids, state_ids, labels, depth_targets, state_targets)
                loss = out["loss"]
                if loss is None:
                    raise RuntimeError("Struct model returned no training loss.")
                if torch.isnan(loss):
                    raise RuntimeError("Training loss became NaN.")
                scaled_loss = loss / gradient_accumulation_steps

            scaler.scale(scaled_loss).backward()
            for key in metric_sums:
                value = out[key]
                metric_sums[key] += float(value.item()) if value is not None else 0.0

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        metrics = {key: value / gradient_accumulation_steps for key, value in metric_sums.items()}
        progress.set_postfix(
            total=f"{metrics['loss']:.4f}",
            lm=f"{metrics['lm_loss']:.4f}",
            depth=f"{metrics['depth_loss']:.4f}",
            state=f"{metrics['state_loss']:.4f}",
            lr=f"{lr:.2e}",
        )

        if step % eval_interval == 0 or step == 1:
            val = estimate_loss(model, valid_loader, device, eval_iters)
            if val["loss"] < best_val_loss:
                best_val_loss = val["loss"]
                save_checkpoint(checkpoint_path, model, optimizer, scaler, cfg, step, best_val_loss)
            progress.set_postfix(
                total=f"{metrics['loss']:.4f}",
                val=f"{val['loss']:.4f}",
                lm=f"{val['lm_loss']:.4f}",
                depth=f"{val['depth_loss']:.4f}",
                state=f"{val['state_loss']:.4f}",
                lr=f"{lr:.2e}",
            )

        if step % save_interval == 0:
            save_checkpoint(checkpoint_path, model, optimizer, scaler, cfg, step, best_val_loss)

        if max_train_seconds is not None and time.time() - start_time >= max_train_seconds:
            print(f"Reached max train time at step {step}; saving checkpoint.")
            break

    save_checkpoint(checkpoint_path, model, optimizer, scaler, cfg, last_step, best_val_loss)


if __name__ == "__main__":
    main()
