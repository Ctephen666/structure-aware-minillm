"""Formal 200M Chinese structure-aware pretraining entrypoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
import yaml
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.struct_transformer import StructTransformerConfig, StructTransformerModel
from tokenizer.tokenizer_factory import build_tokenizer
from train.structure_dataset import StructureLanguageModelingDataset, encode_structure_row, tensors_from_slices
from parser.structure_annotator import StructureAnnotator
from tools.estimate_params import estimate_struct_params


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


def resolve_amp_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported amp_dtype: {name}")


def build_model_config(cfg: dict[str, Any], tokenizer) -> StructTransformerConfig:
    return StructTransformerConfig(
        vocab_size=int(tokenizer.vocab_size),
        block_size=int(cfg["block_size"]),
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        n_embd=int(cfg["n_embd"]),
        dropout=float(cfg["dropout"]),
        max_depth=int(cfg.get("max_depth", 32)),
        num_states=int(cfg.get("num_states", 9)),
        lambda_depth=float(cfg.get("lambda_depth", 0.03)),
        lambda_state=float(cfg.get("lambda_state", 0.05)),
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def move_batch(batch, device: torch.device):
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


def json_log(path: Path, data: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")


class ManifestShardDataset(IterableDataset):
    """Stream fixed-size chunks from JSONL shards listed in a manifest."""

    def __init__(
        self,
        manifest_path: Path,
        tokenizer,
        block_size: int,
        seed: int = 42,
        shuffle_shards: bool = True,
        shuffle_buffer_size: int = 4096,
        repeat: bool = True,
    ) -> None:
        self.manifest_path = manifest_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.seed = seed
        self.shuffle_shards = shuffle_shards
        self.shuffle_buffer_size = max(1, int(shuffle_buffer_size))
        self.repeat = repeat
        self.shards = self._load_shards()

    def _load_shards(self) -> list[Path]:
        with self.manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        shards = [resolve_path(item["path"]) for item in manifest.get("shards", [])]
        if not shards:
            raise ValueError(f"Manifest has no shards: {self.manifest_path}")
        missing = [path for path in shards if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing shard files: {missing[:3]}")
        return shards

    def _iter_rows(self, path: Path):
        with path.open("r", encoding="utf-8", errors="replace") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    row = line
                yield row

    def _shuffle_rows(self, rows, rng: random.Random):
        buffer = []
        for row in rows:
            buffer.append(row)
            if len(buffer) >= self.shuffle_buffer_size:
                rng.shuffle(buffer)
                while buffer:
                    yield buffer.pop()
        rng.shuffle(buffer)
        while buffer:
            yield buffer.pop()

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        epoch = 0

        while True:
            rng = random.Random(self.seed + epoch)
            shards = list(self.shards)
            if self.shuffle_shards:
                rng.shuffle(shards)

            annotator = StructureAnnotator()
            input_ids: list[int] = []
            labels: list[int] = []
            depth_ids: list[int] = []
            state_ids: list[int] = []
            depth_targets: list[int] = []
            state_targets: list[int] = []

            for shard_index, shard in enumerate(shards):
                if shard_index % num_workers != worker_id:
                    continue
                row_iter = self._shuffle_rows(self._iter_rows(shard), rng)
                for row in row_iter:
                    encoded = encode_structure_row(row, self.tokenizer, annotator)
                    if encoded is None:
                        continue
                    row_input, row_labels, row_depth, row_state, row_depth_targets, row_state_targets = encoded
                    input_ids.extend(row_input)
                    labels.extend(row_labels)
                    depth_ids.extend(row_depth)
                    state_ids.extend(row_state)
                    depth_targets.extend(row_depth_targets)
                    state_targets.extend(row_state_targets)

                    while len(input_ids) >= self.block_size:
                        item = slice(0, self.block_size)
                        yield tensors_from_slices(
                            input_ids,
                            labels,
                            depth_ids,
                            state_ids,
                            depth_targets,
                            state_targets,
                            item,
                        )
                        del input_ids[: self.block_size]
                        del labels[: self.block_size]
                        del depth_ids[: self.block_size]
                        del state_ids[: self.block_size]
                        del depth_targets[: self.block_size]
                        del state_targets[: self.block_size]
            if not self.repeat:
                return
            epoch += 1


def learning_rate_for_step(cfg: dict[str, Any], step: int) -> float:
    max_lr = float(cfg["learning_rate"])
    min_lr = float(cfg.get("min_lr", 0.0))
    warmup_steps = int(cfg.get("warmup_steps", 0))
    max_steps = int(cfg["max_steps"])
    if warmup_steps > 0 and step <= warmup_steps:
        return max_lr * step / warmup_steps
    if str(cfg.get("lr_schedule", "cosine")).lower() != "cosine":
        return max_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    cosine_coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + cosine_coeff * (max_lr - min_lr)


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


@torch.no_grad()
def estimate_loss(model: StructTransformerModel, loader: DataLoader, device: torch.device, eval_iters: int) -> dict[str, float]:
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


def rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    model: StructTransformerModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    step: int,
    tokens_seen: int,
    best_valid_loss: float,
    dataset_progress: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "step": step,
            "tokens_seen": tokens_seen,
            "best_valid_loss": None if math.isinf(best_valid_loss) else best_valid_loss,
            "rng_state": rng_state(),
            "dataset_progress": dataset_progress,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: StructTransformerModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint.get("rng_state", {}))
    return checkpoint


def build_loaders(cfg: dict[str, Any], tokenizer) -> tuple[DataLoader, DataLoader]:
    train_dataset = ManifestShardDataset(
        manifest_path=resolve_path(cfg["train_manifest"]),
        tokenizer=tokenizer,
        block_size=int(cfg["block_size"]),
        seed=int(cfg.get("seed", 42)),
        shuffle_shards=bool(cfg.get("shuffle_shards", True)),
        shuffle_buffer_size=int(cfg.get("shuffle_buffer_size", 4096)),
        repeat=True,
    )
    valid_dataset = StructureLanguageModelingDataset(resolve_path(cfg["valid_path"]), tokenizer, int(cfg["block_size"]))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        drop_last=True,
        num_workers=int(cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, valid_loader


def maybe_write_samples(model_path: Path, cfg: dict[str, Any], step: int, device_name: str, log) -> None:
    try:
        from decode.generate_pretrain import DEFAULT_PROMPTS, generate_response, load_model
    except Exception as exc:
        log(f"sample generation skipped: {exc}")
        return
    if not model_path.exists():
        return
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    tokenizer = build_tokenizer(cfg, PROJECT_ROOT, train_texts=None)
    model = load_model(model_path, tokenizer, device)
    sample_path = resolve_path(cfg["runs_dir"]) / f"samples_step_{step}.txt"
    with sample_path.open("w", encoding="utf-8") as file:
        for prompt in DEFAULT_PROMPTS:
            output = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=120,
                temperature=0.6,
                top_k=40,
            )
            file.write(f"PROMPT: {prompt}\n{output}\n\n")
    log(f"samples saved: {sample_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/struct_pretrain_200m_zh.yaml")
    parser.add_argument("--resume", nargs="?", const="__default__", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-samples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg["max_steps"] = int(args.max_steps)
    if args.checkpoint_path is not None:
        cfg["checkpoint_path"] = args.checkpoint_path

    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.set_float32_matmul_precision(str(cfg.get("float32_matmul_precision", "high")))

    runs_dir = resolve_path(cfg["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = runs_dir / "train.log"
    metrics_path = runs_dir / "metrics.jsonl"
    config_snapshot_path = runs_dir / "config_snapshot.yaml"
    config_snapshot_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    def log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line)
        with train_log_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    tokenizer = build_tokenizer(cfg, PROJECT_ROOT, train_texts=None)
    counts = estimate_struct_params(cfg, int(tokenizer.vocab_size))
    model = StructTransformerModel(build_model_config(cfg, tokenizer)).to(device)
    model.gradient_checkpointing = bool(cfg.get("gradient_checkpointing", False))

    actual_params = count_parameters(model)
    log(f"tokenizer vocab size: {tokenizer.vocab_size}")
    log(f"estimated parameters: {counts['total_parameters']:,}")
    log(f"actual parameters: {actual_params:,}")
    log(f"gradient checkpointing: {model.gradient_checkpointing}")
    if args.dry_run:
        log("dry run finished before dataset/optimizer/training")
        return

    train_loader, valid_loader = build_loaders(cfg, tokenizer)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.1)),
        betas=(float(cfg.get("beta1", 0.9)), float(cfg.get("beta2", 0.95))),
    )

    amp_enabled = bool(cfg.get("use_amp", cfg.get("amp", False))) and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(str(cfg.get("amp_dtype", "float16"))) if amp_enabled else torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    log(f"AMP enabled: {amp_enabled} dtype={amp_dtype}")

    checkpoint_path = resolve_path(cfg["checkpoint_path"])
    best_checkpoint_path = resolve_path(cfg.get("best_checkpoint_path", checkpoint_path.with_name(checkpoint_path.stem + "_best.pt")))
    latest_checkpoint_path = resolve_path(cfg.get("latest_checkpoint_path", checkpoint_path.with_name(checkpoint_path.stem + "_latest.pt")))

    start_step = 0
    tokens_seen = 0
    best_valid_loss = math.inf
    if args.resume is not None:
        resume_path = checkpoint_path if args.resume == "__default__" else resolve_path(args.resume)
        checkpoint = load_checkpoint(resume_path, model, optimizer, scaler, device)
        start_step = int(checkpoint.get("step", 0))
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        if checkpoint.get("best_valid_loss") is not None:
            best_valid_loss = float(checkpoint["best_valid_loss"])
        log(f"resumed checkpoint: {resume_path} step={start_step} tokens_seen={tokens_seen}")
    else:
        log("starting from scratch; no old checkpoint loaded")

    max_steps = int(cfg["max_steps"])
    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    eval_interval = int(cfg["eval_interval"])
    save_interval = int(cfg["save_interval"])
    eval_iters = int(cfg.get("eval_iters", 50))
    grad_clip = float(cfg.get("grad_clip", 1.0))

    data_iter = iter(train_loader)
    model.train()
    start_time = time.time()
    last_train_metrics = {"loss": math.nan, "lm_loss": math.nan, "depth_loss": math.nan, "state_loss": math.nan}

    try:
        for step in range(start_step + 1, max_steps + 1):
            step_start = time.time()
            lr = learning_rate_for_step(cfg, step)
            set_optimizer_lr(optimizer, lr)
            optimizer.zero_grad(set_to_none=True)
            sums = {"loss": 0.0, "lm_loss": 0.0, "depth_loss": 0.0, "state_loss": 0.0}

            for _ in range(grad_accum):
                batch = next(data_iter)
                input_ids, labels, depth_ids, state_ids, depth_targets, state_targets = move_batch(batch, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    out = model(input_ids, depth_ids, state_ids, labels, depth_targets, state_targets)
                    loss = out["loss"]
                    if loss is None:
                        raise RuntimeError("model returned no training loss")
                    if torch.isnan(loss):
                        raise RuntimeError("training loss became NaN")
                    scaled_loss = loss / grad_accum
                scaler.scale(scaled_loss).backward()
                tokens_seen += int(input_ids.numel())
                for key in sums:
                    value = out[key]
                    sums[key] += float(value.item()) if value is not None else 0.0

            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            last_train_metrics = {key: value / grad_accum for key, value in sums.items()}

            if step == 1 or step % eval_interval == 0:
                valid = estimate_loss(model, valid_loader, device, eval_iters)
                elapsed = time.time() - start_time
                steps_done = max(step - start_step, 1)
                tokens_per_second = tokens_seen / max(elapsed, 1e-6)
                remaining_steps = max(max_steps - step, 0)
                estimated_remaining = remaining_steps * (elapsed / steps_done)
                cuda_alloc = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
                cuda_reserved = torch.cuda.memory_reserved(device) if device.type == "cuda" else 0

                metrics = {
                    "step": step,
                    "train_loss": last_train_metrics["loss"],
                    "valid_loss": valid["loss"],
                    "lm_loss": valid["lm_loss"],
                    "depth_loss": valid["depth_loss"],
                    "state_loss": valid["state_loss"],
                    "learning_rate": lr,
                    "tokens_seen": tokens_seen,
                    "tokens_per_second": tokens_per_second,
                    "elapsed_time": elapsed,
                    "estimated_remaining_time": estimated_remaining,
                    "epoch_or_shard_progress": {"step": step, "streaming_manifest": cfg["train_manifest"]},
                    "cuda_memory_allocated": int(cuda_alloc),
                    "cuda_memory_reserved": int(cuda_reserved),
                }
                json_log(metrics_path, metrics)
                log(
                    "step={step} train={train_loss:.4f} valid={valid_loss:.4f} "
                    "lm={lm_loss:.4f} depth={depth_loss:.4f} state={state_loss:.4f} "
                    "lr={learning_rate:.2e} tok/s={tokens_per_second:.1f}".format(**metrics)
                )
                if valid["loss"] < best_valid_loss:
                    best_valid_loss = valid["loss"]
                    save_checkpoint(
                        best_checkpoint_path,
                        model,
                        optimizer,
                        scaler,
                        cfg,
                        step,
                        tokens_seen,
                        best_valid_loss,
                        {"step": step, "tokens_seen": tokens_seen, "manifest": cfg["train_manifest"]},
                    )
                    log(f"best checkpoint saved: {best_checkpoint_path}")

            if step % save_interval == 0:
                progress = {"step": step, "tokens_seen": tokens_seen, "manifest": cfg["train_manifest"]}
                save_checkpoint(checkpoint_path, model, optimizer, scaler, cfg, step, tokens_seen, best_valid_loss, progress)
                save_checkpoint(latest_checkpoint_path, model, optimizer, scaler, cfg, step, tokens_seen, best_valid_loss, progress)
                log(f"checkpoint saved: {checkpoint_path}")
                log(f"latest checkpoint saved: {latest_checkpoint_path}")
                if not args.no_samples:
                    maybe_write_samples(checkpoint_path, cfg, step, str(device), log)

            if step % 10 == 0:
                log(
                    f"step={step} train={last_train_metrics['loss']:.4f} "
                    f"lr={lr:.2e} tokens_seen={tokens_seen:,} step_time={time.time() - step_start:.2f}s"
                )
    except KeyboardInterrupt:
        log("interrupted; saving latest checkpoint before exit")
        progress = {"step": step, "tokens_seen": tokens_seen, "manifest": cfg["train_manifest"]}
        save_checkpoint(latest_checkpoint_path, model, optimizer, scaler, cfg, step, tokens_seen, best_valid_loss, progress)
        raise

    progress = {"step": max_steps, "tokens_seen": tokens_seen, "manifest": cfg["train_manifest"]}
    save_checkpoint(checkpoint_path, model, optimizer, scaler, cfg, max_steps, tokens_seen, best_valid_loss, progress)
    save_checkpoint(latest_checkpoint_path, model, optimizer, scaler, cfg, max_steps, tokens_seen, best_valid_loss, progress)
    log("training finished")


if __name__ == "__main__":
    main()
