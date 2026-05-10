"""Train the structure-aware decoder-only Transformer."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.struct_transformer import StructTransformerConfig, StructTransformerModel
from train.dataset import read_text_samples
from train.structure_dataset import StructureLanguageModelingDataset
from tokenizer.tokenizer_factory import build_tokenizer


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def move_batch(batch, device: torch.device):
    return tuple(tensor.to(device) for tensor in batch)


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
    config: dict,
    step: int,
    val_loss: float | None,
) -> None:
    path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


def build_model_config(cfg: dict, tokenizer) -> StructTransformerConfig:
    return StructTransformerConfig(
        vocab_size=tokenizer.vocab_size,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "struct.yaml"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--train-path", default=None)
    parser.add_argument("--valid-path", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.block_size is not None:
        cfg["block_size"] = args.block_size
    if args.eval_iters is not None:
        cfg["eval_iters"] = args.eval_iters
    if args.train_path is not None:
        cfg["train_path"] = args.train_path
    if args.valid_path is not None:
        cfg["valid_path"] = args.valid_path
    if args.checkpoint_path is not None:
        cfg["checkpoint_path"] = args.checkpoint_path

    torch.manual_seed(int(cfg.get("seed", 42)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_path = PROJECT_ROOT / cfg["train_path"]
    valid_path = PROJECT_ROOT / cfg["valid_path"]
    train_texts = None
    if str(cfg.get("tokenizer_type", "regex")).lower() != "hf":
        train_texts = read_text_samples(train_path)
    tokenizer = build_tokenizer(cfg, PROJECT_ROOT, train_texts)

    block_size = int(cfg["block_size"])
    train_dataset = StructureLanguageModelingDataset(train_path, tokenizer, block_size)
    valid_source = valid_path if valid_path.exists() else train_path
    valid_dataset = StructureLanguageModelingDataset(valid_source, tokenizer, block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(cfg.get("num_workers", 0)),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(cfg.get("num_workers", 0)),
    )

    model = StructTransformerModel(build_model_config(cfg, tokenizer)).to(device)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Model parameters: {count_parameters(model) / 1_000_000:.2f}M")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.1)),
        betas=(0.9, 0.95),
    )

    max_steps = int(cfg["max_steps"])
    eval_interval = int(cfg["eval_interval"])
    save_interval = int(cfg["save_interval"])
    eval_iters = int(cfg.get("eval_iters", 20))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    checkpoint_path = cfg["checkpoint_path"]

    model.train()
    data_iter = iter(train_loader)
    best_val_loss = math.inf
    progress = tqdm(range(1, max_steps + 1), desc="struct training")
    for step in progress:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids, labels, depth_ids, state_ids, depth_targets, state_targets = move_batch(batch, device)
        out = model(input_ids, depth_ids, state_ids, labels, depth_targets, state_targets)
        loss = out["loss"]
        if loss is None:
            raise RuntimeError("Struct model returned no training loss.")
        if torch.isnan(loss):
            raise RuntimeError("Training loss became NaN.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        progress.set_postfix(
            total=f"{loss.item():.4f}",
            lm=f"{out['lm_loss'].item():.4f}",
            depth=f"{out['depth_loss'].item():.4f}",
            state=f"{out['state_loss'].item():.4f}",
        )

        if step % eval_interval == 0 or step == 1:
            val = estimate_loss(model, valid_loader, device, eval_iters)
            progress.set_postfix(
                total=f"{loss.item():.4f}",
                val=f"{val['loss']:.4f}",
                lm=f"{val['lm_loss']:.4f}",
                depth=f"{val['depth_loss']:.4f}",
                state=f"{val['state_loss']:.4f}",
            )
            if val["loss"] < best_val_loss:
                best_val_loss = val["loss"]
                save_checkpoint(checkpoint_path, model, optimizer, cfg, step, best_val_loss)

        if step % save_interval == 0:
            save_checkpoint(checkpoint_path, model, optimizer, cfg, step, best_val_loss)

    save_checkpoint(checkpoint_path, model, optimizer, cfg, max_steps, best_val_loss)


if __name__ == "__main__":
    main()
