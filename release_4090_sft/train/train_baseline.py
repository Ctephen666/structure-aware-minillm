"""Train the baseline decoder-only Transformer."""

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

from model.transformer import TransformerConfig, TransformerModel
from tokenizer.regex_tokenizer import RegexTokenizer
from train.dataset import LanguageModelingDataset, read_text_samples


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


@torch.no_grad()
def estimate_loss(model: TransformerModel, loader: DataLoader, device: torch.device, eval_iters: int) -> float:
    model.eval()
    losses: list[float] = []
    for step, (x, y) in enumerate(loader):
        if step >= eval_iters:
            break
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(len(losses), 1)


def save_checkpoint(
    path: str | Path,
    model: TransformerModel,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    torch.manual_seed(int(cfg.get("seed", 42)))

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_path = PROJECT_ROOT / cfg["train_path"]
    valid_path = PROJECT_ROOT / cfg["valid_path"]
    train_texts = read_text_samples(train_path)
    valid_texts = read_text_samples(valid_path) if valid_path.exists() else train_texts[: max(1, len(train_texts) // 10)]

    tokenizer_path = PROJECT_ROOT / cfg.get("tokenizer_path", "checkpoints/baseline_tokenizer.json")
    if tokenizer_path.exists():
        tokenizer = RegexTokenizer.load(tokenizer_path)
    else:
        tokenizer = RegexTokenizer.train_from_texts(train_texts, vocab_size=int(cfg["vocab_size"]))
        tokenizer.save(tokenizer_path)

    block_size = int(cfg["block_size"])
    train_dataset = LanguageModelingDataset(train_texts, tokenizer, block_size)
    valid_dataset = LanguageModelingDataset(valid_texts, tokenizer, block_size)
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

    model_cfg = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        n_embd=int(cfg["n_embd"]),
        dropout=float(cfg["dropout"]),
    )
    model = TransformerModel(model_cfg).to(device)
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
    progress = tqdm(range(1, max_steps + 1), desc="training")
    best_val_loss = math.inf
    for step in progress:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        progress.set_postfix(train_loss=f"{loss.item():.4f}")

        if step % eval_interval == 0 or step == 1:
            val_loss = estimate_loss(model, valid_loader, device, eval_iters)
            progress.set_postfix(train_loss=f"{loss.item():.4f}", val_loss=f"{val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(checkpoint_path, model, optimizer, cfg, step, val_loss)

        if step % save_interval == 0:
            save_checkpoint(checkpoint_path, model, optimizer, cfg, step, best_val_loss)

    save_checkpoint(checkpoint_path, model, optimizer, cfg, max_steps, best_val_loss)


if __name__ == "__main__":
    main()
