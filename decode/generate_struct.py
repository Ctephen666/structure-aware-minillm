
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.struct_transformer import StructTransformerConfig, StructTransformerModel
from parser.structure_annotator import StructureAnnotator
from parser.structure_states import PLAIN
from tokenizer.tokenizer_factory import load_tokenizer


SPECIAL_TOKENS = {"<pad>", "<unk>", "<bos>", "<eos>"}


def annotate_token_ids(
    ids: list[int],
    tokenizer,
    annotator: StructureAnnotator,
) -> tuple[list[int], list[int]]:
    """
    根据当前完整 token id 序列重新计算 depth/state。
    注意：这里用完整上下文标注，然后再截取 block_size，
    避免 block 截断后丢失前文结构状态。
    """
    tokens = []
    positions = []

    for pos, token_id in enumerate(ids):
        token = tokenizer.id_to_token.get(int(token_id), tokenizer.unk_token)
        if token in SPECIAL_TOKENS:
            continue
        tokens.append(token)
        positions.append(pos)

    depth_ids = [0] * len(ids)
    state_ids = [PLAIN] * len(ids)

    if tokens:
        depths, states = annotator.annotate_tokens(tokens)
        for pos, depth, state in zip(positions, depths, states):
            depth_ids[pos] = depth
            state_ids[pos] = state

    return depth_ids, state_ids


def load_model(
    model_path: str | Path,
    tokenizer,
    device: torch.device,
) -> StructTransformerModel:
    ckpt = torch.load(PROJECT_ROOT / model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model_config = StructTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=int(cfg["block_size"]),
        n_layer=int(cfg["n_layer"]),
        n_head=int(cfg["n_head"]),
        n_embd=int(cfg["n_embd"]),
        dropout=float(cfg.get("dropout", 0.0)),
        max_depth=int(cfg.get("max_depth", 32)),
        num_states=int(cfg.get("num_states", 9)),
        lambda_depth=float(cfg.get("lambda_depth", 0.1)),
        lambda_state=float(cfg.get("lambda_state", 0.2)),
    )

    model = StructTransformerModel(model_config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def generate_response(
    model: StructTransformerModel,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 50,
    do_sample: bool = True,
) -> str:
    annotator = StructureAnnotator(max_depth=model.config.max_depth)

    ids = [tokenizer.bos_id] + tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        full_ids = input_ids[0].tolist()
        full_depths, full_states = annotate_token_ids(full_ids, tokenizer, annotator)

        context_ids = full_ids[-model.config.block_size:]
        context_depths = full_depths[-model.config.block_size:]
        context_states = full_states[-model.config.block_size:]

        x = torch.tensor([context_ids], dtype=torch.long, device=device)
        d = torch.tensor([context_depths], dtype=torch.long, device=device)
        s = torch.tensor([context_states], dtype=torch.long, device=device)

        out = model(x, d, s)
        logits = out["lm_logits"][:, -1, :]

        if temperature <= 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / max(temperature, 1e-6)

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))

            probs = F.softmax(logits, dim=-1)
            if do_sample:
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(probs, dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_id], dim=1)

        if int(next_id.item()) == tokenizer.eos_id:
            break

    return tokenizer.decode(input_ids[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer_type = "hf" if "/" in args.tokenizer and not (PROJECT_ROOT / args.tokenizer).exists() else "auto"
    tokenizer = load_tokenizer(args.tokenizer, PROJECT_ROOT, tokenizer_type=tokenizer_type)
    model = load_model(args.model, tokenizer, device)

    output = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        do_sample=not args.greedy,
    )
    print(output)


if __name__ == "__main__":
    main()
