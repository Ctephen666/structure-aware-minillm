"""Generate text with the structure-aware MiniLLM."""

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
from tokenizer.regex_tokenizer import RegexTokenizer


def build_instruction_prompt(prompt: str) -> str:
    return "### Instruction:\n" + prompt.strip() + "\n\n### Response:\n"


def load_model(model_path: str | Path, tokenizer: RegexTokenizer, device: torch.device) -> StructTransformerModel:
    checkpoint = torch.load(PROJECT_ROOT / model_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    state_dict = checkpoint["model"]

    checkpoint_vocab_size = state_dict["token_embedding.weight"].shape[0]
    if checkpoint_vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size={tokenizer.vocab_size}, but checkpoint vocab_size={checkpoint_vocab_size}. "
            "Please use the tokenizer saved with this checkpoint."
        )

    model_cfg = StructTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=int(cfg.get("block_size", 512)),
        n_layer=int(cfg.get("n_layer", 4)),
        n_head=int(cfg.get("n_head", 4)),
        n_embd=int(cfg.get("n_embd", 256)),
        dropout=float(cfg.get("dropout", 0.0)),
        max_depth=int(cfg.get("max_depth", 32)),
        num_states=int(cfg.get("num_states", 9)),
        lambda_depth=float(cfg.get("lambda_depth", 0.1)),
        lambda_state=float(cfg.get("lambda_state", 0.2)),
    )
    model = StructTransformerModel(model_cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def annotate_id_context(input_ids: list[int], tokenizer: RegexTokenizer) -> tuple[list[int], list[int]]:
    if not input_ids:
        return [], []

    tokens = [tokenizer.id_to_token.get(int(idx), tokenizer.unk_token) for idx in input_ids[1:]]
    annotator = StructureAnnotator()
    depth_ids, state_ids = annotator.annotate_tokens(tokens)
    return [0] + depth_ids, [PLAIN] + state_ids


@torch.no_grad()
def generate_response(
    model: StructTransformerModel,
    tokenizer: RegexTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 50,
    do_sample: bool = True,
) -> str:
    formatted_prompt = build_instruction_prompt(prompt)
    input_ids = [tokenizer.bos_id] + tokenizer.encode(formatted_prompt, add_special_tokens=False)
    prompt_len = len(input_ids)

    for _ in range(max_new_tokens):
        depth_ids, state_ids = annotate_id_context(input_ids, tokenizer)
        context_ids = input_ids[-model.config.block_size :]
        context_depth = depth_ids[-model.config.block_size :]
        context_state = state_ids[-model.config.block_size :]

        input_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)
        depth_tensor = torch.tensor([context_depth], dtype=torch.long, device=device)
        state_tensor = torch.tensor([context_state], dtype=torch.long, device=device)
        out = model(input_tensor, depth_tensor, state_tensor)
        logits = out["lm_logits"][:, -1, :]

        if not do_sample:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / max(float(temperature), 1e-6)
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        next_value = int(next_id.item())
        if next_value == tokenizer.eos_id:
            break
        input_ids.append(next_value)

    new_ids = input_ids[prompt_len:]
    return tokenizer.decode(new_ids).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model checkpoint, e.g. checkpoints/struct.pt")
    parser.add_argument("--tokenizer", default="checkpoints/baseline_tokenizer.json")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.prompt is None and args.prompt_file is None:
        raise ValueError("Please provide --prompt or --prompt-file.")

    prompt = args.prompt
    if args.prompt_file is not None:
        prompt = (PROJECT_ROOT / args.prompt_file).read_text(encoding="utf-8")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = RegexTokenizer.load(PROJECT_ROOT / args.tokenizer)
    model = load_model(args.model, tokenizer, device)

    response = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        do_sample=not args.greedy,
    )

    if args.output:
        out_path = PROJECT_ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(response, encoding="utf-8")

    print(response)


if __name__ == "__main__":
    main()
