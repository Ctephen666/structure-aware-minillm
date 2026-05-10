"""Tokenizer construction helpers."""

from __future__ import annotations

from pathlib import Path

from tokenizer.regex_tokenizer import RegexTokenizer


def build_tokenizer(cfg: dict, project_root: Path, train_texts: list[str] | None = None):
    tokenizer_type = str(cfg.get("tokenizer_type", "regex")).lower()

    if tokenizer_type == "hf":
        from tokenizer.hf_tokenizer import HFTokenizer

        tokenizer_name = cfg.get("tokenizer_name")
        if not tokenizer_name:
            raise ValueError("tokenizer_type=hf requires tokenizer_name.")
        cache_dir = cfg.get("hf_cache_dir")
        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = str(project_root / cache_dir)
        return HFTokenizer.from_pretrained(tokenizer_name, **kwargs)

    tokenizer_path = project_root / cfg.get("tokenizer_path", "checkpoints/baseline_tokenizer.json")
    if tokenizer_path.exists():
        return RegexTokenizer.load(tokenizer_path)
    if train_texts is None:
        raise ValueError(f"Regex tokenizer file does not exist and train_texts were not provided: {tokenizer_path}")
    tokenizer = RegexTokenizer.train_from_texts(train_texts, vocab_size=int(cfg["vocab_size"]))
    tokenizer.save(tokenizer_path)
    return tokenizer


def load_tokenizer(tokenizer_ref: str, project_root: Path, tokenizer_type: str = "auto"):
    tokenizer_path = project_root / tokenizer_ref
    if tokenizer_type == "regex" or tokenizer_path.exists():
        return RegexTokenizer.load(tokenizer_path)

    from tokenizer.hf_tokenizer import HFTokenizer

    return HFTokenizer.from_pretrained(tokenizer_ref)

