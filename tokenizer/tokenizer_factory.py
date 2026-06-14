import os
from pathlib import Path
from tokenizer.regex_tokenizer import RegexTokenizer

def build_tokenizer(cfg, project_root: Path, train_texts=None):
    tokenizer_type = cfg.get("tokenizer_type", "auto")
    tokenizer_name = cfg.get("tokenizer_name", "baseline")
    hf_cache_dir = cfg.get("hf_cache_dir", None)
    
    if tokenizer_type == "regex" or tokenizer_name == "baseline":
        tokenizer_path = project_root / f"checkpoints/{tokenizer_name}_tokenizer.json"
        if train_texts is not None and not tokenizer_path.exists():
            print("Building regex tokenizer...")
            tokenizer = RegexTokenizer()
            tokenizer.train(train_texts, vocab_size=cfg.get("vocab_size", 256))
            tokenizer.save(tokenizer_path)
            return tokenizer
        else:
            return RegexTokenizer.load(tokenizer_path)
            
    elif tokenizer_type == "hf":
        from tokenizer.hf_tokenizer import HFTokenizer
        kwargs = {}
        if hf_cache_dir:
            kwargs["cache_dir"] = str(project_root / hf_cache_dir)
        return HFTokenizer.from_pretrained(tokenizer_name, **kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

def load_tokenizer(
    tokenizer_ref: str,
    project_root: Path,
    tokenizer_type: str = "auto",
    hf_cache_dir: str | None = None,
    local_files_only: bool = False,
):
    tokenizer_path = project_root / tokenizer_ref
    
    # 🌟 关键防御：只有当路径存在【并且它确实是个单文件】时，才走 RegexTokenizer
    if tokenizer_type == "regex" or (tokenizer_path.exists() and tokenizer_path.is_file()):
        return RegexTokenizer.load(tokenizer_path)

    # 📂 路径是文件夹，说明是 Hugging Face 格式词表，完美分流
    from tokenizer.hf_tokenizer import HFTokenizer

    kwargs = {}
    if hf_cache_dir:
        kwargs["cache_dir"] = str(project_root / hf_cache_dir)
    if local_files_only:
        kwargs["local_files_only"] = True
        
    return HFTokenizer.from_pretrained(tokenizer_ref, **kwargs)