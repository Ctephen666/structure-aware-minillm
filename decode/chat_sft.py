import argparse
import sys
from pathlib import Path
import torch

# 将项目根目录加入环境路径
PROJECT_ROOT = Path(__file__).absolute().parents[1]
sys.path.append(str(PROJECT_ROOT))

from model.struct_transformer import StructTransformerModel, StructTransformerConfig
from tokenizer.tokenizer_factory import load_tokenizer

def load_model(model_path: str, tokenizer, device: str):
    print(f"Loading checkpoint from {model_path}...")
    # weights_only=False 消除未来警告
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 🌟 核心安全机制：直接根据你训练的 configs/struct_sft_200m_4090.yaml 硬编码模型结构
    model_cfg = StructTransformerConfig(
        block_size=1024,
        vocab_size=tokenizer.vocab_size,  # 动态完美对齐 UER 21128 词表
        n_layer=26,                       # 对应 200M 模型层数
        n_head=12,                        # 对应 200M 模型头数
        n_embd=768,                       # 对应 200M 模型隐藏层维度
        dropout=0.1,
        max_depth=32,
        num_states=9,
        lambda_depth=0.03,
        lambda_state=0.05,
    )
    
    model = StructTransformerModel(model_cfg)
    
    # 自动识别是复合 checkpoint 还是单纯的 state_dict
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_chat_prompt(prompt: str, history: list[tuple[str, str]] | None = None, history_turns: int = 3) -> str:
    """Build the same SFT-style prompt, with optional recent dialogue history."""
    history = history or []
    kept_history = history[-history_turns:] if history_turns > 0 else []

    if not kept_history:
        return f"问题：{prompt.strip()}\n回答："

    parts = ["以下是用户与助手的多轮对话。请结合上下文回答当前用户问题。\n"]
    for user_text, assistant_text in kept_history:
        parts.append(f"问题：{user_text.strip()}\n回答：{assistant_text.strip()}\n")
    parts.append(f"问题：{prompt.strip()}\n回答：")
    return "".join(parts)


def clean_generated_text(text: str) -> str:
    """Stop the model from spilling into the next artificial dialogue turn."""
    stop_markers = ["\n问题：", "\n回答：", "\nPrompt:", "\nAnswer:", "用户：", "助手：", "User ＞", "Model ＞"]
    for marker in stop_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def get_stop_token_ids(tokenizer) -> set[int]:
    """Collect real special token ids used by local/HF tokenizer wrappers."""
    stop_ids: set[int] = set()
    for attr in [
        "eos_id",
        "eos_token_id",
        "sep_id",
        "sep_token_id",
        "pad_id",
        "pad_token_id",
    ]:
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int) and value >= 0:
            stop_ids.add(int(value))
    # Some local tokenizers use 0 as PAD. Keep it as a safe stop id.
    stop_ids.add(0)
    return stop_ids


def has_text_stop_marker(text: str) -> bool:
    """Detect common spillover markers while generating."""
    markers = [
        "\n问题：",
        "\n回答：",
        "\nPrompt:",
        "\nAnswer:",
        "用户：",
        "助手：",
    ]
    return any(marker in text for marker in markers)

@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    history: list[tuple[str, str]] | None = None,
    history_turns: int = 3,
):
    # 严格按照 SFT 阶段的推理包裹格式；交互模式下可拼接最近几轮历史
    raw_text = build_chat_prompt(prompt, history=history, history_turns=history_turns)
    
    if hasattr(tokenizer, 'encode'):
        input_ids = tokenizer.encode(raw_text)
    else:
        input_ids = tokenizer.encode_text(raw_text) if hasattr(tokenizer, 'encode_text') else tokenizer(raw_text)["input_ids"]
        
    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    generated_ids = []
    stop_token_ids = get_stop_token_ids(tokenizer)
    for _ in range(max_new_tokens):
        x_cond = x if x.size(1) <= 1024 else x[:, -1024:]
        
        # 结构感知模型（StructTransformer）前向传播
        # 推理阶段先使用默认 TEXT 状态：depth=0, state=0
        depth_ids = torch.zeros_like(x_cond, dtype=torch.long, device=device)
        state_ids = torch.zeros_like(x_cond, dtype=torch.long, device=device)

        try:
            out = model(
                input_ids=x_cond,
                depth_ids=depth_ids,
                state_ids=state_ids,
            )
        except TypeError:
            # 兼容部分旧版 forward(input_ids, depth_ids, state_ids) 写法
            out = model(x_cond, depth_ids, state_ids)

        # 兼容 dict / tuple / tensor 三种返回格式
        if isinstance(out, dict):
            logits = out.get("logits", None)
            if logits is None:
                logits = out.get("lm_logits", None)
            if logits is None:
                raise KeyError("Model output dict does not contain 'logits' or 'lm_logits'.")
        elif isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out

        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None and top_k > 0:
            top_values, top_indices = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(dim=-1, index=top_indices, src=top_values)
            logits = filtered
        probs = torch.softmax(logits, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1).item()
        
        # 遇到真实 EOS/SEP/PAD 等特殊符号立即停止。
        # 注意：本项目 tokenizer wrapper 常用 eos_id，而不是 HF 的 eos_token_id。
        if next_id in stop_token_ids:
            break

        generated_ids.append(next_id)

        # 如果文本层面已经进入下一轮样本/下一轮对话，也提前停止，避免串样本。
        partial_text = tokenizer.decode(generated_ids)
        if has_text_stop_marker(partial_text):
            break

        x = torch.cat((x, torch.tensor([[next_id]], device=device)), dim=1)
        
    return clean_generated_text(tokenizer.decode(generated_ids))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to your trained .pt checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer folder or json file")
    parser.add_argument("--tokenizer_type", type=str, default="auto")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--interactive", action="store_true", default=True)
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt mode. If set, generate once and exit.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--history-turns", type=int, default=3, help="Number of recent dialogue turns to keep in interactive mode.")
    parser.add_argument("--no-history", action="store_true", help="Disable multi-turn dialogue history in interactive mode.")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    # 安全分流并载入 21128 全量中文分词器
    tokenizer = load_tokenizer(
        tokenizer_ref=args.tokenizer,
        project_root=PROJECT_ROOT,
        tokenizer_type=args.tokenizer_type,
        local_files_only=args.local_files_only
    )
    
    # 强制将 200M 模型与本地分词器结构对齐并载入权重
    model = load_model(args.model, tokenizer, args.device)

    if args.prompt is not None:
        response = generate_response(
            model,
            tokenizer,
            args.prompt,
            args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            history=None,
            history_turns=0,
        )
        print(response)
        return
    
    print("\n" + "="*50)
    print("✨ 结构感知大模型（Structure-Aware MiniLLM）200M SFT 交互就绪！")
    print("输入 'exit' 或 'quit' 可退出对话。")
    print("="*50 + "\n")
    print("连续对话已开启：默认保留最近 %d 轮上下文。输入 clear 可清空历史。\n" % args.history_turns)

    history: list[tuple[str, str]] = []
    
    while True:
        try:
            user_input = input("User ＞ ")
            lower_input = user_input.strip().lower()
            if lower_input in ["exit", "quit"]:
                break
            if lower_input in ["clear", "reset", "清空"]:
                history.clear()
                print("Model ＞ 已清空对话历史。\n")
                continue
            if not user_input.strip():
                continue
                
            response = generate_response(
                model,
                tokenizer,
                user_input,
                args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                history=None if args.no_history else history,
                history_turns=0 if args.no_history else args.history_turns,
            )
            print(f"Model ＞ {response}\n")
            if not args.no_history:
                history.append((user_input, response))
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()