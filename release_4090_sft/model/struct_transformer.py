"""Structure-aware decoder-only Transformer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from model.transformer import TransformerBlock
from parser.structure_states import MAX_DEPTH, NUM_STATES


@dataclass
class StructTransformerConfig:
    vocab_size: int
    block_size: int = 512
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    max_depth: int = MAX_DEPTH
    num_states: int = NUM_STATES
    lambda_depth: float = 0.1
    lambda_state: float = 0.2


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    if not torch.any(targets != ignore_index):
        return logits.sum() * 0.0
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=ignore_index)


class StructTransformerModel(nn.Module):
    """GPT-style LM conditioned on token, position, depth, and parser state."""

    def __init__(self, config: StructTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.depth_embedding = nn.Embedding(config.max_depth + 1, config.n_embd)
        self.state_embedding = nn.Embedding(config.num_states, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.depth_head = nn.Linear(config.n_embd, config.max_depth + 1)
        self.state_head = nn.Linear(config.n_embd, config.num_states)
        self.token_embedding.weight = self.lm_head.weight
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        depth_ids: torch.Tensor,
        state_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        depth_targets: torch.Tensor | None = None,
        state_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        _, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size {self.config.block_size}.")

        depth_ids = depth_ids.clamp(0, self.config.max_depth)
        state_ids = state_ids.clamp(0, self.config.num_states - 1)
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.depth_embedding(depth_ids)
            + self.state_embedding(state_ids)
        )
        x = self.dropout(x)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)

        lm_logits = self.lm_head(x)
        depth_logits = self.depth_head(x)
        state_logits = self.state_head(x)

        lm_loss = None
        depth_loss = None
        state_loss = None
        total_loss = None
        if targets is not None:
            lm_loss = masked_cross_entropy(lm_logits, targets)
            if depth_targets is not None:
                depth_loss = masked_cross_entropy(depth_logits, depth_targets)
            else:
                depth_loss = lm_logits.sum() * 0.0
            if state_targets is not None:
                state_loss = masked_cross_entropy(state_logits, state_targets)
            else:
                state_loss = lm_logits.sum() * 0.0
            total_loss = lm_loss + self.config.lambda_depth * depth_loss + self.config.lambda_state * state_loss

        return {
            "lm_logits": lm_logits,
            "depth_logits": depth_logits,
            "state_logits": state_logits,
            "loss": total_loss,
            "lm_loss": lm_loss,
            "depth_loss": depth_loss,
            "state_loss": state_loss,
        }
