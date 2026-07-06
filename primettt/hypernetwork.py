"""PrimeTTT hypernetwork H_psi."""

from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from .lora import SlotInfo


class AttentionPool(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_size))
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.proj(hidden_states).matmul(self.query)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)
        return torch.einsum("bt,btd->bd", weights, hidden_states)


class PrimeTTTHyperNetwork(nn.Module):
    """Hidden states -> residual Delta B for every LoRA slot.

    The output heads are zero-initialized, so an untrained hypernetwork returns
    B_base exactly.
    """

    def __init__(
        self,
        hidden_size: int,
        slots: list[SlotInfo] | list[dict],
        *,
        rank: int = 8,
        width: int = 512,
        layers: int = 4,
        heads: int = 8,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.slots = [s if isinstance(s, SlotInfo) else SlotInfo(**s) for s in slots]
        self.pool = AttentionPool(hidden_size)
        self.in_proj = nn.Linear(hidden_size, width)
        block = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.core = nn.TransformerEncoder(block, num_layers=layers)
        self.slot_query = nn.Parameter(torch.randn(len(self.slots), width) * (width**-0.5))
        self.out_heads = nn.ModuleDict()
        max_out = max(slot.out_features for slot in self.slots) if self.slots else 0
        self.register_buffer("B_base", torch.zeros(len(self.slots), max_out, rank), persistent=True)
        for idx, slot in enumerate(self.slots):
            head = nn.Linear(width, slot.out_features * rank)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            self.out_heads[str(idx)] = head

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        residual_only: bool = False,
    ) -> dict[str, torch.Tensor]:
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        pooled = self.pool(hidden_states.float(), attention_mask=attention_mask)
        context = self.in_proj(pooled).unsqueeze(1)
        tokens = context + self.slot_query.unsqueeze(0)
        encoded = self.core(tokens)
        out: dict[str, torch.Tensor] = {}
        for idx, slot in enumerate(self.slots):
            delta = self.out_heads[str(idx)](encoded[:, idx]).view(-1, slot.out_features, self.rank)
            base = self.B_base[idx, : slot.out_features].unsqueeze(0).to(delta.device, delta.dtype)
            value = delta if residual_only else base + delta
            out[slot.key] = value.squeeze(0) if value.size(0) == 1 else value
        return out

    def set_B_base(self, values: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for idx, slot in enumerate(self.slots):
                if slot.key in values:
                    self.B_base[idx, : slot.out_features].copy_(values[slot.key].to(self.B_base))

    def manifest(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "rank": self.rank,
            "slots": [asdict(slot) for slot in self.slots],
        }

