"""LoRA slots used by PrimeTTT.

PrimeTTT keeps A fixed at a deterministic random initialization and trains or
predicts only B.  The release bundle must therefore include either the exact A
tensors or the seed/config used here.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SlotInfo:
    key: str
    in_features: int
    out_features: int


def stable_slot_seed(key: str, base_seed: int = 1729) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


class LoRALinear(nn.Module):
    """Frozen base linear layer plus trainable/predicted B and fixed A."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        slot_seed: int = 0,
    ) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self._B_override: torch.Tensor | None = None

        generator = torch.Generator(device="cpu").manual_seed(int(slot_seed))
        bound = math.sqrt(6.0 / float(base.in_features))
        a_init = torch.empty(rank, base.in_features, dtype=torch.float32)
        a_init.uniform_(-bound, bound, generator=generator)
        a_init = a_init.to(device=base.weight.device, dtype=base.weight.dtype)

        self.register_buffer("A_init", a_init.clone(), persistent=True)
        self.A = nn.Parameter(a_init.clone(), requires_grad=False)
        self.B = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_y = self.base(x)
        b_eff = self._B_override if self._B_override is not None else self.B
        return base_y + F.linear(F.linear(x, self.A.to(x.dtype)), b_eff.to(x.dtype)) * self.scale

    @torch.no_grad()
    def reset(self) -> None:
        self.A.copy_(self.A_init)
        self.B.zero_()
        self._B_override = None

    @torch.no_grad()
    def load_B(self, value: torch.Tensor) -> None:
        self.A.copy_(self.A_init)
        self.B.copy_(value.to(device=self.B.device, dtype=self.B.dtype))
        self._B_override = None


def default_target_modules(model: nn.Module) -> tuple[str, ...]:
    model_type = getattr(getattr(model, "config", None), "model_type", "") or ""
    if model_type == "gpt_oss":
        return ("q_proj", "v_proj")
    return ("q_proj", "v_proj", "down_proj")


def _find_linear(root: nn.Module, name: str) -> tuple[nn.Module, str] | tuple[None, None]:
    for module in root.modules():
        if hasattr(module, name) and isinstance(getattr(module, name), nn.Linear):
            return module, name
    return None, None


def transformer_layers(model: nn.Module) -> Sequence[nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Cannot locate transformer layers on this model")


def inject_lora(
    model: nn.Module,
    top_layers: int = 16,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: Iterable[str] | None = None,
    seed: int = 1729,
) -> list[tuple[str, LoRALinear]]:
    """Replace target Linear modules in the top layers with PrimeTTT LoRA slots."""

    layers = transformer_layers(model)
    targets = tuple(target_modules or default_target_modules(model))
    start = max(0, len(layers) - int(top_layers))
    slots: list[tuple[str, LoRALinear]] = []

    for layer_idx in range(start, len(layers)):
        block = layers[layer_idx]
        for module_name in targets:
            parent, attr = _find_linear(block, module_name)
            if parent is None:
                continue
            base = getattr(parent, attr)
            key = f"L{layer_idx}.{module_name}"
            wrapped = LoRALinear(
                base,
                rank=rank,
                alpha=alpha,
                slot_seed=stable_slot_seed(key, seed),
            )
            setattr(parent, attr, wrapped)
            slots.append((key, wrapped))

    for name, param in model.named_parameters():
        if ".B" not in name:
            param.requires_grad_(False)
    return slots


def slot_info(slots: Sequence[tuple[str, LoRALinear]]) -> list[SlotInfo]:
    return [SlotInfo(key, slot.base.in_features, slot.base.out_features) for key, slot in slots]


def lora_b_parameters(slots: Sequence[tuple[str, LoRALinear]]) -> list[nn.Parameter]:
    return [slot.B for _, slot in slots]


def reset_lora(slots: Sequence[tuple[str, LoRALinear]]) -> None:
    for _, slot in slots:
        slot.reset()


def snapshot_B(slots: Sequence[tuple[str, LoRALinear]]) -> list[torch.Tensor]:
    return [slot.B.detach().clone() for _, slot in slots]


@torch.no_grad()
def restore_B(slots: Sequence[tuple[str, LoRALinear]], snapshot: Sequence[torch.Tensor]) -> None:
    for (_, slot), value in zip(slots, snapshot):
        slot.B.copy_(value.to(device=slot.B.device, dtype=slot.B.dtype))
        slot._B_override = None


@torch.no_grad()
def install_B(slots: Sequence[tuple[str, LoRALinear]], values: dict[str, torch.Tensor]) -> None:
    for key, slot in slots:
        slot.load_B(values[key])


def install_override(slots: Sequence[tuple[str, LoRALinear]], values: dict[str, torch.Tensor]) -> None:
    for key, slot in slots:
        slot._B_override = values[key]


def clear_override(slots: Sequence[tuple[str, LoRALinear]]) -> None:
    for _, slot in slots:
        slot._B_override = None

