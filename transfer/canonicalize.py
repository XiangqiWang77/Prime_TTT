"""Canonicalize grouped-query attention LoRA tensors."""

from __future__ import annotations

import torch


def average_to_kv_heads(tensor: torch.Tensor, num_query_heads: int, num_kv_heads: int) -> torch.Tensor:
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("query heads must be divisible by kv heads")
    group = num_query_heads // num_kv_heads
    return tensor.view(num_kv_heads, group, *tensor.shape[1:]).mean(dim=1)

