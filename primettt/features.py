"""Feature extraction from the frozen base model."""

from __future__ import annotations

import torch


@torch.no_grad()
def top_hidden_states(model, input_ids: torch.Tensor) -> torch.Tensor:
    was_training = model.training
    model.eval()
    out = model(input_ids, use_cache=False, output_hidden_states=True)
    if was_training:
        model.train()
    return out.hidden_states[-1].float()

