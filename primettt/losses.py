"""PrimeTTT NTP objectives and answer/prompt masks."""

from __future__ import annotations

import torch


def _labels_like(input_ids: torch.Tensor) -> torch.Tensor:
    return input_ids.clone()


def pm_ntp_loss(model, prompt_ids: torch.Tensor) -> torch.Tensor:
    """Prompt-only NTP used by test-time polishing."""
    return model(prompt_ids, labels=_labels_like(prompt_ids), use_cache=False).loss


def ac_ntp_loss(model, full_ids: torch.Tensor, answer_start: int) -> torch.Tensor:
    """Answer-conditioned NTP: mask prompt labels and score answer tokens."""
    labels = _labels_like(full_ids)
    labels[:, :answer_start] = -100
    return model(full_ids, labels=labels, use_cache=False).loss


def full_ntp_loss(model, full_ids: torch.Tensor, answer_start: int | None = None) -> torch.Tensor:
    """Full sequence NTP used for the appendix ablation."""
    del answer_start
    return model(full_ids, labels=_labels_like(full_ids), use_cache=False).loss


def masked_prompt_loss(model, full_ids: torch.Tensor, answer_start: int) -> torch.Tensor:
    labels = _labels_like(full_ids)
    labels[:, answer_start:] = -100
    return model(full_ids, labels=labels, use_cache=False).loss

