"""Shared PM-NTP inner-loop update logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch.optim import SGD

from .lora import LoRALinear, lora_b_parameters, restore_B, snapshot_B
from .losses import pm_ntp_loss


@dataclass
class InnerLoopTrace:
    init_loss: float
    best_loss: float
    losses: list[float]


def polish_pm_ntp(
    model,
    slots: list[tuple[str, LoRALinear]],
    prompt_ids: torch.Tensor,
    *,
    steps: int = 8,
    lr: float = 1e-3,
    clip_norm: float = 1.0,
) -> InnerLoopTrace:
    """Inference-time PM-NTP SGD with best-loss rollback."""

    @torch.no_grad()
    def evaluate() -> float:
        model.eval()
        return float(pm_ntp_loss(model, prompt_ids).item())

    trace = [evaluate()]
    best = trace[0]
    best_snapshot = snapshot_B(slots)
    optim = SGD(lora_b_parameters(slots), lr=lr)
    for _ in range(max(0, int(steps))):
        model.train()
        loss = pm_ntp_loss(model, prompt_ids)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_b_parameters(slots), clip_norm)
        optim.step()
        current = evaluate()
        trace.append(current)
        if current < best:
            best = current
            best_snapshot = snapshot_B(slots)
    restore_B(slots, best_snapshot)
    return InnerLoopTrace(init_loss=trace[0], best_loss=best, losses=trace)


def differentiable_sgd_unroll(
    params: dict[str, torch.Tensor],
    loss_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    *,
    steps: int = 8,
    lr: float = 1e-3,
    first_order: bool = False,
) -> dict[str, torch.Tensor]:
    """Functional SGD unroll for Stage 3.

    `loss_fn` must close over the model via `torch.func.functional_call` or an
    equivalent override mechanism.  With `first_order=False`, autograd keeps the
    Hessian-vector terms required by exact MAML.
    """

    fast = {name: value for name, value in params.items()}
    for _ in range(max(0, int(steps))):
        loss = loss_fn(fast)
        grads = torch.autograd.grad(
            loss,
            tuple(fast.values()),
            create_graph=not first_order,
            retain_graph=not first_order,
            allow_unused=False,
        )
        if first_order:
            grads = tuple(grad.detach() for grad in grads)
        fast = {name: value - lr * grad for (name, value), grad in zip(fast.items(), grads)}
    return fast

