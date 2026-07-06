"""Stage 3: MAML fine-tuning through the PM-NTP inner loop."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch
from torch.optim import AdamW

from .features import top_hidden_states
from .hypernetwork import PrimeTTTHyperNetwork
from .lora import clear_override, inject_lora, install_override, reset_lora
from .losses import ac_ntp_loss, pm_ntp_loss
from .model_loader import input_device, load_base_model


def _clone_fast(predicted: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in predicted.items()}


def _pm_unroll(
    model,
    slots,
    prompt_ids: torch.Tensor,
    fast: dict[str, torch.Tensor],
    *,
    steps: int,
    lr: float,
    first_order: bool,
) -> dict[str, torch.Tensor]:
    """K PM-NTP SGD steps. Exact mode keeps Hessian-vector terms."""

    for _ in range(max(0, int(steps))):
        install_override(slots, fast)
        loss = pm_ntp_loss(model, prompt_ids)
        grads = torch.autograd.grad(
            loss,
            tuple(fast.values()),
            create_graph=not first_order,
            retain_graph=not first_order,
            allow_unused=False,
        )
        if first_order:
            grads = tuple(grad.detach() for grad in grads)
        fast = {key: value - lr * grad for (key, value), grad in zip(fast.items(), grads)}
    return fast


def train_one_epoch(
    model,
    slots,
    hyper: PrimeTTTHyperNetwork,
    files: list[str],
    optimizer: AdamW,
    *,
    device: torch.device,
    inner_steps: int,
    inner_lr: float,
    first_order: bool,
) -> float:
    total = 0.0
    for path in files:
        state = torch.load(path, map_location="cpu", weights_only=False)
        prompt_ids = state["prompt_ids"].to(device).long()
        answer_ids = state["answer_ids"].to(device).long()
        full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
        answer_start = prompt_ids.size(1)

        reset_lora(slots)
        clear_override(slots)
        hidden = top_hidden_states(model, prompt_ids)
        fast = _clone_fast(hyper(hidden))
        fast = _pm_unroll(
            model,
            slots,
            prompt_ids,
            fast,
            steps=inner_steps,
            lr=inner_lr,
            first_order=first_order,
        )
        install_override(slots, fast)
        outer_loss = ac_ntp_loss(model, full_ids, answer_start)
        optimizer.zero_grad(set_to_none=True)
        outer_loss.backward()
        optimizer.step()
        clear_override(slots)
        reset_lora(slots)
        total += float(outer_loss.detach().cpu())
    return total / max(1, len(files))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-ckpt", required=True)
    parser.add_argument("--oracle-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--inner-steps", type=int, default=8)
    parser.add_argument("--inner-lr", type=float, default=1e-3)
    parser.add_argument("--outer-lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-position", type=int, default=0)
    parser.add_argument("--lora-top-layers", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--first-order", action="store_true")
    args = parser.parse_args(argv)

    files = sorted(glob.glob(str(Path(args.oracle_dir) / "ex_*.pt")))
    if not files:
        raise FileNotFoundError(f"no oracle dumps found under {args.oracle_dir}")

    model = load_base_model(args.model, max_position=args.max_position).to("cuda")
    device = input_device(model)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    slots = inject_lora(model, top_layers=args.lora_top_layers, rank=args.lora_rank)

    ckpt = torch.load(args.warmup_ckpt, map_location=device, weights_only=False)
    hyper = PrimeTTTHyperNetwork(**ckpt["manifest"]).to(device)
    hyper.load_state_dict(ckpt["state_dict"])
    hyper.train()
    optimizer = AdamW(hyper.parameters(), lr=args.outer_lr)

    history = []
    for epoch in range(args.epochs):
        loss = train_one_epoch(
            model,
            slots,
            hyper,
            files,
            optimizer,
            device=device,
            inner_steps=args.inner_steps,
            inner_lr=args.inner_lr,
            first_order=args.first_order,
        )
        history.append({"epoch": epoch, "outer_ac_ntp": loss})
        print(f"[stage3] epoch={epoch} outer_ac_ntp={loss:.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": hyper.state_dict(),
            "manifest": hyper.manifest(),
            "history": history,
            "first_order": args.first_order,
        },
        out,
    )


if __name__ == "__main__":
    main()
