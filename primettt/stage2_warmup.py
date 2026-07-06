"""Stage 2: supervised hypernetwork warm-up on oracle Delta B."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch
from torch.optim import AdamW

from .features import top_hidden_states
from .hypernetwork import PrimeTTTHyperNetwork
from .lora import inject_lora, reset_lora, slot_info
from .model_loader import input_device, load_base_model


def load_oracle_files(oracle_dir: str):
    files = sorted(glob.glob(str(Path(oracle_dir) / "ex_*.pt")))
    if not files:
        raise FileNotFoundError(f"no oracle dumps found under {oracle_dir}")
    return files


def compute_B_base(files: list[str]) -> dict[str, torch.Tensor]:
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    for path in files:
        state = torch.load(path, map_location="cpu", weights_only=False)
        for key, value in state["b_oracle"].items():
            tensor = value.float()
            sums[key] = sums.get(key, torch.zeros_like(tensor)) + tensor
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in sums}


def target_loss(pred: dict[str, torch.Tensor], oracle: dict[str, torch.Tensor], base: dict[str, torch.Tensor], shrink: float):
    loss = 0.0
    for key, value in pred.items():
        target_delta = oracle[key].to(value.device).float() - base[key].to(value.device).float()
        pred_delta = value.float() - base[key].to(value.device).float()
        loss = loss + torch.nn.functional.l1_loss(pred_delta, target_delta)
        loss = loss + shrink * pred_delta.pow(2).mean()
    return loss / max(1, len(pred))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--shrink", type=float, default=1e-4)
    parser.add_argument("--max-position", type=int, default=0)
    parser.add_argument("--lora-top-layers", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=8)
    args = parser.parse_args(argv)

    files = load_oracle_files(args.oracle_dir)
    model = load_base_model(args.model, max_position=args.max_position).to("cuda")
    device = input_device(model)
    slots = inject_lora(model, top_layers=args.lora_top_layers, rank=args.lora_rank)
    slots_meta = slot_info(slots)
    hidden_size = int(getattr(model.config, "hidden_size"))
    hyper = PrimeTTTHyperNetwork(hidden_size, slots_meta, rank=args.lora_rank).to(device)
    base = compute_B_base(files)
    hyper.set_B_base(base)
    optimizer = AdamW(hyper.parameters(), lr=args.lr)
    for _ in range(args.epochs):
        for path in files:
            state = torch.load(path, map_location="cpu", weights_only=False)
            prompt_ids = state["prompt_ids"].to(device).long()
            reset_lora(slots)
            hidden = top_hidden_states(model, prompt_ids)
            pred = hyper(hidden)
            loss = target_loss(pred, state["b_oracle"], base, args.shrink)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": hyper.state_dict(), "manifest": hyper.manifest(), "B_base": base}, out)


if __name__ == "__main__":
    main()
