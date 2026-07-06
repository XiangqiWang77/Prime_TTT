"""Stage 1: per-example oracle fitting."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.optim import Adam

from .io import format_prompt_answer, read_jsonl, tokenize_prompt_answer
from .lora import inject_lora, lora_b_parameters, reset_lora, snapshot_B, restore_B, slot_info
from .losses import ac_ntp_loss
from .model_loader import input_device, load_base_model, load_tokenizer


def fit_oracle(model, tokenizer, slots, example, *, steps: int, lr: float, max_prompt_tokens: int, device):
    prompt, answer = format_prompt_answer(example)
    full_ids, answer_start = tokenize_prompt_answer(tokenizer, prompt, answer, max_prompt_tokens, device)
    reset_lora(slots)
    best_snapshot = snapshot_B(slots)
    with torch.no_grad():
        init = float(ac_ntp_loss(model, full_ids, answer_start).item())
    best = init
    trace = [init]
    optimizer = Adam(lora_b_parameters(slots), lr=lr)
    for _ in range(max(0, steps)):
        loss = ac_ntp_loss(model, full_ids, answer_start)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_b_parameters(slots), 1.0)
        optimizer.step()
        with torch.no_grad():
            current = float(ac_ntp_loss(model, full_ids, answer_start).item())
        trace.append(current)
        if current < best:
            best = current
            best_snapshot = snapshot_B(slots)
    restore_B(slots, best_snapshot)
    return {
        "id": example.get("id"),
        "prompt_ids": full_ids[:, :answer_start].cpu().to(torch.int32),
        "answer_ids": full_ids[:, answer_start:].cpu().to(torch.int32),
        "init_loss": init,
        "final_loss": best,
        "trace": trace,
        "b_oracle": {key: slot.B.detach().cpu().to(torch.bfloat16) for key, slot in slots},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-position", type=int, default=0)
    parser.add_argument("--lora-top-layers", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    args = parser.parse_args(argv)

    tokenizer = load_tokenizer(args.model)
    model = load_base_model(args.model, max_position=args.max_position).to("cuda")
    device = input_device(model)
    slots = inject_lora(model, args.lora_top_layers, args.lora_rank, args.lora_alpha)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    examples = read_jsonl(args.data)
    torch.save({"slot_info": slot_info(slots), "rank": args.lora_rank}, out_dir / "manifest.pt")
    for index, example in enumerate(examples):
        state = fit_oracle(
            model,
            tokenizer,
            slots,
            example,
            steps=args.steps,
            lr=args.lr,
            max_prompt_tokens=args.max_prompt_tokens,
            device=device,
        )
        tmp = out_dir / f"ex_{index:06d}.pt.tmp"
        final = out_dir / f"ex_{index:06d}.pt"
        torch.save(state, tmp)
        os.replace(tmp, final)


if __name__ == "__main__":
    main()

