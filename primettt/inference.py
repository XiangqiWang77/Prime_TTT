"""Test-time PrimeTTT inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .features import top_hidden_states
from .hypernetwork import PrimeTTTHyperNetwork
from .inner_loop import polish_pm_ntp
from .io import decode_generation, format_prompt_answer, read_jsonl, tokenize_prompt_answer
from .lora import inject_lora, install_B, reset_lora
from .model_loader import input_device, load_base_model, load_tokenizer


@torch.no_grad()
def generate(model, tokenizer, prompt_ids, max_new_tokens: int) -> str:
    model.eval()
    out = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        use_cache=True,
    )
    return decode_generation(tokenizer, out[0, prompt_ids.size(1) :])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-position", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--inner-steps", type=int, default=8)
    parser.add_argument("--inner-lr", type=float, default=1e-3)
    parser.add_argument("--lora-top-layers", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=8)
    args = parser.parse_args(argv)

    torch.manual_seed(0)
    tokenizer = load_tokenizer(args.model)
    model = load_base_model(args.model, max_position=args.max_position).to("cuda")
    device = input_device(model)
    slots = inject_lora(model, top_layers=args.lora_top_layers, rank=args.lora_rank)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    hyper = PrimeTTTHyperNetwork(**ckpt["manifest"]).to(device)
    hyper.load_state_dict(ckpt["state_dict"])
    hyper.eval()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for index, example in enumerate(read_jsonl(args.data)):
            prompt, answer = format_prompt_answer(example)
            full_ids, answer_start = tokenize_prompt_answer(
                tokenizer, prompt, answer, args.max_prompt_tokens, device
            )
            prompt_ids = full_ids[:, :answer_start]
            reset_lora(slots)
            hidden = top_hidden_states(model, prompt_ids)
            predicted = hyper(hidden)
            install_B(slots, predicted)
            trace = polish_pm_ntp(
                model, slots, prompt_ids, steps=args.inner_steps, lr=args.inner_lr
            )
            pred = generate(model, tokenizer, prompt_ids, args.max_new_tokens)
            reset_lora(slots)
            handle.write(
                json.dumps(
                    {
                        "id": example.get("id", str(index)),
                        "prediction": pred,
                        "answer": example.get("answer"),
                        "inner_trace": trace.losses,
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()

