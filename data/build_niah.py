"""Synthetic needle-in-a-haystack JSONL builder."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def build_example(index: int, *, context_tokens: int, seed: int) -> dict:
    rng = random.Random(seed + index)
    key = f"NEEDLE-{index:06d}"
    words = [f"tok{rng.randrange(10000)}" for _ in range(context_tokens)]
    pos = rng.randrange(len(words) + 1)
    words.insert(pos, f"The secret key is {key}.")
    return {
        "id": f"niah-{index:06d}",
        "context": " ".join(words),
        "question": "What is the secret key?",
        "answer": key,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("w") as handle:
        for i in range(args.n):
            handle.write(json.dumps(build_example(i, context_tokens=args.context_tokens, seed=args.seed)) + "\n")


if __name__ == "__main__":
    main()

