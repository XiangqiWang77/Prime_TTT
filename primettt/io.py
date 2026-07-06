"""Prompt formatting, tokenization, decoding, and checkpoint IO."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_prompt_answer(example: dict) -> tuple[str, str]:
    answer = example.get("answer")
    if answer is None:
        raise KeyError("example is missing 'answer'")
    if "choices" in example and "question" in example:
        choices = "\n".join(str(choice) for choice in example["choices"])
        return f"Question: {example['question']}\nChoices:\n{choices}\nAnswer:", f" {answer}".strip()
    if "input" in example:
        prompt = str(example["input"]).rstrip()
        if not prompt.endswith("Answer:"):
            prompt += "\nAnswer:"
        return prompt, f" {answer}".strip()
    if "context" in example and "question" in example:
        return f"Context:\n{example['context']}\n\nQuestion: {example['question']}\nAnswer:", f" {answer}".strip()
    raise KeyError(f"unrecognized example schema: {sorted(example.keys())}")


def is_gpt_oss_tokenizer(tokenizer) -> bool:
    name = (getattr(tokenizer, "name_or_path", "") or "").lower()
    return "gpt-oss" in name or "gpt_oss" in name


def tokenize_prompt_answer(tokenizer, prompt: str, answer: str, max_prompt_tokens: int, device):
    if is_gpt_oss_tokenizer(tokenizer) and hasattr(tokenizer, "apply_chat_template"):
        user = prompt.removesuffix("Answer:").rstrip()
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            add_generation_prompt=True,
            tokenize=False,
            reasoning_effort="low",
        )
        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}, {"role": "assistant", "content": str(answer).strip()}],
            add_generation_prompt=False,
            tokenize=False,
            reasoning_effort="low",
        )
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids
        answer_start = prompt_ids.size(1)
        return full_ids.to(device), answer_start

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    answer_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False).input_ids
    if prompt_ids.size(1) > max_prompt_tokens:
        prompt_ids = prompt_ids[:, -max_prompt_tokens:]
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    return full_ids.to(device), prompt_ids.size(1)


_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|<\|start\|>|$)",
    flags=re.DOTALL,
)
_TAG_RE = re.compile(r"<\|[^|]*\|>")


def decode_generation(tokenizer, token_ids) -> str:
    if is_gpt_oss_tokenizer(tokenizer):
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
        match = _FINAL_RE.search(text)
        return (match.group(1) if match else _TAG_RE.sub("", text)).strip()
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()

