"""Unified deterministic scorer for PrimeTTT predictions."""

from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path


def normalize(text: object) -> str:
    text = str(text).lower()
    text = text.translate(str.maketrans({char: " " for char in string.punctuation}))
    return " ".join(text.split())


def contains_rate(prediction: str, answer: object) -> float:
    answers = answer if isinstance(answer, list) else [answer]
    pred_norm = normalize(prediction)
    return float(any(normalize(ans) in pred_norm for ans in answers))


def token_f1(prediction: str, answer: object) -> float:
    gold = normalize(answer).split()
    pred = normalize(prediction).split()
    if not gold or not pred:
        return float(gold == pred)
    common = 0
    pred_counts = {tok: pred.count(tok) for tok in set(pred)}
    for tok in set(gold):
        common += min(gold.count(tok), pred_counts.get(tok, 0))
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(gold)
    return 2 * precision * recall / (precision + recall)


def first_mc_letter(prediction: str) -> str | None:
    match = re.search(r"\b([A-E])\b", prediction.upper())
    return match.group(1) if match else None


def mc_accuracy(prediction: str, answer: object) -> float:
    gold = str(answer).strip().upper()[:1]
    return float(first_mc_letter(prediction) == gold)


def score_record(record: dict, mode: str) -> float:
    pred = record.get("prediction", record.get("pred", ""))
    answer = record.get("answer", record.get("gold", ""))
    if mode == "contains":
        return contains_rate(pred, answer)
    if mode == "f1":
        return token_f1(pred, answer)
    if mode == "mc":
        return mc_accuracy(pred, answer)
    raise ValueError(f"unknown scorer: {mode}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--mode", choices=["contains", "f1", "mc"], default="contains")
    args = parser.parse_args(argv)
    scores = []
    with Path(args.predictions).open() as handle:
        for line in handle:
            if line.strip():
                scores.append(score_record(json.loads(line), args.mode))
    print(json.dumps({"n": len(scores), "score": sum(scores) / max(1, len(scores))}, indent=2))


if __name__ == "__main__":
    main()
