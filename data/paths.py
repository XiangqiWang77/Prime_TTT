"""Known local NTK_TTT dataset/model paths.

These are defaults for the Roberts cluster layout.  Override them in configs
when running elsewhere.
"""

from __future__ import annotations

from pathlib import Path

NTK_TTT_ROOT = Path("/nfs/roberts/scratch/pi_mg269/da839/da839_home_offload/NTK_TTT")
PRIMETTT_ROOT = NTK_TTT_ROOT / "Prime_TTT"
MODELS = {
    "llama8b": NTK_TTT_ROOT / "Models" / "Llama-3.1-8B-Instruct",
    "llama3b": NTK_TTT_ROOT / "Models" / "Llama-3.2-3B-Instruct",
    "qwen4b": NTK_TTT_ROOT / "Models" / "Qwen3-4B-0725-Ins",
    "gptoss20b": NTK_TTT_ROOT / "Models" / "gpt-oss-20b",
    "gptoss20b_tq3": NTK_TTT_ROOT / "Models" / "gpt-oss-20b-tq3",
}
DATASETS = {
    "niah_free_train": NTK_TTT_ROOT / "archive" / "combined" / "niah_free" / "long_niah_free_train.jsonl",
    "niah_free_test": NTK_TTT_ROOT / "archive" / "combined" / "niah_free" / "long_niah_free_test.jsonl",
    "hotpotqa_ood": NTK_TTT_ROOT / "archive" / "combined" / "ood_v2" / "hotpotqa_test.jsonl",
    "2wikimhqa_ood": NTK_TTT_ROOT / "archive" / "combined" / "ood_v2" / "2wikimhqa_test.jsonl",
    "musr_ood": NTK_TTT_ROOT / "archive" / "combined" / "ood_v2" / "musr_test.jsonl",
}


def resolve_dataset(name_or_path: str) -> Path:
    return DATASETS.get(name_or_path, Path(name_or_path))


def resolve_model(name_or_path: str) -> Path:
    return MODELS.get(name_or_path, Path(name_or_path))

