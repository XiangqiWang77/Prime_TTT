"""Model loading policy shared by all PrimeTTT runs."""

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def apply_yarn_config(config, max_position: int = 0, yarn_factor: float | None = None):
    native = int(getattr(config, "max_position_embeddings", 0) or 0)
    if not max_position:
        return config
    if native and max_position <= native:
        return config

    model_type = getattr(config, "model_type", "") or ""
    if model_type in {"llama", "mistral", "qwen2", "qwen3"}:
        factor = float(yarn_factor or (float(max_position) / float(native or max_position)))
        config.rope_scaling = {
            "rope_type": "yarn",
            "factor": factor,
            "original_max_position_embeddings": native or max_position,
        }
    config.max_position_embeddings = int(max_position)
    return config


def load_base_model(
    model_name_or_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    max_position: int = 0,
    yarn_factor: float | None = None,
    device_map: str | None = None,
    flash_attention_2: bool = False,
):
    config = AutoConfig.from_pretrained(model_name_or_path)
    config = apply_yarn_config(config, max_position=max_position, yarn_factor=yarn_factor)
    model_type = getattr(config, "model_type", "") or ""
    attn_impl = "flash_attention_2" if flash_attention_2 else ("eager" if model_type == "gpt_oss" else "sdpa")
    kwargs = {"config": config, "torch_dtype": dtype, "attn_implementation": attn_impl}
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def input_device(model, fallback: str = "cuda") -> torch.device:
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        for key in ("model.embed_tokens", "embed_tokens"):
            if key in device_map:
                value = device_map[key]
                return torch.device(f"cuda:{value}" if isinstance(value, int) else value)
    return torch.device(fallback)

