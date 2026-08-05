"""Gemma runtime loading with frozen local-only assumptions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def snapshot_hash(path: str | Path) -> str:
    root = Path(path)
    digest = hashlib.sha256()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "generation_config.json"):
        target = root / name
        if target.is_file():
            digest.update(name.encode())
            digest.update(hashlib.sha256(target.read_bytes()).digest())
    value = digest.hexdigest()
    if value == hashlib.sha256().hexdigest():
        raise ValueError(f"no snapshot metadata found: {root}")
    return value


def load_runtime(model_path: str | Path, *, attention_backend: str = "sdpa") -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("formal E31 execution requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("E31 requires BF16 support")
    path = str(Path(model_path).resolve())
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
    if not tokenizer.chat_template:
        raise ValueError("tokenizer has no chat template")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither pad nor eos token")
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if attention_backend:
        kwargs["attn_implementation"] = attention_backend
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.to("cuda")
    model.eval()
    return torch, tokenizer, model
