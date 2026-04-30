"""Typed config loading.

Configs live as YAML under ``configs/`` and load into frozen dataclasses so a
typo becomes an ``AttributeError`` at load time instead of a silent
misconfiguration three hours into a training run.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


def _filter_known(cls, data: dict[str, Any]) -> dict[str, Any]:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return {k: v for k, v in data.items() if k in known}


@dataclass(frozen=True)
class DrafterConfig:
    """Architecture + training hyperparameters for the semi-AR drafter."""

    target_model: str = "Qwen/Qwen3-4B"
    block_size: int = 5
    num_draft_layers: int = 3
    target_layer_ids: tuple[int, ...] = (1, 17, 33)
    markov_rank: int = 128
    markov_head_type: str = "vanilla"
    num_anchors: int = 256
    confidence_head_alpha: float = 1.0
    confidence_head_with_markov: bool = True
    ce_loss_alpha: float = 0.1
    l1_loss_alpha: float = 0.9
    loss_decay_gamma: float = 4.0
    precision_train: str = "bf16"
    torch_compile: bool = True
    hidden_size: int = 2560  # Qwen3-4B model dim; overridden from the verifier at build
    vocab_size: int = 151936

    @classmethod
    def from_yaml(cls, path: str | Path) -> DrafterConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data = _filter_known(cls, data)
        if "target_layer_ids" in data:
            data["target_layer_ids"] = tuple(data["target_layer_ids"])
        return cls(**data)


@dataclass(frozen=True)
class QuantConfig:
    """A single quantization variant (INT8 W8A8 or NF4)."""

    name: str = "int8"
    scheme: str = "w8a8"  # 'w8a8' | 'nf4'
    target: str = "drafter"  # 'drafter' | 'verifier'
    backend: str = "bitsandbytes"  # 'bitsandbytes' | 'awq' | 'gptq' | 'gguf'
    quantize_backbone: bool = True
    quantize_markov_head: bool = False
    quantize_confidence_head: bool = False  # keep the head high-precision by default
    double_quant: bool = True  # NF4 nested quantization
    compute_dtype: str = "bfloat16"

    @classmethod
    def from_yaml(cls, path: str | Path) -> QuantConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**_filter_known(cls, data))


@dataclass(frozen=True)
class PolicyConfig:
    """Verification-length policy parameters."""

    kind: str = "threshold"  # 'threshold' | 'fixed' | 'oracle'
    theta: float = 0.55
    ema_gain: float = 0.15
    min_length: int = 1
    fixed_length: int | None = None  # for the always-verify-all ablation

    @classmethod
    def from_yaml(cls, path: str | Path) -> PolicyConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(**_filter_known(cls, data))


@dataclass(frozen=True)
class BenchConfig:
    """Harness settings for a benchmark run."""

    prompt_sets: tuple[str, ...] = ("chat", "code")
    max_new_tokens: int = 256
    seeds: tuple[int, ...] = (0, 1, 2)
    warmup_prompts: int = 2
    decoding: str = "greedy"  # 'greedy' | 'stochastic'
    temperature: float = 0.0
    output_dir: str = "runs"
    precisions: tuple[str, ...] = ("fp16", "int8", "nf4")

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for key in ("prompt_sets", "seeds", "precisions"):
            if key in data and data[key] is not None:
                data[key] = tuple(data[key])
        return cls(**_filter_known(cls, data))
