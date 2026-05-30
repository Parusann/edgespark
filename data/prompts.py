"""Evaluation prompt sets (spec sections 10, 12).

Small, fixed, single-stream prompt sets for the benchmark harness. The built-in
sets are enough to drive a reproducible run out of the box; ``load_prompt_sets``
also reads newline-delimited files from ``data/datasets/`` when you want the
DeepSpec eval subsets (humaneval / mbpp / gsm8k / mt-bench / alpaca).

Kept deliberately code-heavy: deterministic code raises speculative accept rates
(spec section 17 risk mitigation).
"""

from __future__ import annotations

from pathlib import Path

_CODE = [
    "Write a Python function that returns the nth Fibonacci number iteratively.",
    "Implement binary search over a sorted list and return the index or -1.",
    "Given a list of integers, return the two numbers that sum to a target.",
    "Write a function to check whether a string is a valid palindrome, ignoring case and punctuation.",
    "Implement quicksort in Python without using the standard library sort.",
    "Parse a CSV string into a list of dictionaries keyed by the header row.",
    "Write a decorator that memoizes a single-argument pure function.",
    "Implement a least-recently-used cache with O(1) get and put.",
]

_CHAT = [
    "Explain the difference between speculative decoding and beam search in two paragraphs.",
    "Summarise why quantization can hurt model calibration more than accuracy.",
    "What are the trade-offs between INT8 and 4-bit weight quantization on consumer GPUs?",
    "Describe how a KV cache reduces the cost of autoregressive generation.",
    "Give three reasons a small draft model might have a low acceptance rate.",
    "Explain expected calibration error to someone who knows basic statistics.",
    "What is the intuition behind rejection sampling in speculative decoding?",
    "Describe when verifying fewer drafted tokens can increase throughput.",
]

_BUILTIN = {"code": _CODE, "chat": _CHAT}


def load_prompt_sets(names) -> dict[str, list[str]]:
    """Return ``{set_name: [prompt, ...]}`` for the requested sets."""
    out: dict[str, list[str]] = {}
    for name in names:
        path = Path("data/datasets") / f"{name}.txt"
        if path.exists():
            out[name] = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        elif name in _BUILTIN:
            out[name] = list(_BUILTIN[name])
        else:
            raise KeyError(f"unknown prompt set {name!r}; add data/datasets/{name}.txt")
    return out
