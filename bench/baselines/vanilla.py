"""Vanilla PyTorch decode baseline (spec section 12, baseline 1).

The same verifier EdgeSpark uses, decoded one token per forward pass with no
drafter. Kept in-process (rather than only via llama.cpp) so the baseline runs on
the *identical* model object and precision EdgeSpark runs on — the fairest possible
comparison, same machine / same verifier / same quality budget.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VanillaResult:
    tok_s: float
    tokens: int
    seconds: float


def decode(verifier, prompt: str, max_new_tokens: int = 256) -> VanillaResult:
    import time

    import torch

    seq = verifier.encode(prompt)
    _, _, kv = verifier.forward_with_hidden(seq)
    t0 = time.perf_counter()
    for _ in range(max_new_tokens):
        logits, _, kv = verifier.forward_with_hidden(seq[:, -1:], past_key_values=kv)
        nxt = logits[0, -1].argmax().view(1, 1)
        seq = torch.cat([seq, nxt], dim=1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return VanillaResult(tok_s=max_new_tokens / dt, tokens=max_new_tokens, seconds=dt)
