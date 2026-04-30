"""A tiny order-1 Markov language model, in pure numpy.

Not part of the real stack — the real verifier is Qwen3 via HF Transformers. But
speculative decoding is a probabilistic protocol, and protocols deserve tests
that don't need an 8 GB model or a GPU. ``ToyCategoricalLM`` gives us a fully
specified "language model" over a handful of tokens whose exact next-token
distribution we know in closed form, so the exactness suite can assert
token-for-token equality and statistical unbiasedness against ground truth.

It also powers the CPU path of the demo dashboard, so the side-by-side view runs
on any machine even before the ROCm stack is stood up.
"""

from __future__ import annotations

import numpy as np


class ToyCategoricalLM:
    """Order-1 Markov model: next-token distribution depends only on the last token.

    A random, fixed transition matrix defines the model. ``temperature`` sharpens
    (< 1) or flattens (> 1) the conditionals, which lets tests dial the agreement
    between a "verifier" LM and a "drafter" LM up and down.
    """

    def __init__(self, vocab_size: int = 8, seed: int = 0, temperature: float = 1.0):
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        self.vocab_size = int(vocab_size)
        self.temperature = float(temperature)
        rng = np.random.default_rng(seed)
        logits = rng.standard_normal((vocab_size, vocab_size))
        self._probs = _softmax(logits / self.temperature, axis=1)

    @classmethod
    def perturbed(cls, base: ToyCategoricalLM, noise: float = 0.5, seed: int = 0,
                  temperature: float = 1.0) -> ToyCategoricalLM:
        """A drafter LM correlated with ``base`` — a realistic partial-agreement pair.

        Same-seed models share an argmax and accept everything; independent models
        never agree. Real drafter/verifier pairs are in between. This adds logit
        noise to ``base`` so ``noise`` dials greedy agreement (and thus accept
        rate) continuously: small noise ~ a strong drafter, large noise ~ a weak
        one. Exactness holds either way — only speed changes.
        """
        obj = cls.__new__(cls)
        obj.vocab_size = base.vocab_size
        obj.temperature = float(temperature)
        rng = np.random.default_rng(seed)
        base_logits = np.log(base._probs + 1e-9)
        logits = base_logits + noise * rng.standard_normal(base._probs.shape)
        obj._probs = _softmax(logits / obj.temperature, axis=1)
        return obj

    def dist(self, last_token: int) -> np.ndarray:
        """Next-token probability vector given the previous token."""
        return self._probs[int(last_token)].copy()

    def block_target_dist(self, prefix_last: int, block_tokens: np.ndarray) -> np.ndarray:
        """Verifier-side distributions for a drafted block.

        Given the last accepted token and the ``L`` drafted tokens, returns the
        ``[L + 1, vocab]`` block the acceptance rule expects: row ``j`` is the
        model's distribution at the slot the ``(j+1)``-th token occupies.
        """
        block_tokens = np.asarray(block_tokens, dtype=np.int64).ravel()
        context = np.concatenate([[int(prefix_last)], block_tokens])
        return np.stack([self.dist(t) for t in context])  # [L+1, V]

    def draft_block(
        self, prefix_last: int, block_size: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a block of ``block_size`` tokens autoregressively from this LM.

        Returns ``(tokens[L], draft_dist[L, vocab])`` where row ``j`` is the
        distribution ``tokens[j]`` was drawn from — exactly what
        :func:`edgespark.loop.acceptance.speculative_accept` needs.
        """
        tokens, dists = [], []
        cur = int(prefix_last)
        for _ in range(block_size):
            d = self.dist(cur)
            tok = int(rng.choice(self.vocab_size, p=d))
            tokens.append(tok)
            dists.append(d)
            cur = tok
        return np.asarray(tokens, dtype=np.int64), np.stack(dists)

    def greedy_generate(self, start_token: int, n: int) -> list[int]:
        """Reference greedy decode: the sequence the model emits on its own."""
        out, cur = [], int(start_token)
        for _ in range(n):
            cur = int(self.dist(cur).argmax())
            out.append(cur)
        return out

    def sample_generate(self, start_token: int, n: int, rng: np.random.Generator) -> list[int]:
        """Reference stochastic decode from the model alone."""
        out, cur = [], int(start_token)
        for _ in range(n):
            cur = int(rng.choice(self.vocab_size, p=self.dist(cur)))
            out.append(cur)
        return out


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
