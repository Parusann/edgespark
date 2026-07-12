"""Markov head: cheap intra-block token dependence.

The parallel backbone emits a whole block at once and is therefore blind to what
it just proposed one slot earlier. The Markov head fixes that without paying for
autoregression: it adds a *low-rank* logit bias to position ``j`` computed from
the token drafted at position ``j-1``. Rank ``markov_rank`` (128 here, down from
DSpark's 256) trades a little expressiveness for footprint and speed.

The bias is factored as ``E[prev] @ P`` with ``E: [vocab, rank]`` and
``P: [rank, vocab]``, so it costs one embedding lookup and one ``rank x vocab``
matmul per position, negligible next to the backbone.
"""

from __future__ import annotations


def _torch():
    import torch

    return torch


class MarkovHead:
    """Constructed lazily so importing the package never requires torch."""

    def __new__(cls, vocab_size: int, rank: int = 128, head_type: str = "vanilla"):
        torch = _torch()
        nn = torch.nn

        class _MarkovHead(nn.Module):
            def __init__(self):
                super().__init__()
                if head_type != "vanilla":
                    raise ValueError(f"only 'vanilla' markov head is implemented, got {head_type!r}")
                self.vocab_size = vocab_size
                self.rank = rank
                # Left factor: previous-token -> rank code. Right factor: rank -> vocab bias.
                self.prev_embed = nn.Embedding(vocab_size, rank)
                self.to_bias = nn.Linear(rank, vocab_size, bias=False)
                # Start as a no-op so early training is dominated by the backbone.
                nn.init.zeros_(self.to_bias.weight)

            def forward(self, prev_tokens):
                """prev_tokens: [batch, block]. Returns bias [batch, block, vocab]."""
                code = self.prev_embed(prev_tokens)  # [b, block, rank]
                return self.to_bias(code)  # [b, block, vocab]

        return _MarkovHead()
