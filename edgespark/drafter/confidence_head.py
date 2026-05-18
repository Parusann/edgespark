"""Confidence head: predicted per-position acceptance probability ``a_j``.

This is the component the whole calibration study is about (spec section 9.4).
It reads each block position's backbone hidden state — optionally concatenated
with the Markov head's contribution, since the token that actually gets proposed
depends on the Markov bias — and emits a single logit per position. A sigmoid
turns that into ``a_j``, the model's estimate of "will the verifier keep this
token?".

We deliberately keep this head small and, by default, *unquantized*: it is
exactly the thing that miscalibrates under quantization, so making it the last
thing we quantize isolates the effect cleanly.
"""

from __future__ import annotations


def _torch():
    import torch

    return torch


class ConfidenceHead:
    def __new__(
        cls,
        hidden_size: int,
        alpha: float = 1.0,
        with_markov: bool = True,
        markov_feat_size: int | None = None,
    ):
        torch = _torch()
        nn = torch.nn

        in_dim = hidden_size + (markov_feat_size or hidden_size if with_markov else 0)
        if not with_markov:
            in_dim = hidden_size

        class _ConfidenceHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = float(alpha)
                self.with_markov = with_markov
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden_size // 2),
                    nn.GELU(),
                    nn.Linear(hidden_size // 2, 1),
                )

            def forward(self, block_hidden, markov_feat=None, return_logits: bool = False):
                """block_hidden: [b, block, hidden]. Returns a_j in (0,1) [b, block]."""
                if self.with_markov:
                    if markov_feat is None:
                        raise ValueError("with_markov=True requires markov_feat")
                    x = torch.cat([block_hidden, markov_feat], dim=-1)
                else:
                    x = block_hidden
                logit = self.net(x).squeeze(-1) * self.alpha  # [b, block]
                if return_logits:
                    return logit
                return torch.sigmoid(logit)

        return _ConfidenceHead()
