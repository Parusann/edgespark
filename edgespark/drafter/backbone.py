"""Parallel block backbone.

Predicts an entire ``block_size`` block of tokens in one forward pass from the
verifier's selected-layer hidden states fused with an embedding of the accepted
prefix. "Parallel" is the whole speed story: instead of ``block_size`` sequential
verifier-sized passes, the backbone runs once, and the Markov head reintroduces
just enough intra-block dependence to keep accept rates up.

Shrunk relative to DSpark's Qwen3-4B reference: ``num_draft_layers`` 2-3 instead
of 5, fusing 3 hidden-state sources instead of 5. Small enough to sit beside a
quantized verifier in 24 GB.
"""

from __future__ import annotations


def _torch():
    import torch

    return torch


class HiddenFusion:
    """Projects each selected verifier layer to the draft width and sums them."""

    def __new__(cls, layer_ids, verifier_hidden: int, draft_hidden: int):
        torch = _torch()
        nn = torch.nn

        class _HiddenFusion(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_ids = list(layer_ids)
                self.proj = nn.ModuleDict(
                    {str(lid): nn.Linear(verifier_hidden, draft_hidden) for lid in layer_ids}
                )
                self.norm = nn.LayerNorm(draft_hidden)

            def forward(self, hidden_by_layer: dict):
                acc = None
                for lid in self.layer_ids:
                    h = self.proj[str(lid)](hidden_by_layer[lid])
                    acc = h if acc is None else acc + h
                return self.norm(acc)

        return _HiddenFusion()


class ParallelBackbone:
    def __new__(
        cls,
        verifier_hidden: int,
        vocab_size: int,
        block_size: int = 5,
        num_layers: int = 3,
        draft_hidden: int = 1024,
        num_heads: int = 8,
        target_layer_ids=(1, 17, 33),
    ):
        torch = _torch()
        nn = torch.nn

        class _ParallelBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.block_size = block_size
                self.draft_hidden = draft_hidden
                self.fusion = HiddenFusion(target_layer_ids, verifier_hidden, draft_hidden)
                self.prefix_embed = nn.Embedding(vocab_size, draft_hidden)
                # One learned query per block position; the backbone reads fused
                # context and turns each query into that position's hidden state.
                self.block_queries = nn.Parameter(torch.randn(block_size, draft_hidden) * 0.02)
                layer = nn.TransformerDecoderLayer(
                    d_model=draft_hidden, nhead=num_heads,
                    dim_feedforward=draft_hidden * 2, batch_first=True, activation="gelu",
                )
                self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
                self.lm_head = nn.Linear(draft_hidden, vocab_size, bias=False)

            def forward(self, hidden_by_layer: dict, prefix_last_token):
                fused = self.fusion(hidden_by_layer)  # [b, ctx, draft_hidden]
                prefix = self.prefix_embed(prefix_last_token).unsqueeze(1)  # [b, 1, h]
                memory = torch.cat([fused, prefix], dim=1)
                b = memory.shape[0]
                queries = self.block_queries.unsqueeze(0).expand(b, -1, -1)  # [b, block, h]
                block_hidden = self.decoder(queries, memory)  # [b, block, h]
                logits = self.lm_head(block_hidden)  # [b, block, vocab]
                return logits, block_hidden

        return _ParallelBackbone()
