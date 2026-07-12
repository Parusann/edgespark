"""The assembled semi-autoregressive drafter (spec sections 9.2, 15).

Wires the three pieces together and exposes the interface the inference loop and
policy expect:

    draft(hidden_states, accepted_prefix) -> DraftBlock(tokens, confidence, dist)

``confidence`` is the *calibrated* ``a_j``, if a recalibrator has been attached
(``attach_recalibrator``), the raw confidence-head sigmoid is passed through it,
so everything downstream (the verification-length policy in particular) consumes
calibrated survival probabilities. Without a recalibrator it returns the raw
head output, which is exactly the miscalibrated signal the study measures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edgespark.drafter.backbone import ParallelBackbone
from edgespark.drafter.confidence_head import ConfidenceHead
from edgespark.drafter.markov_head import MarkovHead


@dataclass
class DraftBlock:
    tokens: np.ndarray  # [block] proposed token ids
    confidence: np.ndarray  # [block] calibrated acceptance probs a_j
    dist: np.ndarray  # [block, vocab] drafter distributions (for stochastic verify)
    raw_confidence: np.ndarray  # [block] uncalibrated head output, for logging/study


def _torch():
    import torch

    return torch


class EdgeSparkDrafter:
    """Lazily-built torch module; import-safe without torch installed."""

    def __new__(cls, config, verifier_hidden: int | None = None):
        torch = _torch()
        nn = torch.nn

        vhidden = verifier_hidden or config.hidden_size
        draft_hidden = min(1024, vhidden)

        class _EdgeSparkDrafter(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = config
                self.block_size = config.block_size
                self.backbone = ParallelBackbone(
                    verifier_hidden=vhidden,
                    vocab_size=config.vocab_size,
                    block_size=config.block_size,
                    num_layers=config.num_draft_layers,
                    draft_hidden=draft_hidden,
                    target_layer_ids=config.target_layer_ids,
                )
                self.markov = MarkovHead(
                    vocab_size=config.vocab_size,
                    rank=config.markov_rank,
                    head_type=config.markov_head_type,
                )
                self.confidence = ConfidenceHead(
                    hidden_size=draft_hidden,
                    alpha=config.confidence_head_alpha,
                    with_markov=config.confidence_head_with_markov,
                    markov_feat_size=draft_hidden,
                )
                # Feature-regression head. The L1 distillation target is the
                # verifier's Hv-wide hidden state, but the shrunk backbone works in
                # draft_hidden (<= 1024). Project the block hidden up to the
                # verifier width so drafter_loss can regress it onto target_hidden
                # (EAGLE-style feature distillation) even when draft_hidden < Hv.
                self.regress = nn.Linear(draft_hidden, vhidden)
                # Markov features for the confidence head: reuse the backbone
                # hidden shifted by one position (what "the previous slot" saw).
                self._recalibrator = None

            # -- calibration plumbing -------------------------------------
            def attach_recalibrator(self, recalibrator) -> None:
                """Attach a fitted TemperatureScaler / PlattScaler (or None)."""
                self._recalibrator = recalibrator

            def _calibrate(self, raw_conf: np.ndarray) -> np.ndarray:
                if self._recalibrator is None:
                    return raw_conf
                return self._recalibrator.transform(raw_conf)

            # -- forward / draft ------------------------------------------
            def forward(self, hidden_by_layer, prefix_last_token):
                logits, block_hidden = self.backbone(hidden_by_layer, prefix_last_token)
                # Greedy tokens from the backbone drive the Markov bias for the
                # *next* slot (teacher-forced with argmax at inference).
                prelim = logits.argmax(dim=-1)  # [b, block]
                prev = torch.roll(prelim, shifts=1, dims=1)
                prev[:, 0] = prefix_last_token
                bias = self.markov(prev)  # [b, block, vocab]
                logits = logits + bias
                markov_feat = block_hidden  # confidence conditioned on the same hidden
                conf_logit = self.confidence(block_hidden, markov_feat, return_logits=True)
                # Verifier-width feature prediction for the L1 regression term in
                # drafter_loss. Inference (draft) ignores it; training consumes it.
                block_feature = self.regress(block_hidden)  # [b, block, vhidden]
                return logits, conf_logit, block_feature

            @torch.no_grad()
            def draft(self, hidden_by_layer, accepted_prefix) -> DraftBlock:
                prefix_last = accepted_prefix[:, -1] if accepted_prefix.dim() == 2 else accepted_prefix
                logits, conf_logit, _ = self.forward(hidden_by_layer, prefix_last)
                probs = torch.softmax(logits.float(), dim=-1)[0]  # [block, vocab]
                tokens = probs.argmax(dim=-1)  # greedy proposal
                raw_conf = torch.sigmoid(conf_logit)[0]  # [block]

                tokens_np = tokens.cpu().numpy()
                raw_np = raw_conf.cpu().numpy()
                dist_np = probs.cpu().numpy()
                return DraftBlock(
                    tokens=tokens_np,
                    confidence=self._calibrate(raw_np),
                    dist=dist_np,
                    raw_confidence=raw_np,
                )

        return _EdgeSparkDrafter()
