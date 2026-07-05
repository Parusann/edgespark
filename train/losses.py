"""Drafter training losses (spec sections 9.2, 15 config).

Three terms, mixed per the DSpark recipe:

* **token cross-entropy** (``ce_loss_alpha`` ~ 0.1) — the drafter's block logits
  against the verifier's tokens.
* **hidden-state L1 regression** (``l1_loss_alpha`` ~ 0.9) — the dominant term:
  match the verifier's hidden state at each block position (EAGLE-style feature
  distillation). Getting the *representation* right is what makes the cheap head
  propose acceptable tokens.
* **confidence BCE** (scaled by ``confidence_head_alpha``) — the confidence head
  against the actual accept label (did the drafter's argmax match the target?).

All three are weighted by a positional decay ``exp(-j / loss_decay_gamma)`` so
early block positions — the ones most likely to be accepted and thus to matter —
dominate the gradient.
"""

from __future__ import annotations


def positional_weights(block_size, gamma, device=None):
    """Normalised ``exp(-j / gamma)`` weights over block positions."""
    import torch

    j = torch.arange(block_size, dtype=torch.float32, device=device)
    w = torch.exp(-j / gamma)
    return w / w.sum()


def drafter_loss(
    block_logits,          # [b, block, vocab]  drafter proposals
    block_hidden,          # [b, block, hidden]  drafter block hidden states
    target_tokens,         # [b, block]          verifier tokens at those positions
    target_hidden,         # [b, block, hidden]  verifier hidden states (regression target)
    confidence_logit,      # [b, block]          confidence head pre-sigmoid
    config,
):
    """Return ``(total, parts_dict)``. ``parts_dict`` is for TensorBoard logging."""
    import torch
    import torch.nn.functional as F

    b, block, vocab = block_logits.shape
    w = positional_weights(block, config.loss_decay_gamma, block_logits.device)  # [block]

    # token cross-entropy, per position, positionally weighted
    ce = F.cross_entropy(
        block_logits.reshape(-1, vocab), target_tokens.reshape(-1), reduction="none"
    ).reshape(b, block)
    ce = (ce * w).sum(dim=1).mean()

    # hidden-state L1 regression (dominant term). Cast the target to the
    # prediction dtype so the term is well-defined under bf16 autocast.
    l1 = F.l1_loss(
        block_hidden, target_hidden.to(block_hidden.dtype), reduction="none"
    ).mean(dim=-1)  # [b, block]
    l1 = (l1 * w).sum(dim=1).mean()

    # confidence: BCE against the true accept label (argmax match)
    with torch.no_grad():
        accept = (block_logits.argmax(dim=-1) == target_tokens).float()  # [b, block]
    conf = F.binary_cross_entropy_with_logits(
        confidence_logit, accept.to(confidence_logit.dtype), reduction="none"
    )
    conf = (conf * w).sum(dim=1).mean()

    total = config.ce_loss_alpha * ce + config.l1_loss_alpha * l1 + config.confidence_head_alpha * conf
    return total, {
        "ce": float(ce.detach()), "l1": float(l1.detach()),
        "confidence": float(conf.detach()), "total": float(total.detach()),
    }
