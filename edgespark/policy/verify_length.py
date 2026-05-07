"""How many drafted tokens are worth verifying on a single stream.

DSpark's datacenter scheduler weighs confidence against fleet load. At batch
size 1 there is no fleet, and the whole thing collapses to one question:

    given calibrated per-position survival a_1..a_block, choose a verification
    length ell in [0, block_size].

The trade-off is real even alone on the GPU. Verifier cost grows with ``ell``,
while expected accepted length grows with diminishing returns (a token deep in
the block only helps if *every* earlier token also survived). Verifying a tail
that will almost surely be rejected just burns ``T_verify``.

Crucially, ``ell`` only bounds *how many* tokens are submitted for checking. The
verifier still applies the exact accept/reject rule to those tokens, so the
output is unchanged (spec section 5). This module can be as wrong as it likes
about ``ell`` and the worst outcome is lost speed, never a wrong token.

Two policies:

* :class:`ThresholdPolicy` — verify the longest prefix whose cumulative predicted
  survival stays above ``theta``. One interpretable knob, tuned on held-out data.
* :func:`optimal_length` — the cost-aware optimum given a verifier timing model;
  used offline to *choose* a good ``theta`` and as the ceiling the threshold rule
  is measured against.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def _survival(confidence_profile: np.ndarray) -> np.ndarray:
    """Cumulative survival S_j = prod_{k<=j} a_k for each position j."""
    a = np.clip(np.asarray(confidence_profile, dtype=np.float64).ravel(), 0.0, 1.0)
    return np.cumprod(a)


def expected_accepted_length(confidence_profile: np.ndarray, ell: int) -> float:
    """E[accepted(ell)] = 1 + sum_{j=1..ell} prod_{k<=j} a_k  (spec Appendix B).

    The leading ``1`` is the guaranteed verifier token (bonus or correction) that
    every round emits regardless of how the drafts fare.
    """
    a = np.asarray(confidence_profile, dtype=np.float64).ravel()
    if not 0 <= ell <= a.shape[0]:
        raise ValueError(f"ell={ell} out of range [0, {a.shape[0]}]")
    if ell == 0:
        return 1.0
    return float(1.0 + _survival(a[:ell]).sum())


def optimal_length(
    confidence_profile: np.ndarray,
    t_draft: float,
    t_verify: Callable[[int], float],
) -> int:
    """Cost-aware optimum: argmax over ell of throughput per unit wall-clock.

    Maximises ``E[accepted(ell)] / (t_draft + t_verify(ell))``. ``t_verify`` maps a
    verification length to a predicted verifier time (measured offline; often
    close to affine in ``ell`` on one GPU). Returns the best ``ell`` in
    ``[1, block_size]`` — verifying nothing is never throughput-optimal when any
    draft has positive survival.
    """
    a = np.asarray(confidence_profile, dtype=np.float64).ravel()
    block = a.shape[0]
    best_ell, best_rate = 1, -np.inf
    for ell in range(1, block + 1):
        denom = t_draft + t_verify(ell)
        if denom <= 0:
            continue
        rate = expected_accepted_length(a, ell) / denom
        if rate > best_rate:
            best_ell, best_rate = ell, rate
    return best_ell


class VerifyLengthPolicy:
    """Interface every policy honours (spec section 15)."""

    def choose_length(self, confidence_profile, recent_accept_ema: float) -> int:  # noqa: D401
        raise NotImplementedError


@dataclass
class ThresholdPolicy(VerifyLengthPolicy):
    """Threshold on cumulative survival.

    ``theta`` is the one tuned knob. ``ema_gain`` gently lengthens verification
    when the recent stream has been accepting well and shortens it when accepts
    have dried up — a cheap, exactness-neutral adaptation to locally easy or hard
    text. Set ``ema_gain=0`` for the pure static rule used in the ablation.
    """

    theta: float = 0.5
    ema_gain: float = 0.0
    min_length: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.theta <= 1.0:
            raise ValueError("theta must be in (0, 1]")

    def choose_length(self, confidence_profile, recent_accept_ema: float = 1.0) -> int:
        a = np.asarray(confidence_profile, dtype=np.float64).ravel()
        block = a.shape[0]
        if block == 0:
            return 0
        survival = _survival(a)
        # Nudge the effective threshold by recent stream behaviour, then clamp so
        # the adaptation can never invert the rule.
        theta_eff = float(np.clip(self.theta - self.ema_gain * (recent_accept_ema - 0.5), 0.05, 0.999))
        keep = np.nonzero(survival >= theta_eff)[0]
        ell = int(keep[-1] + 1) if keep.size else 0
        ell = max(ell, self.min_length)
        return int(np.clip(ell, 0, block))
