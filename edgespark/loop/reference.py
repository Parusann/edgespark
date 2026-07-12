"""A numpy reference speculative-decoding loop.

This is the executable specification of what ``edgespark.loop.generate`` (the
torch/ROCm loop) does, minus the neural machinery. It drives any pair of objects
that expose the toy-LM interface (``dist`` / ``block_target_dist`` /
``draft_block``), so the exactness suite can run the *entire* protocol, draft,
choose ``ell``, verify, accept, correct, on the CPU and check it against the
verifier decoding alone.

The key property it exists to demonstrate: for greedy decoding the output is
identical to the verifier's own greedy output *for every verification-length
policy*, because ``ell`` only bounds how many drafts are checked, never how they
are judged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from edgespark.loop.acceptance import verify_block


@dataclass
class StepRecord:
    """Per-round telemetry, mirroring the JSON-lines record in spec section 15."""

    step: int
    ell: int
    n_accepted: int
    tau: int
    conf_profile: list[float]


@dataclass
class DecodeResult:
    tokens: list[int]
    steps: list[StepRecord] = field(default_factory=list)

    @property
    def mean_accepted(self) -> float:
        acc = [s.n_accepted for s in self.steps]
        return float(np.mean(acc)) if acc else 0.0

    @property
    def mean_tau(self) -> float:
        taus = [s.tau for s in self.steps]
        return float(np.mean(taus)) if taus else 0.0


def _greedy_draft(drafter, last: int, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    tokens, dists = [], []
    cur = int(last)
    for _ in range(block_size):
        d = drafter.dist(cur)
        cur = int(d.argmax())
        tokens.append(cur)
        dists.append(d)
    return np.asarray(tokens, dtype=np.int64), np.stack(dists)


def speculative_decode(
    verifier,
    drafter,
    start_token: int,
    n_tokens: int,
    *,
    block_size: int = 5,
    mode: str = "greedy",
    policy=None,
    rng: np.random.Generator | None = None,
) -> DecodeResult:
    """Run the loop until ``n_tokens`` have been emitted.

    ``policy`` is an optional :class:`~edgespark.policy.verify_length.VerifyLengthPolicy`.
    When ``None`` the loop verifies the whole block (the always-verify-all
    baseline). The confidence profile handed to the policy is the drafter's own
    probability for each token it proposed, a stand-in for the confidence head's
    ``a_j`` that keeps the reference loop self-contained.
    """
    if mode == "stochastic" and rng is None:
        raise ValueError("stochastic mode needs an rng")

    out: list[int] = []
    steps: list[StepRecord] = []
    last = int(start_token)
    recent_ema = 1.0
    step = 0

    while len(out) < n_tokens:
        if mode == "stochastic":
            draft_tokens, draft_dist = drafter.draft_block(last, block_size, rng)
        else:
            draft_tokens, draft_dist = _greedy_draft(drafter, last, block_size)

        conf = draft_dist[np.arange(block_size), draft_tokens].astype(np.float64)
        ell = block_size if policy is None else int(policy.choose_length(conf, recent_ema))
        ell = int(np.clip(ell, 0, block_size))

        target = verifier.block_target_dist(last, draft_tokens)
        result = verify_block(
            target, draft_tokens, ell, mode=mode, draft_dist=draft_dist, rng=rng
        )

        out.extend(result.tokens)
        last = result.tokens[-1]
        if ell > 0:
            recent_ema = 0.9 * recent_ema + 0.1 * (result.n_accepted / ell)
        steps.append(
            StepRecord(
                step=step,
                ell=ell,
                n_accepted=result.n_accepted,
                tau=result.tau,
                conf_profile=[float(c) for c in conf[:ell]],
            )
        )
        step += 1

    return DecodeResult(tokens=out[:n_tokens], steps=steps)
