"""Exact speculative-decoding acceptance.

This module is the load-bearing wall of the whole project. Everything else, the
drafter, the quantization, the confidence head, the verification-length policy,
is allowed to be approximate, heuristic, or quantized to death. This file is not.
It implements the acceptance rule that makes EdgeSpark *lossless with respect to
the deployed verifier* (spec section 5).

Two decoding modes, two guarantees:

* **Greedy** (temperature 0). The emitted sequence is *token-for-token identical*
  to running the verifier autoregressively on its own. We can and do assert this
  in the test suite.

* **Stochastic** (temperature > 0). Standard speculative sampling
  (Leviathan et al. 2023; Chen et al. 2023). The emitted sequence is drawn from a
  distribution *identical* to sampling directly from the verifier. It is **not**
  token-for-token identical to a naive sampler under a shared seed, the two
  consume randomness differently, so the correct test is a statistical one
  (unbiasedness over many samples), which ``tests/test_exactness.py`` performs.

The functions here operate on plain ``numpy`` arrays of probabilities so the rule
can be reasoned about, unit-tested, and audited without a GPU or torch in the
loop. The torch inference loop (``edgespark.loop.generate``) converts verifier
logits to a probability block and hands them here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Below this we treat a drafter probability as effectively zero and accept
# unconditionally rather than dividing by it. Chosen well under any value a
# softmax over a realistic vocabulary produces for a token that was actually
# sampled, but comfortably above fp32 round-off.
_EPS = 1e-12


@dataclass
class AcceptResult:
    """Outcome of verifying one drafted block.

    Attributes
    ----------
    tokens:
        The tokens EdgeSpark emits for this round. Always contains at least one
        token: on a rejection it ends with the verifier's corrected token, and
        on a full-block acceptance it ends with the verifier's bonus token. So
        ``len(tokens)`` ranges over ``[1, ell + 1]``.
    n_accepted:
        How many *drafted* tokens were accepted, i.e. ``tau`` minus the bonus.
        Ranges over ``[0, ell]``. This is the quantity the accept-rate metrics
        and the verification-length policy care about.
    bonus:
        ``True`` when the final token is a fresh verifier token (bonus on full
        acceptance, or correction on rejection) rather than a drafted one.
    """

    tokens: list[int]
    n_accepted: int
    bonus: bool

    @property
    def tau(self) -> int:
        """Total tokens produced this round (accepted drafts + the trailing one)."""
        return len(self.tokens)


def _as_prob_block(dist: np.ndarray) -> np.ndarray:
    dist = np.asarray(dist, dtype=np.float64)
    if dist.ndim != 2:
        raise ValueError(f"expected a [positions, vocab] block, got shape {dist.shape}")
    return dist


def greedy_accept(target_dist: np.ndarray, draft_tokens: np.ndarray) -> AcceptResult:
    """Greedy (argmax) acceptance.

    Parameters
    ----------
    target_dist:
        ``[ell + 1, vocab]`` verifier distributions. Row ``j`` for ``j < ell`` is
        the verifier's distribution for the position occupied by ``draft_tokens[j]``;
        row ``ell`` is the bonus position that only matters on full acceptance.
    draft_tokens:
        ``[ell]`` integer tokens proposed by the drafter.

    The verifier's greedy choice is ``argmax`` of each row. We accept the longest
    prefix of ``draft_tokens`` that matches, then append exactly one verifier
    token: the correction at the first mismatch, or the bonus if all matched.
    """
    target_dist = _as_prob_block(target_dist)
    draft_tokens = np.asarray(draft_tokens, dtype=np.int64).ravel()
    ell = draft_tokens.shape[0]
    if target_dist.shape[0] != ell + 1:
        raise ValueError(
            f"need ell+1={ell + 1} verifier rows for ell={ell} drafts, "
            f"got {target_dist.shape[0]}"
        )

    verifier_greedy = target_dist.argmax(axis=1)
    out: list[int] = []
    for j in range(ell):
        tok = int(draft_tokens[j])
        if tok == int(verifier_greedy[j]):
            out.append(tok)
            continue
        # First disagreement: emit the verifier's token and stop. This is what the
        # verifier would have produced here on its own, which is why greedy output
        # is provably identical to running the verifier alone.
        out.append(int(verifier_greedy[j]))
        return AcceptResult(tokens=out, n_accepted=j, bonus=True)

    out.append(int(verifier_greedy[ell]))  # bonus token
    return AcceptResult(tokens=out, n_accepted=ell, bonus=True)


def speculative_accept(
    target_dist: np.ndarray,
    draft_dist: np.ndarray,
    draft_tokens: np.ndarray,
    rng: np.random.Generator,
) -> AcceptResult:
    """Stochastic speculative-sampling acceptance (unbiased w.r.t. the verifier).

    Parameters
    ----------
    target_dist:
        ``[ell + 1, vocab]`` verifier distributions (see :func:`greedy_accept`).
    draft_dist:
        ``[ell, vocab]`` drafter distributions the ``draft_tokens`` were sampled
        from. Only the probability the drafter assigned to each *sampled* token is
        strictly needed, but the full rows keep the residual-sampling step honest.
    draft_tokens:
        ``[ell]`` integer tokens proposed by the drafter.
    rng:
        A ``numpy`` generator, so runs are reproducible and the test suite can
        drive many independent trials.

    For each position ``j`` we accept ``x = draft_tokens[j]`` with probability
    ``min(1, p_j(x) / q_j(x))``. On the first rejection we resample from the
    normalised residual ``relu(p_j - q_j)`` and stop. If every draft is accepted
    we sample a bonus token from ``p_ell``.
    """
    target_dist = _as_prob_block(target_dist)
    draft_dist = _as_prob_block(draft_dist)
    draft_tokens = np.asarray(draft_tokens, dtype=np.int64).ravel()
    ell = draft_tokens.shape[0]
    if target_dist.shape[0] != ell + 1:
        raise ValueError(
            f"need ell+1={ell + 1} verifier rows for ell={ell} drafts, "
            f"got {target_dist.shape[0]}"
        )
    if draft_dist.shape[0] != ell:
        raise ValueError(f"need ell={ell} drafter rows, got {draft_dist.shape[0]}")

    out: list[int] = []
    for j in range(ell):
        x = int(draft_tokens[j])
        p_x = float(target_dist[j, x])
        q_x = float(draft_dist[j, x])
        # ratio >= 1 whenever the verifier likes the token at least as much as the
        # drafter did; q_x ~ 0 means the drafter would essentially never have
        # proposed x, so any verifier mass makes it a free accept.
        ratio = 1.0 if q_x <= _EPS else p_x / q_x
        if rng.random() < min(1.0, ratio):
            out.append(x)
            continue
        residual = target_dist[j] - draft_dist[j]
        np.clip(residual, 0.0, None, out=residual)
        total = residual.sum()
        # If p and q coincide on the rejected token the residual can vanish; fall
        # back to the raw verifier distribution, which is the correct limit.
        corrected = _sample(residual / total if total > _EPS else target_dist[j], rng)
        out.append(corrected)
        return AcceptResult(tokens=out, n_accepted=j, bonus=True)

    out.append(_sample(target_dist[ell], rng))  # bonus token from the verifier
    return AcceptResult(tokens=out, n_accepted=ell, bonus=True)


def _sample(prob: np.ndarray, rng: np.random.Generator) -> int:
    prob = np.asarray(prob, dtype=np.float64)
    s = prob.sum()
    if s <= _EPS:  # degenerate row; should not happen for a real softmax
        return int(prob.argmax())
    return int(rng.choice(prob.shape[0], p=prob / s))


def verify_block(
    target_dist: np.ndarray,
    draft_tokens: np.ndarray,
    ell: int,
    *,
    mode: str = "greedy",
    draft_dist: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> AcceptResult:
    """Verify the first ``ell`` tokens of a drafted block.

    This is the single entry point the inference loop calls. ``ell`` only *bounds
    how many* drafted tokens are checked, it never changes the accept/reject
    decision for the tokens that are checked, which is what keeps the
    verification-length policy (spec section 9.5) exactness-preserving.
    """
    draft_tokens = np.asarray(draft_tokens, dtype=np.int64).ravel()
    block = draft_tokens.shape[0]
    if not 0 <= ell <= block:
        raise ValueError(f"ell={ell} out of range [0, {block}]")

    if ell == 0:
        # Verified nothing; the verifier still owns the next token.
        row = _as_prob_block(target_dist)[0]
        if mode == "greedy":
            return AcceptResult(tokens=[int(row.argmax())], n_accepted=0, bonus=True)
        if rng is None:
            raise ValueError("stochastic mode requires an rng")
        return AcceptResult(tokens=[_sample(row, rng)], n_accepted=0, bonus=True)

    target_head = _as_prob_block(target_dist)[: ell + 1]
    heads = draft_tokens[:ell]
    if mode == "greedy":
        return greedy_accept(target_head, heads)
    if mode == "stochastic":
        if draft_dist is None or rng is None:
            raise ValueError("stochastic mode requires draft_dist and rng")
        return speculative_accept(target_head, np.asarray(draft_dist)[:ell], heads, rng)
    raise ValueError(f"unknown mode {mode!r}; expected 'greedy' or 'stochastic'")
