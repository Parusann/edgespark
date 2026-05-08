"""The mandatory exactness suite (spec section 13).

EdgeSpark is only allowed to be faster, never different. These tests pin that
down two ways:

* Greedy decoding is *token-for-token identical* to the verifier decoding on its
  own — and stays identical under every verification-length policy, because the
  policy only bounds how many drafts are checked.
* Stochastic decoding is *distribution-identical*: the acceptance rule is
  unbiased, so over many samples the emitted token distribution matches the
  verifier's own. (Token-identity is the wrong claim under sampling — the two
  paths consume randomness differently — so we test the property that is
  actually true.)
"""

from __future__ import annotations

import numpy as np
import pytest

from edgespark.loop import speculative_decode, verify_block
from edgespark.loop.acceptance import greedy_accept, speculative_accept
from edgespark.policy import ThresholdPolicy
from edgespark.utils import ToyCategoricalLM

# --- greedy: token-for-token identity, under any policy ----------------------

@pytest.mark.parametrize("block_size", [1, 3, 5, 7])
@pytest.mark.parametrize("agreement", [0.6, 1.0, 1.7])
def test_greedy_matches_verifier_alone(block_size, agreement):
    # A shared verifier; the drafter is the same model warmed to a different
    # temperature so agreement (and thus accept rate) varies across cases.
    verifier = ToyCategoricalLM(vocab_size=12, seed=1, temperature=1.0)
    drafter = ToyCategoricalLM(vocab_size=12, seed=1, temperature=agreement)

    reference = verifier.greedy_generate(start_token=0, n=200)
    spec = speculative_decode(
        verifier, drafter, start_token=0, n_tokens=200,
        block_size=block_size, mode="greedy",
    )
    assert spec.tokens == reference


def test_greedy_identical_across_verification_lengths():
    # Same output regardless of how aggressively the policy truncates verification.
    verifier = ToyCategoricalLM(vocab_size=10, seed=3)
    drafter = ToyCategoricalLM(vocab_size=10, seed=7, temperature=1.3)
    reference = verifier.greedy_generate(0, 150)

    for theta in (0.1, 0.4, 0.7, 0.95):
        policy = ThresholdPolicy(theta=theta)
        spec = speculative_decode(
            verifier, drafter, 0, 150, block_size=5, mode="greedy", policy=policy
        )
        assert spec.tokens == reference, f"diverged at theta={theta}"


def test_drafter_quality_changes_speed_not_output():
    # A deliberately awful drafter still yields identical tokens — just slower.
    verifier = ToyCategoricalLM(vocab_size=16, seed=2)
    good = ToyCategoricalLM(vocab_size=16, seed=2, temperature=0.7)
    awful = ToyCategoricalLM(vocab_size=16, seed=99, temperature=3.0)
    ref = verifier.greedy_generate(0, 120)

    a = speculative_decode(verifier, good, 0, 120, block_size=5, mode="greedy")
    b = speculative_decode(verifier, awful, 0, 120, block_size=5, mode="greedy")
    assert a.tokens == ref and b.tokens == ref
    # The good drafter should accept more on average (sanity, not correctness).
    assert a.mean_accepted >= b.mean_accepted


# --- greedy acceptance unit behaviour ----------------------------------------

def test_greedy_accept_full_and_reject():
    # Construct a block whose verifier argmax at row j is token (j mod V).
    V = 5
    target = np.zeros((4, V))
    for j in range(4):
        target[j, j % V] = 1.0

    # All correct: drafts [0,1,2] then bonus argmax at row 3 -> 3.
    res = greedy_accept(target, np.array([0, 1, 2]))
    assert res.tokens == [0, 1, 2, 3]
    assert res.n_accepted == 3 and res.bonus

    # Mismatch at position 1: accept [0], correct to verifier's argmax (1), stop.
    res = greedy_accept(target, np.array([0, 4, 2]))
    assert res.tokens == [0, 1]
    assert res.n_accepted == 1


# --- stochastic: unbiasedness ------------------------------------------------

def _tv(a, b):
    return 0.5 * float(np.abs(a - b).sum())


def test_speculative_sampling_is_unbiased():
    # Single-position block: the emitted token must be distributed as the
    # verifier's p, whatever the drafter's q is.
    rng = np.random.default_rng(0)
    V = 8
    p = _softmax(rng.standard_normal(V))
    q = _softmax(rng.standard_normal(V))

    n = 60000
    counts = np.zeros(V)
    for _ in range(n):
        x = int(rng.choice(V, p=q))
        res = speculative_accept(
            target_dist=np.stack([p, p]),  # row 0 checks x, row 1 is bonus
            draft_dist=q[None, :],
            draft_tokens=np.array([x]),
            rng=rng,
        )
        counts[res.tokens[0]] += 1

    emp = counts / counts.sum()
    assert _tv(emp, p) < 0.02, f"TV={_tv(emp, p):.4f} too large"


def test_multi_token_sampling_matches_verifier_distribution():
    # Whole short sequences: distribution of speculative output ~ verifier output.
    # Length 2 keeps the outcome space small (V**2) so Monte-Carlo noise in the
    # total-variation estimate stays well under the tolerance; the single-token
    # test above is the sharp unbiasedness check.
    rng = np.random.default_rng(11)
    verifier = ToyCategoricalLM(vocab_size=6, seed=5)
    drafter = ToyCategoricalLM(vocab_size=6, seed=8, temperature=1.4)
    L, n = 2, 80000

    def hist(sample_fn):
        table = {}
        for _ in range(n):
            key = tuple(sample_fn())
            table[key] = table.get(key, 0) + 1
        return table

    ref = hist(lambda: verifier.sample_generate(0, L, rng))
    spec = hist(
        lambda: speculative_decode(
            verifier, drafter, 0, L, block_size=3, mode="stochastic", rng=rng
        ).tokens
    )
    keys = set(ref) | set(spec)
    tv = 0.5 * sum(abs(ref.get(k, 0) / n - spec.get(k, 0) / n) for k in keys)
    assert tv < 0.02, f"sequence-level TV={tv:.4f} too large"


# --- policy safety -----------------------------------------------------------

def test_ell_never_exceeds_block():
    rng = np.random.default_rng(4)
    target = np.stack([_softmax(rng.standard_normal(7)) for _ in range(6)])
    with pytest.raises(ValueError):
        verify_block(target, np.arange(5), ell=6)  # ell > block


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()
