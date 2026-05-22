"""The harness plumbing runs end-to-end on CPU and preserves exactness."""

from __future__ import annotations

from bench.harness import run_smoke


def test_smoke_run_is_exact_and_accepts_tokens():
    report = run_smoke(n_tokens=100, block_size=5, seed=0)
    assert len(report.runs) == 2
    for run in report.runs:
        # Exactness must hold whatever the drafter quality.
        assert run.exact_ok, f"{run.label} broke exactness"
        assert run.tokens_generated == 100
        # tau counts the trailing verifier token, so it is always >= 1.
        assert run.mean_tau >= 1.0

    edgespark = next(r for r in report.runs if r.label == "edgespark")
    weak = next(r for r in report.runs if r.label == "weak-drafter")
    # A better drafter accepts more per round; both stay exact.
    assert edgespark.mean_accepted >= weak.mean_accepted
