"""Benchmark harness (spec sections 9.7, 12).

Replays fixed prompt sets single-stream, logs every metric from section 12 as
JSON lines, records exactness pass/fail per run, and emits a summary. Three ways
to run it:

* ``run_hardware`` — the real thing on the RX 7900 XTX: builds the verifier and
  quantized drafter, times EdgeSpark against the vanilla baseline.
* ``run_smoke`` — the numpy reference loop with toy LMs. No torch, no GPU; it
  exercises the whole plumbing (draft → gate → verify → log → summarise) so the
  harness itself is testable in CI.
* modelled numbers come from ``bench.simulate`` via ``scripts/run_benchmark.py
  --simulate``.

The point of keeping all three behind one summary shape is that a figure or a
README table never has to care which produced it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from edgespark.loop.reference import speculative_decode
from edgespark.policy import ThresholdPolicy
from edgespark.utils import ToyCategoricalLM


@dataclass
class RunSummary:
    label: str
    precision: str
    prompt_set: str
    tokens_generated: int
    mean_tau: float
    mean_accepted: float
    exact_ok: bool
    tok_s: float = 0.0
    tok_s_baseline: float = 0.0
    vram_mb: float = 0.0

    @property
    def speedup(self) -> float:
        return self.tok_s / self.tok_s_baseline if self.tok_s_baseline else 0.0


@dataclass
class BenchReport:
    runs: list[RunSummary] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({"runs": [asdict(r) for r in self.runs]}, indent=2)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json(), encoding="utf-8")


# --- CPU smoke path (actually runs, no torch) --------------------------------

def run_smoke(n_tokens: int = 120, block_size: int = 5, seed: int = 0) -> BenchReport:
    """Exercise the full loop with toy LMs and assert exactness holds.

    Returns a summary in the same shape as a hardware run. ``tok_s`` here is
    meaningless (CPU toy model) and left at zero; the value is that the accept /
    gate / verify / summarise pipeline runs and is checked for exactness.
    """
    report = BenchReport()
    verifier = ToyCategoricalLM(vocab_size=32, seed=seed)
    # A strong drafter (low logit noise) and a weak one (high noise), both
    # correlated with the verifier so acceptance is partial and realistic.
    for label, noise in [("edgespark", 0.35), ("weak-drafter", 1.1)]:
        drafter = ToyCategoricalLM.perturbed(verifier, noise=noise, seed=seed + 1)
        reference = verifier.greedy_generate(0, n_tokens)
        result = speculative_decode(
            verifier, drafter, 0, n_tokens, block_size=block_size,
            mode="greedy", policy=ThresholdPolicy(theta=0.45),
        )
        report.runs.append(
            RunSummary(
                label=label, precision="cpu-toy", prompt_set="synthetic",
                tokens_generated=len(result.tokens),
                mean_tau=result.mean_tau, mean_accepted=result.mean_accepted,
                exact_ok=(result.tokens == reference),
            )
        )
    return report


# --- Hardware path (torch/ROCm) ----------------------------------------------

def run_hardware(bench_config, drafter_config, policy_config, prompts: dict) -> BenchReport:
    """Full hardware benchmark. Imports torch lazily; raises if unavailable."""
    import torch  # noqa: F401

    from edgespark.loop.generate import EdgeSparkGenerator
    from edgespark.policy import ThresholdPolicy
    from edgespark.utils import vram
    from edgespark.verifier import Verifier, VerifierConfig

    report = BenchReport()
    for precision in bench_config.precisions:
        verifier = Verifier(VerifierConfig(model_name=drafter_config.target_model)).load()
        drafter = _build_drafter(drafter_config, precision, verifier.hidden_size)
        policy = ThresholdPolicy(theta=policy_config.theta, ema_gain=policy_config.ema_gain,
                                 min_length=policy_config.min_length)
        gen = EdgeSparkGenerator(verifier, drafter, policy, precision=precision)

        for prompt_set in bench_config.prompt_sets:
            base_tok_s = _measure_baseline(verifier, prompts[prompt_set], bench_config)
            agg_tau, agg_acc, toks, exact = [], [], 0, True
            t0 = _now()
            for prompt in prompts[prompt_set]:
                out = gen.generate(prompt, bench_config.max_new_tokens,
                                   mode=bench_config.decoding, exactness_check=True)
                agg_tau.append(out.mean_tau)
                agg_acc.append(out.mean_accepted)
                toks += len(out.tokens)
                exact = exact and all(m.exact_ok for m in out.metrics)
            elapsed = _now() - t0
            report.runs.append(
                RunSummary(
                    label="edgespark", precision=precision, prompt_set=prompt_set,
                    tokens_generated=toks, mean_tau=float(np.mean(agg_tau)),
                    mean_accepted=float(np.mean(agg_acc)), exact_ok=exact,
                    tok_s=toks / elapsed if elapsed else 0.0, tok_s_baseline=base_tok_s,
                    vram_mb=vram.snapshot().reserved_mb,
                )
            )
    return report


def _build_drafter(drafter_config, precision, verifier_hidden):
    from edgespark.drafter import EdgeSparkDrafter
    from edgespark.quantize import quantize_drafter_int8, quantize_drafter_nf4
    from edgespark.utils.config import QuantConfig

    drafter = EdgeSparkDrafter(drafter_config, verifier_hidden=verifier_hidden)
    if precision == "int8":
        drafter, _ = quantize_drafter_int8(drafter, QuantConfig.from_yaml("configs/quant_int8.yaml"))
    elif precision == "nf4":
        drafter, _ = quantize_drafter_nf4(drafter, QuantConfig.from_yaml("configs/quant_nf4.yaml"))
    return drafter


def _measure_baseline(verifier, prompts, bench_config) -> float:
    """Vanilla autoregressive decode tok/s — baseline 1 from section 12."""
    import torch

    toks, t0 = 0, _now()
    for prompt in prompts:
        seq = verifier.encode(prompt)
        _, _, kv = verifier.forward_with_hidden(seq)
        for _ in range(bench_config.max_new_tokens):
            logits, _, kv = verifier.forward_with_hidden(seq[:, -1:], past_key_values=kv)
            nxt = logits[0, -1].argmax().view(1, 1)
            seq = torch.cat([seq, nxt], dim=1)
            toks += 1
    dt = _now() - t0
    return toks / dt if dt else 0.0


def _now() -> float:
    import time

    return time.perf_counter()
