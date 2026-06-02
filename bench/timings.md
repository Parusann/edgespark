# Latency parameters (RX 7900 XTX, Qwen3-4B)

The throughput model in `bench/simulate.py` is driven by per-op timings measured
on the target machine. They live here so the model and the hardware stay in sync:
re-measure with `scripts/run_benchmark.py --hardware`, update the constants in
`bench/simulate.py`, and the modelled numbers move with them.

| Symbol | Constant in `simulate.py` | Meaning | Value |
|---|---|---|---|
| `T_decode` | `_T_DECODE_MS` | one baseline decode forward (INT8 verifier) | 15.5 ms |
| `T_verify(0)` | `_T_VERIFY0_MS` | verify-pass fixed cost (forward + attention over the block) | 19.0 ms |
| `dT_verify/dℓ` | `_T_VERIFY_PER_ELL_MS` | marginal cost per verified position (LM head over a 151,936-way vocab) | 4.2 ms |
| `T_draft` (fp16) | `_PRECISION["fp16"]["t_draft_ms"]` | fp16 drafter forward | 6.5 ms |
| `T_draft` (int8) | `_PRECISION["int8"]["t_draft_ms"]` | INT8 drafter forward | 5.0 ms |
| `T_draft` (nf4) | `_PRECISION["nf4"]["t_draft_ms"]` | NF4 drafter forward | 4.5 ms |

## Why the per-ℓ term matters

The verify pass is not free per position: each of the `ℓ+1` scored positions needs
a projection through the 151,936-way LM head plus its slice of block attention. On
a 4B model that per-position term is small but non-negligible, and it is exactly
what creates an optimum verification length **below** the full block — the reason
the confidence-gated policy (§9.5) beats always-verify-all. If you re-measure and
find `dT_verify/dℓ` near zero on your build (e.g. a fused kernel that hides it),
expect the gating win to shrink and always-verify-all to become near-optimal;
that is a legitimate hardware-dependent outcome, and the harness will report it.

## Measuring

`GpuTimer` (in `edgespark/utils/timing.py`) brackets the draft and verify regions
with CUDA/HIP events and synchronises before reading elapsed time — do not trust
wall-clock around async kernel launches. Average over a warm run (discard the
first two prompts) to avoid compile/allocation noise.
