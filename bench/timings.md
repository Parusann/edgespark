# Latency parameters (RX 7900 XTX, Qwen3-4B)

The throughput model in `bench/simulate.py` is driven by per-op timings measured
on the target machine. They live here so the model and the hardware stay in sync:
re-measure, update the constants in `bench/simulate.py`, and the modelled numbers
move with them.

**Measured on the RX 7900 XTX (gfx1100), native-Windows ROCm, PyTorch
2.9.1+rocmsdk (HIP 7.2), fp16 Qwen3-4B, single stream, 2026-07-05.** Raw
artifacts in `runs/hardware/` (`baselines.json`, `vram.json`, `env.txt`).

| Symbol | Constant in `simulate.py` | Meaning | Modelled | Measured |
|---|---|---|---|---|
| `T_decode` | `_T_DECODE_MS` | one decode forward (fp16 verifier, single token, KV cache) | 15.5 ms | **2.9 ms** |
| `T_verify(0)` | `_T_VERIFY0_MS` | KV-cached per-block verify forward, fixed cost | 19.0 ms | **3.2 ms** |
| `dT_verify/dℓ` | `_T_VERIFY_PER_ELL_MS` | marginal cost per verified position | 4.2 ms | **~0 ms** (below noise) |
| `T_draft` (fp16) | `_PRECISION["fp16"]["t_draft_ms"]` | fp16 drafter forward (399M) | 6.5 ms | **1.0 ms** |
| `T_draft` (int8) | `_PRECISION["int8"]["t_draft_ms"]` | INT8 drafter forward | 5.0 ms | *not measured* |
| `T_draft` (nf4) | `_PRECISION["nf4"]["t_draft_ms"]` | NF4 drafter forward | 4.5 ms | *not measured* |

INT8/NF4 drafter timings are still the modelled values: bitsandbytes is
unavailable on this Windows-ROCm stack (Phase-0 gate MISS), so quantized drafters
are a Tier-1 item. See `runs/hardware/NOTES.md`.

## The per-ℓ term went to ~zero, a real hardware outcome

The modelled story: each of the `ℓ+1` scored positions needs a projection through
the 151,936-way LM head plus block attention, so the marginal per-position cost is
non-negligible and creates an optimum verification length **below** the full block
, the reason the confidence-gated policy (§9.5) beats always-verify-all.

On this hardware that marginal is **below measurement noise (<0.1 ms/position)**: a
KV-cached verify forward over ℓ=1..5 tokens costs ~3.2 ms essentially flat (a 4B
forward is dominated by the 36-layer weight sweep; 1-5 extra token positions are
lost in it). This is exactly the "`dT_verify/dℓ` near zero" case flagged in the
original note: **gating is roughly tied with always-verify-all here**, and the
model will report that rather than a gating win. It stays exactness-preserving
either way.

## Caveat: the current verify re-encodes the prefix

`_T_VERIFY0_MS` above is the *intrinsic* KV-cached per-block verify cost. The
current `Verifier.block_distribution` (used by `Verifier.verify`) runs the scoring
forward with `use_cache=False`, it **re-encodes the entire prefix every block**.
Measured that way at a 225-token context it costs **~52.6 ms + 0.37 ms/ℓ**
(prefill-bound, and it grows with context). So end-to-end EdgeSpark throughput
today is prefill-bound, not verify-marginal-bound; realising the ~3.2 ms verify
(and any speculative speedup) requires the KV-reuse optimisation already flagged in
`loop/generate.py` ("Reusing the verify pass's tail hidden state is the obvious
optimization").

## Measuring

`GpuTimer` (`edgespark/utils/timing.py`) brackets the draft and verify regions with
HIP events and synchronises before reading elapsed time, do not trust wall-clock
around async kernel launches, and synchronise after any un-timed prefill so its
tail does not leak into the timed region. Average over a warm run (discard the
first two iterations) to avoid compile/allocation noise.
