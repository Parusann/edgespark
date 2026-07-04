# Development log

A running journal of the build. Phase numbering follows the project spec (§16).

---

### Phase 0 — Environment + verification gate  ·  Apr 26 – May 2

Stood up ROCm 7.2.4 via the "ROCm on Radeon and Ryzen" consumer path; `rocminfo`
sees `gfx1100`. PyTorch ROCm wheel runs a Qwen3-4B fp16 forward pass on the card.
`scripts/check_env.py` is the gate — it refuses to proceed unless the GPU, a
Qwen3 forward, and an INT8 quant tool all work.

Notes / friction:
- flash-attention: the stock kernels don't build; the `howiejay/navi_support`
  fork does but was flaky under load. Decided early to make **SDPA the default**
  and treat flash-attn as an optional optimization, per the spec's risk table.
- bitsandbytes-ROCm: `Linear8bitLt` runs on gfx1100 after matching the ROCm
  version; a tiny 8-bit matmul smoke test is now part of the Phase 0 gate.
- **Path decision:** attempted to load a released DSpark/EAGLE-3 Qwen3 drafter.
  Went with Path A (distill from released checkpoint) as primary, Path B
  (from-scratch on the small cache) kept as fallback.

### Phase 1 — Baselines  ·  May 3 – May 9

Two baselines on the target machine: vanilla INT8 decode (~64 tok/s) and
llama.cpp `--model-draft` speculative decoding as an independent oracle. Wrote the
harness skeleton and the JSONL metrics record first so every later phase logs the
same shape.

### Phase 2 — DSpark-style drafter (fp16)  ·  May 10 – May 24

Built the shrunk semi-AR drafter: parallel backbone with hidden-state fusion over
`[1, 17, 33]`, low-rank Markov head (rank 128), confidence head. The exactness
harness came online here and immediately earned its keep — an early Markov-bias
sign error passed "looks fine" eyeballing but failed token-for-token equality.
Lesson re-learned: **the exactness test is the spec, not a formality.**

Gate: fp16 EdgeSpark passes exactness, τ > 1, net tok/s over baseline. ✅

### Phase 3 — Quantize the drafter  ·  May 25 – Jun 3

INT8 (W8A8) and NF4 variants. Added *fake-quant* simulators alongside the real
bitsandbytes path so the calibration study could be developed and unit-tested off
the GPU. INT8 keeps proposal agreement high; NF4 is visibly rougher — which is the
point, it's the stress case for Phase 4.

Gate: INT8 in VRAM budget, exactness holds, proposals acceptable; NF4 produced. ✅

### Phase 4 — Confidence calibration study (headline)  ·  Jun 4 – Jun 17

The core of the project. Logged (predicted `a_j`, observed accept) pairs per
precision, computed ECE / Brier / reliability. The gap is real and large: NF4 ECE
0.166 vs fp16 0.006, while top-1 proposals barely move. Implemented temperature
and Platt recalibration; temperature scaling alone recovers NF4 ECE to 0.014. The
"money plot" (reliability before/after) is `docs/assets/reliability_diagrams.svg`.

Gate: reliability diagrams exist; recalibrated ECE near fp16. ✅

### Phase 5 — Verification-length policy  ·  Jun 18 – Jun 26

Single-stream threshold policy on calibrated cumulative survival. Tuned θ on
held-out data (0.45). Ablated against always-verify-all: gating wins at every
precision, and — the satisfying part — the win *depends* on calibration, because
an over-confident head over-verifies. This is where the two research threads
turned out to be one.

Gate: gated policy improves tok/s over always-verify-all, exactness preserved. ✅

### Phase 6 — End-to-end evaluation  ·  Jun 27 – Jul 1

Full harness across precisions and prompt sets; produced the four key charts;
confirmed the primary criterion (≥25% tok/s, quality unchanged). INT8 code +43%.

### Phase 7 — Demo + writeup  ·  Jul 2 – Jul 5

Side-by-side dashboard (`demo/`) — vanilla vs EdgeSpark, live tok/s, draft-depth
strip. Technical report positioning EdgeSpark against vanilla speculative
decoding, EAGLE-3, DFlash, Medusa, and DSpark's public recipe. README, results
page, and this log.
