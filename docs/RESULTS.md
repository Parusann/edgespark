# Results

> **Provenance.** Two layers of results. **Measured on the RX 7900 XTX**
> (native-Windows ROCm, 2026-07-05 — see [`runs/hardware/`](../runs/hardware/)): the
> exactness invariant on the real Qwen3-4B, fp16 per-op latencies, VRAM, and
> llama.cpp baselines. **Modelled** (`bench/simulate.py` →
> [`summary.json`](../runs/reference/summary.json)): the INT8/NF4 throughput, the
> gated-policy ablation, and the calibration study — because bitsandbytes is
> unavailable on this Windows-ROCm stack, so no quantized drafter could be built or
> trained yet. The calibration *figures* are produced by the **real**
> `edgespark.calibration` code on simulated confidence data (the same path a
> hardware run takes). §0 separates what is measured from what is modelled.

## 0. Hardware validation status

| Claim | State | Evidence |
|---|---|---|
| Output identical to the verifier (greedy) | ✅ **measured on real Qwen3-4B** | `loop/generate.py` end-to-end, `exact_ok=true` for any drafter quality |
| Fits 24 GB | ✅ **measured** | fp16 verifier 7.7 GB + drafter 1.8 GB, 9.5 GB generation peak, ~15 GB free ([vram.json](../runs/hardware/vram.json)) |
| Baselines | ✅ **measured** | llama.cpp Q4_K_M 214 tok/s, Q8_0 151 tok/s; vanilla speculative 0.96× ([baselines.json](../runs/hardware/baselines.json)) |
| fp16 per-op latency | ✅ **measured** | decode 2.9 ms, intrinsic verify 3.2 ms, draft 1.0 ms ([timings.md](../bench/timings.md)) |
| INT8 / NF4 throughput | 🔶 **modelled** | no bitsandbytes on native-Windows ROCm (Phase-0 gate MISS) |
| Calibration study (ECE / recalibration) | 🔶 **modelled** | needs a trained quantized drafter; the method itself is code-tested (§2) |
| Gated policy beats always-verify-all | ⚠️ **modelled only** | on-hardware the per-ℓ verify cost is ≈ 0, so gating **ties** always-verify-all |
| End-to-end speculative speedup | ⏳ **pending** | current `block_distribution` re-encodes the prefix (prefill-bound ~52.6 ms); needs the KV-reuse fix |

Sections 1–3 and 5 below are the **design-time model** (`bench/simulate.py`), kept
as a coherent projection and labelled as such — the target the hardware run is
working toward, not a claim of measured throughput. Section 4 (exactness) and the
fp16 VRAM figures are measured.

## Design-time targets

Whether the **model** meets each success criterion (see §0 for what is measured on
hardware vs. still modelled):

| Result | Criterion | Model / status |
|---|---|---|
| End-to-end throughput, INT8 drafter, code | ≥ 25% over vanilla quantized baseline | +43% · 🔶 modelled |
| End-to-end throughput, INT8 drafter, chat | ≥ 25% | +36% · 🔶 modelled |
| Confidence ECE recovered by recalibration (NF4) | near fp16 | 0.166 → 0.014 · 🔶 modelled |
| Confidence-gated policy vs. always-verify-all | higher tok/s | wins · ⚠️ ties on this GPU |
| Verifier + drafter + KV in 24 GB | fits with headroom | ✅ measured 9.5 GB peak, ~15 GB free |
| Output vs. deployed verifier | token-for-token identical (greedy) | ✅ **measured on real Qwen3-4B** |

## 1. Throughput (modelled)

![Throughput — design-time model](assets/throughput.svg)

**Design-time model** (§0) — single stream, Qwen3-4B INT8 verifier, greedy
decoding, gated verification; baseline is the vanilla quantized verifier with no
speculation (64.5 tok/s). Not yet measured on hardware: INT8 needs bitsandbytes,
and end-to-end throughput needs the KV-reuse fix (today's verify is prefill-bound).

| Drafter precision | code (tok/s) | code speedup | chat (tok/s) | chat speedup |
|---|---|---|---|---|
| fp16 | 93.9 | +46% | 86.9 | +35% |
| **INT8** | **92.4** | **+43%** | **87.8** | **+36%** |
| NF4 | 88.1 | +37% | 83.9 | +30% |

The fp16 drafter isolates the quantization effect: INT8 gives up only ~3% of the
fp16 speedup while cutting drafter VRAM roughly in half. Every variant clears the
25% primary criterion, with output identical to the baseline (§4).

## 2. Calibration — the headline study

![Reliability diagrams](assets/reliability_diagrams.svg)

Quantization damages the confidence head's *calibration* far more than its token
*proposals*. Temperature scaling on a small held-out set restores it to near-fp16.

| Precision | ECE (raw) | ECE (recalibrated) | Brier (raw → recal) | Temperature |
|---|---|---|---|---|
| fp16 | 0.006 | 0.003 | 0.159 → 0.159 | 1.03 |
| INT8 | 0.097 | 0.006 | 0.179 → 0.168 | 1.81 |
| NF4 | 0.166 | 0.014 | 0.218 → 0.188 | 2.70 |

Two things to read off the table. First, the fitted temperature climbs with
quantization aggressiveness (1.03 → 1.81 → 2.70): the 4-bit head is badly
over-confident and needs to be cooled hard. Second, the ECE recovery is almost
total — a large, cleanly *recoverable* miscalibration gap, which is exactly the
positive result the project set out to find (spec §17).

## 3. Verification-length policy

![Policy ablation](assets/policy_ablation.svg)

The confidence-gated policy stops verifying once cumulative predicted survival
falls below θ = 0.45, rather than always verifying the full block. It accepts a
*lower* τ per round but spends much less verifier time, so tokens/sec rises at
every precision — **in this design-time model**.

> ⚠️ **Hardware caveat (§0).** The gating win depends on the verifier's per-position
> cost growing with ℓ. On the RX 7900 XTX that marginal measured ≈ 0 (a 4B forward
> is dominated by the 36-layer weight sweep), so gating **ties** always-verify-all
> there. The result holds on hardware where verify scales with ℓ; it stays
> exactness-preserving either way.

| Drafter | gated ℓ | gated τ | always-verify-all ℓ | always τ | gated wins? |
|---|---|---|---|---|---|
| fp16 | 4 | 3.97 | 5 | 4.27 | ✅ (+46% vs +42%) |
| INT8 | 3 | 3.38 | 5 | 4.08 | ✅ (+43% vs +41%) |
| NF4 | 3 | 3.18 | 5 | 3.75 | ✅ (+37% vs +31%) |

**Why calibration and the policy are the same story.** The policy is only as good
as the confidence it gates on. An uncalibrated NF4 head is over-confident, so the
threshold rule keeps verifying a low-survival tail that rarely gets accepted:

| NF4 gating | ℓ | throughput |
|---|---|---|
| uncalibrated (over-confident) | 5 (over-verifies) | +31% |
| recalibrated | 3 | **+37%** |

Recalibration recovers ~6 points of throughput purely by making the gate honest —
without ever touching the accept/reject decision, so output is unchanged.

## 4. Exactness

Greedy EdgeSpark output is token-for-token identical to the verifier decoding on
its own, for any drafter quality and any verification length. Stochastic decoding
is distribution-identical (the acceptance rule is unbiased). Both are enforced in
[`tests/test_exactness.py`](../tests/test_exactness.py) — 18 checks including a
Monte-Carlo unbiasedness test at TV < 0.02 (37 tests across the full numpy suite).

## 5. VRAM

![VRAM breakdown](assets/vram_breakdown.svg)

| Configuration | Total | Headroom in 24 GB | Source |
|---|---|---|---|
| **fp16 verifier + fp16 drafter** | **9.5 GB peak** | **~15 GB free** | ✅ **measured** (short ctx; [vram.json](../runs/hardware/vram.json)) |
| fp16 verifier + INT8 drafter | 12.9 GB | 11.1 GB | 🔶 modelled (~8k KV) |
| INT8 verifier + INT8 drafter | 9.4 GB | 14.6 GB | 🔶 modelled (~8k KV) |
| INT8 verifier + NF4 drafter | 9.1 GB | 14.9 GB | 🔶 modelled (~8k KV) |

The measured fp16+fp16 peak (9.5 GB, verifier 7.7 + drafter 1.8) is at short
context; the modelled rows add ~4 GB of KV cache + activations at ~8k context.
Either way the pair sits well inside 24 GB — comfortable room for an 8B verifier
(quantized) or a much longer context (spec §11).
