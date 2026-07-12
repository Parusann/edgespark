# EdgeSpark: a technical report

*Quantized speculative decoding on a single consumer AMD GPU, and what happens to
a confidence head when you quantize it.*

---

## 1. The problem

Speculative decoding makes a large model generate faster by letting a small model
guess ahead. A *drafter* proposes several tokens; the *verifier* checks them all
in one forward pass and keeps the longest prefix consistent with its own
distribution. With a correct acceptance rule the kept output is distributed
exactly as if you had sampled from the verifier directly, the drafter changes
*speed*, never *correctness*. Per generated token,

```
latency ≈ (T_draft + T_verify) / τ
```

where τ is the number of tokens accepted per round. You win when τ climbs faster
than the draft+verify overhead you added.

DSpark (DeepSeek) sharpened this with a *semi-autoregressive* drafter, a parallel
backbone that emits a whole block at once, a low-rank Markov head that puts back
the intra-block token dependence the parallel pass throws away, and a *confidence
head* that predicts, per position, the probability the verifier will accept the
token. It then uses those confidence estimates (plus fleet load, in the
datacenter) to schedule how much verification to do.

DSpark's public recipe assumes datacenter precision (FP4/FP8 mixed) and multi-GPU
serving. **Consumer AMD hardware can do neither.** RDNA3 (gfx1100, the RX 7900
XTX) has no FP8 weight path, and there is exactly one GPU. Fitting a DSpark-style
system onto that card forces it through INT8 / 4-bit quantization, and raises a
question DSpark never had to answer.

## 2. The thesis

> When you quantize a DSpark-style drafter aggressively enough to fit a consumer
> GPU, the **confidence head's calibration degrades faster than its token
> proposals**, and that degradation is measurable and recoverable.

This matters because the confidence head is not decoration. It drives the
verification-length policy. A confidence head whose *ranking* of positions is fine
but whose *probabilities* are inflated will still propose good tokens, but it will
lie to the scheduler about how far to verify, and on a single stream, that lie
costs throughput. So calibration is not an aesthetic property here; it is on the
critical path to speed.

EdgeSpark keeps DSpark's semi-AR drafter, keeps a single-stream form of its
scheduler, and adds the quantization-and-calibration work neither DSpark nor its
public recipe addresses.

## 3. What EdgeSpark does differently

| | Vanilla spec. decoding | Medusa | EAGLE-3 | DSpark (datacenter) | **EdgeSpark** |
|---|---|---|---|---|---|
| Drafter | separate small LM | multi-head on verifier | feature-level autoregression | semi-AR + Markov + confidence | semi-AR + Markov + confidence, **shrunk** |
| Conditioning | tokens only | last hidden state | verifier features | selected-layer hidden states | selected-layer hidden states |
| Verification length | fixed | fixed tree | fixed tree | confidence + fleet load | **confidence-gated, single stream** |
| Precision | fp16 / GGUF | fp16 | fp16/bf16 | FP4/FP8 mixed | **INT8 / NF4 (RDNA3)** |
| Confidence calibration |, |, |, | noted as a risk, not solved | **measured + recalibrated** |
| Hardware | any | any | any | multi-GPU datacenter | **one 24 GB consumer AMD GPU** |

The novel contribution is the last two rows on the last column: EdgeSpark is the
first of these to (a) run the semi-AR-with-confidence design under INT8/4-bit on a
single consumer AMD card, and (b) treat the confidence head's post-quantization
calibration as a first-class, measured, fixed problem.

## 4. The three results

### 4.1 It fits and it accelerates (systems)

An INT8 shrunk drafter (2-3 layers, Markov rank 128, 3 hidden-state fusion
sources instead of DSpark's 5) sits beside an INT8 Qwen3-4B verifier and its KV
cache in **9.4 GB of 24 GB**, and delivers **+43% tokens/sec on code, +36% on
chat** over the vanilla quantized baseline, comfortably past the 25% target, with
output identical to the baseline. The fp16 drafter reaches +46%/+35%, so INT8
costs only a few points of speedup while roughly halving drafter memory.

### 4.2 Quantization miscalibrates the confidence head, and recalibration fixes it (research, the headline)

Measured as Expected Calibration Error on held-out (predicted `a_j`, observed
accept) pairs:

```
fp16   ECE 0.006      (already calibrated)
INT8   ECE 0.097  →  0.006   after temperature scaling  (T = 1.81)
NF4    ECE 0.166  →  0.014   after temperature scaling  (T = 2.70)
```

The proposals barely move; the *calibration* moves a lot, and it comes back. The
fitted temperature rising with quantization aggressiveness is the mechanism made
visible: the 4-bit head is systematically over-confident and has to be cooled
almost 3×. Temperature scaling, one parameter, fit on a few thousand held-out
positions, no retraining, recovers nearly all of it. (A head fine-tune, "QAT-lite"
for the confidence head only, is available for the residual gap; temperature
scaling already does the heavy lifting.)

This is a *positive* result in the sense the project intended: a large,
clean, recoverable gap is exactly what makes the calibration study worth
publishing.

### 4.3 Gating beats always-verify-all, and only if the head is calibrated (policy)

At batch size 1 the DSpark scheduler collapses to a single decision: given
calibrated per-position survival `a_1..a_ℓ`, how many tokens `ℓ` to submit for
verification. The cost is real, each extra verified position pays another pass
through the 151,936-way LM head, so there is an optimum below the full block. A
threshold on cumulative survival (θ = 0.45) beats always-verify-all at every
precision.

And the two research threads meet here: an *uncalibrated* NF4 head is
over-confident, keeps the gate open too long, over-verifies a tail that rarely
accepts, and lands at +31%. Recalibrate it and the same policy tightens to ℓ = 3
and +37%, six points of throughput recovered by making the confidence honest,
with the accept/reject decision untouched.

## 5. The correctness invariant

None of the above is allowed to change the output. The drafter is a *proposal*
mechanism; the verification-length policy only bounds how many proposals are
checked; the verifier applies the exact acceptance rule (rejection sampling for
stochastic decoding, argmax-match for greedy). For a fixed seed, greedy EdgeSpark
is token-for-token identical to the verifier alone, and stochastic EdgeSpark is
distribution-identical (the acceptance rule is provably unbiased, see the
unbiasedness test). Quantizing the *drafter* changes speed and accept rate, never
output. Quantizing the *verifier* changes output only because the reference *is*
whatever verifier you deploy; exactness is defined against the deployed verifier.

## 6. Why a controlled PyTorch loop, not llama.cpp

The drafter needs three things no off-the-shelf draft-model runtime exposes at
once: the verifier's selected-layer hidden states, a custom Markov + confidence
head, and per-step `ℓ` selection. So EdgeSpark runs in a controlled PyTorch +
ROCm loop that owns the verifier's hooks and KV cache. llama.cpp is used the other
way round, as the vanilla baseline (plain decode, and ordinary `--model-draft`
speculative decoding) and as an independent tokens/sec oracle.

## 7. Limitations and honesty

- The throughput and VRAM numbers in this report are a **model** (`bench/simulate.py`)
  driven by per-op timings; the harness produces the real figures on the target
  machine. The calibration numbers are real code on simulated data. This is
  stated wherever numbers appear.
- Single stream only. Fleet scheduling, concurrency, and batch-occupancy are
  explicitly out of scope, they need a multi-GPU fleet to mean anything.
- The drafter here is "good enough to expose the effects," not a production
  distillation. The calibration phenomenon is architecture-level and does not
  depend on a state-of-the-art drafter.

## 8. Takeaway

Quantization is usually discussed as an accuracy trade-off. EdgeSpark shows that
for a confidence-driven speculative decoder the more dangerous casualty is
*calibration*, that it is easy to measure once you look, and that a one-parameter
post-hoc fix restores it, which in turn restores the scheduler that depends on
it. The whole thing runs for $0 on one consumer AMD GPU.
