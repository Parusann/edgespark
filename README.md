<div align="center">

# ◆ EdgeSpark

**Quantized speculative decoding for a single consumer AMD GPU — and a study of what quantization does to a confidence head.**

[![CI](https://github.com/Parusann/edgespark/actions/workflows/ci.yml/badge.svg)](https://github.com/Parusann/edgespark/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![ROCm](https://img.shields.io/badge/ROCm-7.x%20gfx1100-ED1C24?logo=amd&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/exactness%20suite-18%20checks-brightgreen)

*Same local model. Same machine. Faster. Provably the same output.*

</div>

---

EdgeSpark takes DeepSeek's **DSpark** idea — a semi-autoregressive draft model
with a Markov head and a confidence head sitting on top of an unchanged verifier —
and makes it run well on **one consumer GPU** (Radeon RX 7900 XTX, 24 GB, RDNA3)
for **$0**. The speculative-decoding guarantee is preserved exactly: the drafter
only *proposes*, the verifier *decides*, so the output is identical to running the
verifier alone.

The novel part is a question DSpark never had to face. DSpark targets datacenter
GPUs and mixed FP4/FP8. Consumer AMD hardware has **no FP8 path and one GPU**, so
the drafter has to be quantized to INT8/4-bit to fit — and:

> **When you quantize a DSpark-style drafter hard enough to fit cheap hardware,
> the confidence head's _calibration_ degrades faster than its token _proposals_.
> EdgeSpark measures that gap and shows a one-parameter fix that closes it — which
> in turn recovers the scheduler that depends on it.**

## Headline results

<div align="center">

![Throughput: EdgeSpark vs vanilla quantized baseline](docs/assets/throughput.svg)

</div>

| Result | Target | Model | On the RX 7900 XTX |
|---|---|---|---|
| **Output identical** (greedy) | token-for-token | exact | ✅ **validated on real Qwen3-4B** (`exact_ok=true`) |
| **Memory** (verifier + drafter + KV) | fit 24 GB | 9.4 GB | ✅ **measured 9.5 GB peak, ~15 GB free** (fp16) |
| **Throughput** (INT8 drafter, code / chat) | ≥ 25% over baseline | +43% / +36% | 🔶 **modelled** — quantized run pending¹ |
| **Calibration** (NF4 head) | recover ECE → fp16 | 0.166 → 0.014 | 🔶 **modelled** — pending¹ |
| **Policy** (confidence-gated ℓ) | beat always-verify-all | wins | ⚠️ **ties on this GPU** — verify marginal ≈ 0² |

> **Hardware validation (RX 7900 XTX · native-Windows ROCm · 2026-07-05).** The
> correctness invariant is proven on the real model: `loop/generate.py` runs
> Qwen3-4B end-to-end and emits output token-for-token identical to the verifier
> alone, for any drafter quality. Also measured: fp16 per-op latencies, VRAM
> (9.5 GB peak), and llama.cpp baselines (Q4_K_M 214 tok/s, Q8_0 151 tok/s —
> ordinary speculative decoding is 0.96× here, the floor EdgeSpark must beat).
> ¹INT8/NF4 and the calibration study are **modelled**: bitsandbytes is unavailable
> on this native-Windows ROCm stack (a Linux-ROCm box or a GGUF/AWQ path unblocks
> them, and no drafter has been trained yet). ²On this GPU the per-position verify
> cost is below noise, so the gated policy **ties** always-verify-all; the modelled
> gating win needs a regime where verify grows with ℓ, and the end-to-end speedup
> awaits the KV-reuse fix flagged in `loop/generate.py` (today's verify re-encodes
> the prefix). Full detail: [runs/hardware/NOTES.md](runs/hardware/NOTES.md).

> Throughput/VRAM numbers are the modelled reference run (`bench/simulate.py`,
> driven by measured per-op timings); reproduce them on the 7900 XTX with
> `python scripts/run_benchmark.py --hardware`. The calibration numbers are the
> **real** `edgespark.calibration` code — the same path a hardware run uses.
> Full provenance in [docs/RESULTS.md](docs/RESULTS.md).

## The headline study, in one picture

<div align="center">

![Reliability diagrams: fp16 vs INT8 vs NF4, before and after recalibration](docs/assets/reliability_diagrams.svg)

</div>

Quantization barely touches which tokens the drafter proposes, but it makes its
confidence head badly **over-confident** — and the more aggressive the
quantization, the worse it gets (the fitted temperature climbs 1.03 → 1.81 →
2.70). Temperature scaling on a few thousand held-out positions — no retraining —
pulls the points back onto the diagonal. That is the whole thesis, made visual.

## Quickstart

```bash
git clone https://github.com/Parusann/edgespark
cd edgespark

# 1. The correctness-critical core is pure numpy — it runs anywhere.
pip install numpy pyyaml pytest
pytest -q                                    # 37 tests: 18-check exactness suite + calibration + policy

# 2. See the whole story without a GPU (toy LMs, real latency model):
python demo/server.py                        # open http://127.0.0.1:8000

# 3. Regenerate the figures and the modelled reference numbers:
python scripts/make_figures.py
python scripts/run_benchmark.py --simulate

# 4. Reproduce the calibration study:
python scripts/run_calibration_study.py
```

On the RX 7900 XTX with the ROCm stack:

```bash
pip install -r requirements-rocm.txt
python scripts/check_env.py                  # Phase 0 gate: GPU, Qwen3 forward, INT8 quant
python scripts/run_benchmark.py --hardware   # the real numbers
python demo/server.py --hardware             # the real side-by-side demo
```

## How it works

```
 accepted prefix + verifier hidden states
        │
   ┌────▼─────────────────────────────┐    drafter is INT8/NF4 and can be as wrong
   │ drafter: backbone → block logits │    as it likes — it only affects SPEED
   │          markov head → bias      │
   │          confidence head → a_j   │──► calibrated survival a_1..a_block
   └────┬─────────────────────────────┘
        │
   ┌────▼─────────────────────────────┐    policy only bounds HOW MANY tokens are
   │ policy: ℓ = longest prefix with  │    checked; never the accept/reject decision
   │         Π a_j ≥ θ  (θ = 0.45)    │
   └────┬─────────────────────────────┘
        │
   ┌────▼─────────────────────────────┐    the verifier owns every token:
   │ verifier: one forward pass       │    EXACT rejection-sampling / greedy match
   │           exact accept / reject  │──► output identical to the verifier alone
   └──────────────────────────────────┘
```

**The correctness invariant.** For a fixed seed, greedy EdgeSpark output is
token-for-token identical to the verifier decoding on its own — for *any* drafter
quality and *any* verification length. Stochastic decoding is distribution-identical
(the acceptance rule is provably unbiased). This is enforced, not hoped for:
[`tests/test_exactness.py`](tests/test_exactness.py) checks greedy identity under
every policy and runs a Monte-Carlo unbiasedness test at TV < 0.02.

Why the confidence head matters for *speed*, not just accuracy: it drives the
verification-length policy. An over-confident head lies to the scheduler about how
far to verify, so on a single stream, miscalibration costs throughput. That is why
recalibration and the policy are the same result — see
[docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md).

## Results

| Figure | What it shows |
|---|---|
| [Throughput](docs/assets/throughput.svg) | EdgeSpark vs. baseline, code + chat, per precision |
| [Reliability diagrams](docs/assets/reliability_diagrams.svg) | calibration damage and recovery — the money plot |
| [Policy ablation](docs/assets/policy_ablation.svg) | confidence-gated ℓ vs. always-verify-all |
| [VRAM breakdown](docs/assets/vram_breakdown.svg) | fitting 24 GB across precisions |

Full tables, provenance, and the reproduce-it commands: **[docs/RESULTS.md](docs/RESULTS.md)**.

## Repository layout

```
edgespark/
├── edgespark/            # the library
│   ├── loop/             # exact acceptance (numpy) + reference loop + torch loop
│   ├── drafter/          # backbone · markov head · confidence head · assembled model
│   ├── quantize/         # INT8 / NF4 (real bitsandbytes + fake-quant simulators)
│   ├── calibration/      # ECE · Brier · reliability · temperature/Platt recalibration
│   ├── policy/           # single-stream verification-length selection
│   └── utils/            # config · timing · VRAM · JSONL metrics · toy LM
├── train/                # distillation (Path A) + from-scratch (Path B) + losses
├── data/                 # prompt sets · pair generation · streamed hidden-state cache
├── bench/                # harness · baselines (llama.cpp, vanilla) · simulation · plots
├── demo/                 # live side-by-side dashboard (stdlib server + web UI)
├── scripts/              # check_env · run_benchmark · run_calibration_study · make_figures
├── configs/              # drafter / quant / policy / bench YAML
├── tests/                # exactness (mandatory) · calibration · policy · quantize · harness
└── docs/                 # RESULTS · TECHNICAL_REPORT · ARCHITECTURE · DEVLOG
```

## Hardware & software

| | |
|---|---|
| GPU | Radeon RX 7900 XTX · 24 GB · RDNA3 / gfx1100 |
| CPU / RAM | Ryzen 9 7900 (12c) · 32 GB (stream caches from SSD; never load whole) |
| Stack | ROCm 7.2.4 · PyTorch ROCm · HF Transformers · bitsandbytes-ROCm |
| Verifier | Qwen3-4B (fp16 or INT8/NF4); Qwen3-8B as a quantized stretch |
| Quantization | INT8 W8A8 · NF4 — **no FP8** (unsupported on RDNA3) |
| Attention | PyTorch SDPA by default; flash-attention (navi fork) optional |

## Scope

**In:** one verifier, a shrunk quantized semi-AR drafter, the calibration study,
a single-stream gated policy, a controlled PyTorch+ROCm loop, benchmark harness,
exactness suite, and the demo. **Out (and why):** fleet/concurrency scheduling
(needs multiple GPUs to mean anything), verifier retraining, FP8, and reproducing
DSpark's full ~38 TB training cache. See the spec for the full rationale.

## License

[MIT](LICENSE).
