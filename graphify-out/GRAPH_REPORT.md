# Graph Report - C:\Users\parus\Projects\EdgeSpark  (2026-07-11)

## Corpus Check
- Corpus is ~31,191 words - fits in a single context window. You may not need a graph.

## Summary
- 632 nodes · 1125 edges · 43 communities (33 shown, 10 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 86 edges (avg confidence: 0.69)
- Token cost: 173,286 input · 0 output

## Community Hubs (Navigation)
- Calibration & Simulation
- Speculative Acceptance Loop
- Benchmark Harness
- Drafter Training & Config
- Generator & Metrics
- Drafter Model Heads
- Verification-Length Policy
- Figure Generation
- Architecture Docs
- Demo Server
- Exactness Invariant & CI
- EdgeSpark Core Concept
- Hardware Results Notes
- DEVLOG Phases
- Policy Simulation Params
- VRAM Breakdown Figure
- Environment Gate & VRAM
- Quantization Configs
- Calibration Reliability Figure
- Throughput Figure
- Policy Ablation Figure
- llama.cpp Baselines
- Dashboard Script
- Demo Dashboard UI
- VRAM Accounting
- Env Check Script
- Paper & Technical Report
- Distillation Corpus Generator
- Baseline Strategy Docs
- Training Integration Fixes
- Vanilla Decode Baseline
- Bench Scripts
- Baselines Package
- Bench Package
- Pytest Config
- Data Package
- Demo Package
- EdgeSpark Package
- Training Package
- Package Marker (init)
- Package Marker (init)
- Package Marker (init)

## God Nodes (most connected - your core abstractions)
1. `ThresholdPolicy` - 21 edges
2. `ToyCategoricalLM` - 19 edges
3. `Verifier` - 18 edges
4. `verify_block()` - 17 edges
5. `TemperatureScaler` - 16 edges
6. `EdgeSparkDrafter` - 16 edges
7. `Canvas` - 15 edges
8. `reliability_curve()` - 15 edges
9. `expected_calibration_error()` - 15 edges
10. `EdgeSparkGenerator` - 15 edges

## Surprising Connections (you probably didn't know these)
- `docs/TECHNICAL_REPORT.md` --semantically_similar_to--> `EdgeSpark: Quantized Semi-Autoregressive Speculative Decoding and Confidence-Head Calibration on a Single Consumer AMD GPU (paper/edgespark.pdf)`  [INFERRED] [semantically similar]
  docs/TECHNICAL_REPORT.md → paper/edgespark.pdf
- `Exactness Is Not Negotiable` --semantically_similar_to--> `Correctness invariant (§5)`  [INFERRED] [semantically similar]
  CONTRIBUTING.md → docs/TECHNICAL_REPORT.md
- `EdgeSpark` --cites--> `EdgeSpark: Quantized Semi-Autoregressive Speculative Decoding and Confidence-Head Calibration on a Single Consumer AMD GPU (paper/edgespark.pdf)`  [EXTRACTED]
  README.md → paper/edgespark.pdf
- `KV-reuse fix (flagged in generate.py)` --semantically_similar_to--> `KV-reuse optimization`  [INFERRED] [semantically similar]
  README.md → bench/timings.md
- `paper/README.md` --semantically_similar_to--> `docs/TECHNICAL_REPORT.md`  [INFERRED] [semantically similar]
  paper/README.md → docs/TECHNICAL_REPORT.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Draft → Gate → Verify generation-round pipeline** — readme_drafter, readme_policy, readme_verifier [EXTRACTED 1.00]
- **Quantization, confidence-head calibration, and policy performance are coupled** — readme_confidence_head, readme_quantization, readme_policy [INFERRED 0.85]
- **2026-07-05 hardware-run evidence chain (env, notes, results)** — runs_hardware_env_hardwareenvdoc, runs_hardware_notes_hardwarenotesdoc, docs_results_hardware_validation_table [EXTRACTED 1.00]
- **Per-precision calibration evaluation: all three precisions evaluated with the same ECE metric and temperature-scaling recalibration on n=30,000 held-out positions each** — docs_assets_reliability_diagrams_fp16, docs_assets_reliability_diagrams_int8, docs_assets_reliability_diagrams_nf4, docs_assets_reliability_diagrams_expectedcalibrationerror, docs_assets_reliability_diagrams_temperaturescaling [EXTRACTED 1.00]
- **** — docs_assets_vram_breakdown_fp16_verifier_int8_drafter, docs_assets_vram_breakdown_int8_verifier_int8_drafter, docs_assets_vram_breakdown_int8_verifier_nf4_drafter, docs_assets_vram_breakdown_vram_budget_24gb [EXTRACTED 1.00]

## Communities (43 total, 10 thin omitted)

### Community 0 - "Calibration & Simulation"
Cohesion: 0.07
Nodes (58): CalibrationResult, _logit(), A reproducible model of the EdgeSpark pipeline.  Real tokens/sec, VRAM, and cali, Generate (confidence, outcome) pairs for a precision and measure/repair them., Model end-to-end tok/s for one configuration.      The verification length is ch, Model the 24 GB VRAM breakdown (spec section 11 feasibility numbers)., Assemble every modelled result the figures and RESULTS.md consume., run_all() (+50 more)

### Community 1 - "Speculative Acceptance Loop"
Cohesion: 0.06
Nodes (45): AcceptResult, _as_prob_block(), greedy_accept(), ndarray, Exact speculative-decoding acceptance.  This module is the load-bearing wall of, Stochastic speculative-sampling acceptance (unbiased w.r.t. the verifier)., Verify the first ``ell`` tokens of a drafted block.      This is the single entr, Outcome of verifying one drafted block.      Attributes     ----------     token (+37 more)

### Community 2 - "Benchmark Harness"
Cohesion: 0.06
Nodes (39): BenchReport, _build_drafter(), _measure_baseline(), _now(), Benchmark harness (spec sections 9.7, 12).  Replays fixed prompt sets single-str, Vanilla autoregressive decode tok/s — baseline 1 from section 12., Exercise the full loop with toy LMs and assert exactness holds.      Returns a s, Full hardware benchmark. Imports torch lazily; raises if unavailable. (+31 more)

### Community 3 - "Drafter Training & Config"
Cohesion: 0.07
Nodes (39): build(), main(), Build a small, streamed selected-layer hidden-state cache (spec section 10).  Th, Yield ``(hidden, ids)`` per cached item, one shard at a time (never all in RAM)., stream_cache(), load_prompt_sets(), Evaluation prompt sets (spec sections 10, 12).  Small, fixed, single-stream prom, Return ``{set_name: [prompt, ...]}`` for the requested sets. (+31 more)

### Community 4 - "Generator & Metrics"
Cohesion: 0.06
Nodes (22): accept_over_time(), matplotlib plotting for real hardware runs (spec section 12).  The committed REA, pairs_by_precision: {precision: (confidence, outcome, recalibrated)}., reliability_plot(), EdgeSparkGenerator, GenerationOutput, The controlled PyTorch + ROCm inference loop (spec section 9.6).  This is the re, Speculative generation with a confidence-gated verification length. (+14 more)

### Community 5 - "Drafter Model Heads"
Cohesion: 0.11
Nodes (18): HiddenFusion, ParallelBackbone, Parallel block backbone.  Predicts an entire ``block_size`` block of tokens in o, Projects each selected verifier layer to the draft width and sums them., _torch(), ConfidenceHead, Confidence head: predicted per-position acceptance probability ``a_j``.  This is, _torch() (+10 more)

### Community 6 - "Verification-Length Policy"
Cohesion: 0.13
Nodes (20): Single-stream verification-length policy (spec section 9.5)., expected_accepted_length(), optimal_length(), ndarray, How many drafted tokens are worth verifying on a single stream.  DSpark's datace, Cumulative survival S_j = prod_{k<=j} a_k for each position j., E[accepted(ell)] = 1 + sum_{j=1..ell} prod_{k<=j} a_k  (spec Appendix B).      T, Cost-aware optimum: argmax over ell of throughput per unit wall-clock.      Maxi (+12 more)

### Community 7 - "Figure Generation"
Cohesion: 0.25
Nodes (11): Canvas, A tiny dependency-free SVG chart primitive set.  matplotlib is the right tool fo, save(), _axes(), main(), policy_figure(), Path, Regenerate the committed SVG figures from the modelled results.      python scri (+3 more)

### Community 8 - "Architecture Docs"
Cohesion: 0.16
Nodes (18): markov_rank: 128, quantize_confidence_head: false, docs/ARCHITECTURE.md, edgespark/drafter/backbone.py, edgespark/drafter/confidence_head.py, EdgeSparkDrafter class, edgespark/drafter/markov_head.py, edgespark/drafter/model.py (+10 more)

### Community 9 - "Demo Server"
Cohesion: 0.19
Nodes (8): BaseHTTPRequestHandler, HardwareDemo, CpuDemo, Handler, main(), EdgeSpark live demo server (spec section 9.8).  The report's ideal demo: *same l, Toy-LM engine that runs anywhere and mimics the measured latency model., _word()

### Community 10 - "Exactness Invariant & CI"
Cohesion: 0.13
Nodes (15): edgespark/loop/acceptance.py, Exactness Is Not Negotiable, pyproject.toml, pytest, ruff (lint tool), tests/test_exactness.py, edgespark/loop/acceptance.py, CPU/GPU split design (+7 more)

### Community 11 - "EdgeSpark Core Concept"
Cohesion: 0.17
Nodes (15): KV-reuse optimization, block_size: 5 (DSpark-5), DFlash, Comparison table: vanilla / Medusa / EAGLE-3 / DSpark / EdgeSpark, EAGLE-3, Medusa, demo/server.py, DSpark (DeepSeek) (+7 more)

### Community 12 - "Hardware Results Notes"
Cohesion: 0.18
Nodes (14): bench/timings.md, loop/generate.py, Verifier.block_distribution, edgespark/loop/generate.py, edgespark/utils/metrics_log.py, edgespark/loop/reference.py, runs/hardware/baselines.json, Hardware validation status (§0) (+6 more)

### Community 13 - "DEVLOG Phases"
Cohesion: 0.21
Nodes (14): confidence_head_alpha: 1.0, Drafter config (drafter_qwen3_4b.yaml), Training loss weights (ce/l1/decay), num_draft_layers: 3, target_layer_ids: [1,17,33], docs/DEVLOG.md, Phase 2 — DSpark-style drafter (fp16), Phase 3 — Quantize the drafter (+6 more)

### Community 14 - "Policy Simulation Params"
Cohesion: 0.15
Nodes (13): dT_verify/dℓ marginal cost, GpuTimer (edgespark/utils/timing.py), bench/simulate.py, T_decode constant, T_draft constant (fp16/int8/nf4), T_verify(0) constant, Benchmark harness config (bench.yaml), prompt_sets: [chat, code] (+5 more)

### Community 15 - "VRAM Breakdown Figure"
Cohesion: 0.27
Nodes (12): Config: fp16 Verifier + INT8 Drafter (12.9 GB), INT8 Weight Quantization, Config: INT8 Verifier + INT8 Drafter (9.4 GB), Config: INT8 Verifier + NF4 Drafter (9.1 GB), KV Cache at ~8k Context (2.5 GB), NF4 Weight Quantization, Insight: All Configs Fit Under 24 GB with Large Headroom; INT8 Verifier Nearly Halves Footprint (12.9 to 9.4 GB), Qwen3-4B Model (+4 more)

### Community 16 - "Environment Gate & VRAM"
Cohesion: 0.17
Nodes (12): scripts/check_env.py, howiejay/navi_support flash-attn fork, Path A (distillation) vs Path B (from-scratch), Phase 0 — Environment + verification gate, runs/hardware/vram.json, VRAM results, scripts/check_env.py, Radeon RX 7900 XTX (+4 more)

### Community 17 - "Quantization Configs"
Cohesion: 0.27
Nodes (10): precisions: [fp16, int8, nf4], bitsandbytes backend, INT8 W8A8 config (quant_int8.yaml), NF4 4-bit config (quant_nf4.yaml), double_quant: true (nested quant), Calibration results (ECE table), bitsandbytes (ROCm build), flash-attention (navi fork) (+2 more)

### Community 18 - "Calibration Reliability Figure"
Cohesion: 0.31
Nodes (11): Observed Acceptance Rate (n=30,000 held-out positions per precision), Confidence Head, Expected Calibration Error (ECE), Reliability Diagrams Figure: Confidence-Head Reliability, Quantization vs. Recalibration, FP16 Precision (ECE 0.006 -> 0.003, T=1.03), INT8 Quantization (ECE 0.097 -> 0.006, T=1.81), NF4 Quantization (ECE 0.166 -> 0.014, T=2.70), Quantization-Induced Overconfidence: miscalibration grows with quantization aggressiveness (T needed: 1.03 -> 1.81 -> 2.70) but temperature scaling restores near-perfect calibration (+3 more)

### Community 19 - "Throughput Figure"
Cohesion: 0.40
Nodes (10): bench/simulate.py Throughput Simulator, Design-Time Modelled Projection (Not Measured), Drafter Precision vs Throughput Trade-off (FP16 94/87 > INT8 92/88 > NF4 88/84 tok/s), EdgeSpark Gated-Verification Speculative Decoding, End-to-end Throughput Figure: EdgeSpark vs Vanilla Quantized Baseline, Output Identical to Baseline (Lossless Guarantee), Projected +30% to +46% Throughput Gain over Baseline, Qwen3-4B INT8 Verifier Model (+2 more)

### Community 20 - "Policy Ablation Figure"
Cohesion: 0.33
Nodes (10): Accepted Tokens per Verifier Call (tau), Always-Verify-All Policy (l=5), Confidence-Gated Verification-Length Policy (gated l), Drafter Quantization Levels (FP16 / INT8 / NF4), Gating Stops Before Low-Survival Tail Wastes Verifier Time, Policy Ablation Figure: Gated vs Always-Verify-All, Heavier Drafter Quantization Lowers Acceptance and Shortens Optimal Verification Length, docs/RESULTS.md Section 0 (Referenced Results Doc) (+2 more)

### Community 21 - "llama.cpp Baselines"
Cohesion: 0.39
Nodes (8): baseline_no_speculation(), baseline_vanilla_speculative(), LlamaResult, _parse_tok_s(), llama.cpp baselines and sanity oracle (spec sections 6, 9.6, 12).  llama.cpp has, Baseline 1: plain decode on the quantized verifier., Baseline 2: llama.cpp built-in speculative decoding with a separate draft model., _run()

### Community 22 - "Dashboard Script"
Cohesion: 0.39
Nodes (8): $(), appendTokens(), drawRound(), reset(), sources, speed, stream(), updateSpeedup()

### Community 23 - "Demo Dashboard UI"
Cohesion: 0.25
Nodes (9): app.js, demo/dashboard/index.html, Draft depth & accepted length panel, Scoreboard UI (vanilla/EdgeSpark/speedup/output match), style.css, demo/README.md, Draft-depth / accepted-length strip, EdgeSparkGenerator (+1 more)

### Community 24 - "VRAM Accounting"
Cohesion: 0.28
Nodes (5): VRAM accounting for the 24 GB budget (spec sections 6, 11).  Everything — verifi, Best-effort whole-device VRAM usage via rocm-smi; 0.0 if unavailable., _rocm_smi_used_mb(), snapshot(), VramSnapshot

### Community 25 - "Env Check Script"
Cohesion: 0.39
Nodes (8): _bitsandbytes(), _check(), main(), _qwen_forward(), Phase 0 environment gate (spec section 16).  Confirms the pieces EdgeSpark is bu, _rocm_smi(), _torch_gpu(), _transformers()

### Community 26 - "Paper & Technical Report"
Cohesion: 0.29
Nodes (8): Limitations and honesty (§7), docs/TECHNICAL_REPORT.md, The Thesis (quantization degrades calibration faster than proposals), EdgeSpark: Quantized Semi-Autoregressive Speculative Decoding and Confidence-Head Calibration on a Single Consumer AMD GPU (paper/edgespark.pdf), paper/edgespark.tex, Evidence ledger (Table 2 / §6 / §8), paper/README.md, refs.bib

### Community 27 - "Distillation Corpus Generator"
Cohesion: 0.43
Nodes (6): generate(), main(), Path, Generate the distillation corpus: prompt -> response pairs from the verifier.  D, A cheap prompt generator: cycle the built-in sets with index suffixes.      Repl, _seed_prompts()

### Community 28 - "Baseline Strategy Docs"
Cohesion: 0.33
Nodes (7): Phase 1 — Baselines, Why a controlled PyTorch loop, not llama.cpp (§6), llama.cpp, Qwen3-4B, scripts/run_benchmark.py, llama.cpp --draft flag drift (build b9873), llama.cpp Vulkan build (winget ggml.llamacpp)

### Community 29 - "Training Integration Fixes"
Cohesion: 0.40
Nodes (5): train/train_drafter.py::_batches, Integration gap fix: block_hidden / regress head, data/build_cache.py, train/distill.py::_distill_batches / load_teacher, train/losses.py

### Community 30 - "Vanilla Decode Baseline"
Cohesion: 0.67
Nodes (3): decode(), Vanilla PyTorch decode baseline (spec section 12, baseline 1).  The same verifie, VanillaResult

### Community 31 - "Bench Scripts"
Cohesion: 0.67
Nodes (3): bench/simulate.py, scripts/make_figures.py, scripts/run_benchmark.py

## Ambiguous Edges - Review These
- `Verification-Length Policy` → `KV-reuse fix (flagged in generate.py)`  [AMBIGUOUS]
  README.md · relation: rationale_for
- `Accepted Tokens per Verifier Call (tau)` → `Speculative Decoding (Drafter/Verifier Scheme)`  [AMBIGUOUS]
  docs/assets/policy_ablation.svg · relation: shares_data_with
- `Confidence Head` → `Speculative Decoding Draft Acceptance`  [AMBIGUOUS]
  docs/assets/reliability_diagrams.svg · relation: conceptually_related_to
- `KV Cache at ~8k Context (2.5 GB)` → `Speculative Decoding (Verifier + Drafter)`  [AMBIGUOUS]
  docs/assets/vram_breakdown.svg · relation: shares_data_with

## Knowledge Gaps
- **42 isolated node(s):** `sources`, `speed`, `edgespark`, `pytest`, `scripts/make_figures.py` (+37 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **10 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Verification-Length Policy` and `KV-reuse fix (flagged in generate.py)`?**
  _Edge tagged AMBIGUOUS (relation: rationale_for) - confidence is low._
- **What is the exact relationship between `Accepted Tokens per Verifier Call (tau)` and `Speculative Decoding (Drafter/Verifier Scheme)`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._
- **What is the exact relationship between `Confidence Head` and `Speculative Decoding Draft Acceptance`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `KV Cache at ~8k Context (2.5 GB)` and `Speculative Decoding (Verifier + Drafter)`?**
  _Edge tagged AMBIGUOUS (relation: shares_data_with) - confidence is low._
- **Why does `ThresholdPolicy` connect `Verification-Length Policy` to `Calibration & Simulation`, `Speculative Acceptance Loop`, `Benchmark Harness`, `Drafter Training & Config`, `Demo Server`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Why does `EdgeSparkDrafter` connect `Drafter Model Heads` to `Benchmark Harness`, `Drafter Training & Config`?**
  _High betweenness centrality (0.069) - this node is a cross-community bridge._
- **Why does `EdgeSparkGenerator` connect `Generator & Metrics` to `Demo Server`, `Benchmark Harness`, `Drafter Training & Config`?**
  _High betweenness centrality (0.048) - this node is a cross-community bridge._