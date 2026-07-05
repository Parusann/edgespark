# EdgeSpark hardware run — NOTES

Run date: 2026-07-05. Machine: AMD Radeon RX 7900 XTX (gfx1100, 48 CU, 24 GB),
Ryzen 9 7900, 32 GB. Stack: **native-Windows ROCm** — PyTorch 2.9.1+rocmsdk20260116
(HIP 7.2.26024), transformers 5.3.0. Branch: `hardware-run`.

Scope executed this session: **Tier 0 (baselines / latency / VRAM) + the code work**
(stubs, integration gap, shape verification). Tier 1 (drafter training, INT8/NF4
quant, full benchmark) and Tier 2 (calibration study) were intentionally not run —
see "Deferred".

## Environment note: native-Windows ROCm, not Linux ROCm

The handoff assumed a Linux ROCm box. The target machine runs the ROCm build of
PyTorch on **native Windows** (no WSL ROCm distro; the only WSL present is
docker-desktop). Almost everything mapped cleanly:

- GPU compute, Qwen3-4B fp16 forward, and `torch.cuda` HIP-event timers all work.
- `rocm-smi` / `rocminfo` are Linux-only — VRAM is read via `torch.cuda.mem_get_info`
  and the device via `torch.cuda.get_device_properties` / Windows CIM instead.
- `bitsandbytes` INT8/NF4 is **not available** on this stack — see Deferred.
- The llama.cpp baseline uses the winget `ggml.llamacpp` **Vulkan** build (not HIP)
  pinned to the 7900 XTX (`GGML_VK_VISIBLE_DEVICES=0`).

## Path A vs Path B

Not decided this session (it belongs to Tier 1). No DeepSpec DSpark/EAGLE-3
checkpoint was loaded. `train/distill.py::load_teacher` now accepts a local `.pt`
in EdgeSpark's own layout (`{"state_dict", "config"}`) and raises a clear
"fall back to Path B" error if none is present, so either path is ready. With the
drafter forward / generate loop now exact on-device (below), **Path B
(`train_drafter.py`) is the safe default**; try Path A only if a compatible
checkpoint loads on this ROCm build.

## Code: implemented / fixed (GPU smoke-tested; numpy suite green — 41 passed)

1. **Integration gap (block_hidden).** `EdgeSparkDrafter.forward` returned only
   `(logits, conf_logit)`, starving `drafter_loss`'s L1 term. It now returns
   `(logits, conf_logit, block_feature)`. Because the drafter is shrunk
   (`draft_hidden` <= 1024) while the L1 target is the verifier's `Hv`=2560 hidden,
   I added a `regress: Linear(draft_hidden, Hv)` head so the block feature matches
   `target_hidden`'s width (EAGLE-style feature distillation). Threaded through
   `model.draft`, `train_drafter.train`, and `distill.distill`.
2. **`train/train_drafter.py::_batches`** implemented: streams the cache one shard
   at a time and slides a block window (ctx=1, matching the inference loop) to yield
   `{hidden_by_layer, prefix_last, target_tokens, target_hidden}`.
3. **`train/distill.py::_distill_batches`** shares that collator;
   **`load_teacher`** implemented (see Path A/B).
4. **`data/build_cache.py`** now stores token ids beside the hidden states
   (`hidden_i` + `ids_i` per shard) — the collator needs `prefix_last` /
   `target_tokens`, which the hidden-only cache did not carry. `stream_cache`
   yields `(hidden, ids)`.
5. **`train/losses.py`** made dtype-robust under bf16 autocast (cast `target_hidden`
   and the accept label to the prediction dtype) and `.detach()`ed the logging
   scalars.
6. **fp16 GradScaler** in `train_drafter` gated to fp16 only (bf16 needs no scaler).

### Flagged shape bugs — all verified CORRECT on-device (no fix needed)

- `out.hidden_states[layer_id]` for `(1,17,33)` -> correct `[1, seq, 2560]` on the
  36-layer Qwen3-4B.
- `ConfidenceHead` `in_dim` resolves to `2*draft_hidden` with Markov — correct.
- `verifier.block_distribution` returns exactly `[ell+1, vocab]` for ell=1..5.
- `loop/generate.py` end-to-end (KV reuse + the double verifier pass) runs and the
  greedy exactness oracle holds (`exact_ok=true`) with a random drafter — output is
  identical to the verifier alone regardless of draft quality.

## Tier 0 — measured (single stream)

| Quantity | Modelled | Measured | Source |
|---|---|---|---|
| Plain decode, GGUF Q4_K_M | — | 213.8 tok/s | llama.cpp Vulkan (llama-bench) |
| Plain decode, GGUF Q8_0 | — | 151.0 tok/s | llama.cpp Vulkan |
| Speculative Q8_0 + 0.6B draft | — | 145.5 tok/s (~0.96x) | llama.cpp |
| T_decode (fp16, per forward) | 15.5 ms | 2.9 ms | GpuTimer |
| T_verify0 (KV-cached block) | 19.0 ms | 3.2 ms | GpuTimer |
| dT_verify/dell | 4.2 ms | ~0 ms | GpuTimer |
| T_draft (fp16, 399M) | 6.5 ms | 1.0 ms | GpuTimer |
| VRAM verifier / drafter / peak (fp16) | 8200 / 1850 / — | 7672 / 1773 / 9540 MB | torch.cuda |

Constants overwritten in `bench/simulate.py`; details + caveats in
`bench/timings.md`. Headline hardware findings:

- **Vanilla speculative decoding does not pay here** (~0.96x): the 0.6B draft's
  overhead isn't offset — the floor EdgeSpark's smarter drafter must beat.
- **The per-position verify marginal is ~0**, so the confidence-gated
  verification-length policy is roughly tied with always-verify-all (the
  `dT_verify/dell -> 0` outcome `timings.md` anticipated). Still exactness-preserving.
- **The current `block_distribution` re-encodes the full prefix** (`use_cache=False`),
  measured ~52.6 ms + 0.37 ms/ell at 225-token context — prefill-bound and growing
  with context. The KV-reuse optimisation flagged in `generate.py` is required for
  end-to-end throughput to approach the 3.2 ms intrinsic verify.

## Deferred / shortfalls (documented, per Phase-6 policy)

- **INT8 / NF4 (bitsandbytes).** Unavailable on native-Windows ROCm; the Phase-0
  gate MISSes on it. So no INT8/NF4 drafter or verifier, no quant VRAM, and the
  Tier-2 "quantization miscalibration" headline could not be produced on this stack.
  Options: a Linux ROCm box for bitsandbytes, or an alternate quant path
  (torchao / GPTQ / AWQ / GGUF) — the latter changes what "INT8/NF4" means and
  should be relabelled.
- **Tier 1 training + Tier 2 calibration** not run (scope = Tier 0 + code). Stubs,
  integration gap, and the generate loop are ready; `run_benchmark.py --hardware`
  and `run_calibration_study.py` need a trained drafter.
- **llama.cpp flag drift.** Build b9873 removed `--draft`; use `--spec-draft-n-max`.
  `bench/baselines/llama_cpp.py` still emits `--draft` and should be updated. Also
  `llama-cli` needs `-st` and a closed stdin to run single-shot non-interactively on
  Windows.
- **T_decode methodology.** The 2.9 ms / 343 tok/s figure is a single-forward,
  short-context latency (used as the per-op constant). The llama.cpp sustained
  tg128 rates (151–214 tok/s) are the more representative end-user decode speeds.

## Reproduce

- Env gate: `python scripts/check_env.py` (PASS except bitsandbytes).
- Committed artifacts: `runs/hardware/{env.txt, baselines.json, vram.json}` + this
  file. `runs/hardware/metrics/` is empty (Tier-1 JSONL not produced this session).
