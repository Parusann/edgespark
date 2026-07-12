# Architecture

## Data flow (one generation round)

```
                accepted prefix + verifier hidden states
                              │
          ┌───────────────────▼───────────────────┐
          │  EdgeSpark drafter (INT8 / NF4)        │
          │   backbone → block logits              │
          │   markov head → intra-block bias       │
          │   confidence head → a_j (calibrated)   │
          └───────────────────┬───────────────────┘
        drafted block, calibrated a_1..a_block
                              │
          ┌───────────────────▼───────────────────┐
          │  Verification-length policy (θ)        │
          │   ℓ = longest prefix with Πa_j ≥ θ     │
          └───────────────────┬───────────────────┘
                        chosen ℓ
                              │
          ┌───────────────────▼───────────────────┐
          │  Verifier (Qwen3-4B, unmodified)       │
          │   one forward pass over prefix+draft   │
          │   EXACT accept/reject (greedy | RS)    │
          └───────────────────┬───────────────────┘
              accepted prefix + verifier token
                              │
                     emit, log, repeat
```

The only component that can change the output is the verifier's acceptance rule.
The drafter and the policy change how *fast* you get there.

## Module map

| Path | Responsibility | Runs on |
|---|---|---|
| `edgespark/loop/acceptance.py` | exact greedy + speculative-sampling acceptance | numpy (CPU) |
| `edgespark/loop/reference.py` | numpy reference of the whole loop (executable spec) | numpy (CPU) |
| `edgespark/loop/generate.py` | the real PyTorch + ROCm inference loop | torch (GPU) |
| `edgespark/verifier.py` | HF wrapper, hidden-state hooks, `verify()` | torch (GPU) |
| `edgespark/drafter/backbone.py` | parallel block backbone + hidden-state fusion | torch (GPU) |
| `edgespark/drafter/markov_head.py` | low-rank intra-block bias | torch (GPU) |
| `edgespark/drafter/confidence_head.py` | per-position acceptance probability `a_j` | torch (GPU) |
| `edgespark/drafter/model.py` | assembled drafter + recalibrator hook | torch (GPU) |
| `edgespark/quantize/{int8,nf4}.py` | real bitsandbytes quant + fake-quant simulators | torch |
| `edgespark/calibration/metrics.py` | ECE, Brier, reliability curves | numpy (CPU) |
| `edgespark/calibration/recalibrate.py` | temperature / Platt scaling | numpy (CPU) |
| `edgespark/policy/verify_length.py` | single-stream ℓ selection | numpy (CPU) |
| `bench/` | harness, baselines, simulation, plots | mixed |
| `demo/` | live side-by-side dashboard | stdlib (+ torch on `--hardware`) |

## The CPU / GPU split

Everything that can be reasoned about as pure math, the acceptance rule,
calibration metrics, recalibration, the policy, is implemented in numpy and unit
tested without a GPU. The torch modules are the neural machinery that produces the
distributions those algorithms consume. This is deliberate:

- The correctness-critical code (acceptance, calibration) is testable, auditable,
  and CI-runnable on any machine, see the 18-check exactness suite.
- The numpy `reference.py` loop is the executable specification the torch
  `generate.py` loop must match, so "does the GPU loop still do the right thing?"
  has a concrete answer.

## Key interfaces

```python
class EdgeSparkDrafter:
    def draft(self, hidden_states, accepted_prefix) -> DraftBlock
        # DraftBlock(tokens[block], confidence[block] (calibrated), dist, raw_confidence)

class Verifier:
    def forward_with_hidden(self, tokens) -> (logits, hidden_states_selected, kv)
    def verify(self, prefix, drafted_block, ell) -> AcceptResult
        # exact rejection-sampling / greedy-match; ell only bounds length

class VerifyLengthPolicy:
    def choose_length(self, confidence_profile, recent_accept_ema) -> int
        # returns ell in [0, block_size]
```

Per-step telemetry is logged as JSON lines (`edgespark/utils/metrics_log.py`):

```json
{"step": 0, "ell": 3, "accepted": 2, "tau": 3, "t_draft_ms": 5.0,
 "t_verify_ms": 31.6, "vram_mb": 9650, "conf_profile": [0.94, 0.86, 0.71],
 "precision": "int8", "exact_ok": true}
```
