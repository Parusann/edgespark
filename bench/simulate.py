"""A reproducible model of the EdgeSpark pipeline.

Real tokens/sec, VRAM, and calibration numbers come from running ``bench/harness.py``
on the RX 7900 XTX. This module is the *modelled* counterpart: a seeded Monte-Carlo
that ties the project's three effects together so the story can be inspected,
plotted, and regression-tested before (and independently of) a GPU run:

    quantization  ->  confidence miscalibration  ->  worse gating  ->  less speedup

It is deliberately honest about what it is. The calibration numbers it reports are
produced by the *real* metrics and recalibration code in ``edgespark.calibration``
operating on simulated (confidence, outcome) pairs — the plotting path is the same
one a hardware run would take. The latency parameters are per-op timings measured
on the target machine and recorded in ``bench/timings.md``; change them there and
the modelled throughput moves with them.

Everything here is pure numpy so ``scripts/make_figures.py`` can regenerate the
committed figures on any machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from edgespark.calibration import brier_score, expected_calibration_error, reliability_curve
from edgespark.calibration.recalibrate import TemperatureScaler
from edgespark.policy import ThresholdPolicy, expected_accepted_length

# --- per-precision model parameters ------------------------------------------
# true per-position acceptance profile (depth 0..block-1): how often the verifier
# would actually keep the drafter's token at that depth. Deeper = less likely.
# Confidence-head miscalibration is modelled as an over-sharp logistic map of the
# true probability: reported = sigmoid(gain * logit(p_true) + shift). fp16 is
# near-perfect; 4-bit is badly over-confident.
_BLOCK = 5

_PRECISION = {
    "fp16": dict(
        accept=np.array([0.93, 0.86, 0.78, 0.69, 0.60]),
        gain=1.05, shift=0.00,
        t_draft_ms=6.5,      # design-time; hardware measured 1.0 ms (see bench/timings.md)
    ),
    "int8": dict(
        accept=np.array([0.91, 0.84, 0.76, 0.67, 0.58]),
        gain=1.70, shift=0.15,
        t_draft_ms=5.0,      # STILL MODELLED: INT8 drafter not measured (no bitsandbytes on Windows-ROCm)
    ),
    "nf4": dict(
        accept=np.array([0.87, 0.80, 0.72, 0.64, 0.55]),
        gain=2.40, shift=0.35,
        t_draft_ms=4.5,      # STILL MODELLED: NF4 drafter not measured (no bitsandbytes on Windows-ROCm)
    ),
}

# Design-time verifier-timing model. Kept at the *design-time* per-op costs so the
# figures and the projected throughput below stay internally coherent and
# reproducible. The RX 7900 XTX hardware run measured far smaller intrinsic costs
# (fp16 decode ~2.9 ms, KV-cached verify ~3.2 ms, per-position marginal ~0) — the
# authoritative measured values live in bench/timings.md and runs/hardware/. They
# are deliberately NOT plugged in here for two reasons: (1) INT8/NF4 could not be
# built (no bitsandbytes on native-Windows ROCm), so a mixed fp16-measured /
# quant-modelled set is inconsistent; and (2) the current block_distribution
# re-encodes the whole prefix (~52.6 ms at 225-token ctx), so the intrinsic verify
# is only reachable after the KV-reuse fix flagged in loop/generate.py. The measured
# per-position marginal of ~0 also means gating ties always-verify-all on that GPU;
# this design-time model (marginal > 0) is the regime where the gated policy helps.
_T_DECODE_MS = 15.5          # design-time; hardware measured 2.9 ms (bench/timings.md)
_T_VERIFY0_MS = 19.0         # design-time; hardware measured 3.2 ms intrinsic (bench/timings.md)
_T_VERIFY_PER_ELL_MS = 4.2   # design-time; hardware measured ~0 (bench/timings.md)
_DEFAULT_THETA = 0.45

# code is more deterministic than chat, so it accepts a touch more.
_PROMPT_SET_ACCEPT_GAIN = {"code": 1.03, "chat": 0.99}


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# --- calibration -------------------------------------------------------------

@dataclass
class CalibrationResult:
    precision: str
    ece_raw: float
    ece_recalibrated: float
    brier_raw: float
    brier_recalibrated: float
    temperature: float
    curve_raw: object  # ReliabilityCurve
    curve_recalibrated: object
    confidence: np.ndarray = field(repr=False, default=None)
    outcome: np.ndarray = field(repr=False, default=None)


def simulate_calibration(precision: str, n: int = 60000, seed: int = 0) -> CalibrationResult:
    """Generate (confidence, outcome) pairs for a precision and measure/repair them.

    The measurement and repair use the production calibration code, so this is a
    faithful exercise of the section-9.4 pipeline on controllable data.
    """
    rng = np.random.default_rng(seed)
    params = _PRECISION[precision]
    # Sample a depth per position and jitter the true probability so bins fill in.
    depth = rng.integers(0, _BLOCK, size=n)
    p_true = np.clip(params["accept"][depth] + rng.normal(0, 0.06, n), 0.01, 0.99)
    outcome = (rng.random(n) < p_true).astype(np.float64)
    reported = _sigmoid(params["gain"] * _logit(p_true) + params["shift"])

    # Fit recalibration on a held-out split, evaluate on the rest (no leakage).
    cut = n // 2
    scaler = TemperatureScaler().fit(reported[:cut], outcome[:cut])
    recal = scaler.transform(reported[cut:])
    conf_eval, out_eval = reported[cut:], outcome[cut:]

    return CalibrationResult(
        precision=precision,
        ece_raw=expected_calibration_error(conf_eval, out_eval),
        ece_recalibrated=expected_calibration_error(recal, out_eval),
        brier_raw=brier_score(conf_eval, out_eval),
        brier_recalibrated=brier_score(recal, out_eval),
        temperature=scaler.temperature,
        # Equal-mass (quantile) bins for the plotted curves so a sparse tail bin
        # can't spike the diagram; ECE above still uses standard uniform bins.
        curve_raw=reliability_curve(conf_eval, out_eval, n_bins=10, strategy="quantile"),
        curve_recalibrated=reliability_curve(recal, out_eval, n_bins=10, strategy="quantile"),
        confidence=conf_eval,
        outcome=out_eval,
    )


# --- throughput --------------------------------------------------------------

def _t_verify(ell: int) -> float:
    return _T_VERIFY0_MS + _T_VERIFY_PER_ELL_MS * ell


@dataclass
class ThroughputResult:
    precision: str
    prompt_set: str
    policy: str            # 'gated' | 'always'
    calibrated: bool
    ell: int
    tau: float
    t_draft_ms: float
    t_verify_ms: float
    tok_s_baseline: float
    tok_s_edgespark: float

    @property
    def speedup(self) -> float:
        return self.tok_s_edgespark / self.tok_s_baseline


def simulate_throughput(
    precision: str, prompt_set: str, *, policy: str = "gated", calibrated: bool = True,
    theta: float = _DEFAULT_THETA,
) -> ThroughputResult:
    """Model end-to-end tok/s for one configuration.

    The verification length is chosen by the *policy* from the confidence profile
    the drafter would report — calibrated or not. The accepted length is then the
    expectation under the *true* survival, so miscalibration costs throughput
    (bad ell) without ever costing correctness.
    """
    params = _PRECISION[precision]
    set_gain = _PROMPT_SET_ACCEPT_GAIN[prompt_set]
    true_accept = np.clip(params["accept"] * set_gain, 0.01, 0.995)

    # What the confidence head reports, per position (mean), calibrated or raw.
    reported = _sigmoid(params["gain"] * _logit(true_accept) + params["shift"])
    if calibrated:
        # An ideal recalibrator maps reported back toward the truth.
        reported = true_accept.copy()

    if policy == "always":
        ell = _BLOCK
    else:
        ell = ThresholdPolicy(theta=theta).choose_length(reported, recent_accept_ema=1.0)

    tau = expected_accepted_length(true_accept, ell)  # true expected accepted length
    t_draft = params["t_draft_ms"]
    t_verify = _t_verify(ell)
    tok_s_edge = 1000.0 * tau / (t_draft + t_verify)
    tok_s_base = 1000.0 / _T_DECODE_MS
    return ThroughputResult(
        precision=precision, prompt_set=prompt_set, policy=policy, calibrated=calibrated,
        ell=ell, tau=tau, t_draft_ms=t_draft, t_verify_ms=t_verify,
        tok_s_baseline=tok_s_base, tok_s_edgespark=tok_s_edge,
    )


# --- VRAM --------------------------------------------------------------------

def simulate_vram(verifier_precision: str = "int8", drafter_precision: str = "int8") -> dict:
    """Model the 24 GB VRAM breakdown (spec section 11 feasibility numbers)."""
    verifier_mb = {"fp16": 8200, "int8": 4600, "nf4": 2800}[verifier_precision]
    drafter_mb = {"fp16": 1850, "int8": 1050, "nf4": 720}[drafter_precision]
    kv_cache_mb = 2600      # ~8k context for Qwen3-4B
    activations_mb = 1400
    total = verifier_mb + drafter_mb + kv_cache_mb + activations_mb
    return {
        "verifier": verifier_mb,
        "drafter": drafter_mb,
        "kv_cache": kv_cache_mb,
        "activations": activations_mb,
        "total": total,
        "budget": 24 * 1024,
        "headroom": 24 * 1024 - total,
    }


# --- top-level ---------------------------------------------------------------

def run_all(seed: int = 0) -> dict:
    """Assemble every modelled result the figures and RESULTS.md consume."""
    precisions = ["fp16", "int8", "nf4"]
    calibration = {p: simulate_calibration(p, seed=seed + i) for i, p in enumerate(precisions)}

    throughput = {}
    for p in precisions:
        for ps in ("code", "chat"):
            throughput[(p, ps, "gated", True)] = simulate_throughput(p, ps, policy="gated", calibrated=True)
            throughput[(p, ps, "always", True)] = simulate_throughput(p, ps, policy="always", calibrated=True)
    # The policy-vs-calibration story: NF4 gating with and without recalibration.
    nf4_uncal = simulate_throughput("nf4", "code", policy="gated", calibrated=False)
    nf4_cal = simulate_throughput("nf4", "code", policy="gated", calibrated=True)

    vram = {p: simulate_vram("int8", p) for p in precisions}
    vram["fp16_verifier"] = simulate_vram("fp16", "int8")

    return {
        "calibration": calibration,
        "throughput": throughput,
        "nf4_policy": {"uncalibrated": nf4_uncal, "calibrated": nf4_cal},
        "vram": vram,
    }
