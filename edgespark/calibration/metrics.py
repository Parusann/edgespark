"""Calibration metrics for the confidence head.

The confidence head predicts, per drafted position, the probability ``a_j`` that
the token will be accepted by the verifier. "Calibrated" means: of all the
positions where the head said 0.7, close to 70% were actually accepted. The
central empirical claim of EdgeSpark is that quantization damages *this* property
faster than it damages the head's token proposals, and that the damage is
recoverable (``recalibrate.py``).

Everything here is pure ``numpy`` and takes two aligned 1-D arrays:

* ``confidence``, predicted acceptance probabilities in ``[0, 1]``.
* ``outcome``, the observed accept signal in ``{0, 1}`` from the real verifier.

These pairs are produced by running the drafter+verifier loop and logging, for
each drafted position, the head's ``a_j`` against whether the exact acceptance
rule actually kept the token.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate(confidence: np.ndarray, outcome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    confidence = np.asarray(confidence, dtype=np.float64).ravel()
    outcome = np.asarray(outcome, dtype=np.float64).ravel()
    if confidence.shape != outcome.shape:
        raise ValueError(
            f"confidence {confidence.shape} and outcome {outcome.shape} must align"
        )
    if confidence.size == 0:
        raise ValueError("need at least one (confidence, outcome) pair")
    if np.any((confidence < -1e-9) | (confidence > 1 + 1e-9)):
        raise ValueError("confidence values must lie in [0, 1]")
    uniq = np.unique(outcome)
    if not np.all(np.isin(uniq, (0.0, 1.0))):
        raise ValueError("outcome values must be 0 or 1")
    return np.clip(confidence, 0.0, 1.0), outcome


def _bin_edges(n_bins: int, confidence: np.ndarray, strategy: str) -> np.ndarray:
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        # Equal-mass bins. Robust when confidences pile up in a narrow band, which
        # is exactly what an over-confident quantized head tends to do.
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(confidence, qs)
        edges[0], edges[-1] = 0.0, 1.0
        return np.unique(edges)
    raise ValueError(f"unknown strategy {strategy!r}; expected 'uniform' or 'quantile'")


@dataclass
class ReliabilityCurve:
    """Per-bin summary used both for ECE and for the reliability diagram."""

    bin_lower: np.ndarray
    bin_upper: np.ndarray
    bin_confidence: np.ndarray  # mean predicted confidence in the bin
    bin_accuracy: np.ndarray  # observed accept rate in the bin
    bin_count: np.ndarray  # number of samples in the bin

    @property
    def bin_center(self) -> np.ndarray:
        return 0.5 * (self.bin_lower + self.bin_upper)

    @property
    def gap(self) -> np.ndarray:
        """Signed miscalibration per bin (accuracy - confidence)."""
        return self.bin_accuracy - self.bin_confidence


def reliability_curve(
    confidence: np.ndarray,
    outcome: np.ndarray,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> ReliabilityCurve:
    """Bin predictions and report accuracy vs. confidence per bin.

    Empty bins are dropped so the curve only contains bins that actually carry
    data, otherwise the diagram sprouts misleading zeros.
    """
    confidence, outcome = _validate(confidence, outcome)
    edges = _bin_edges(n_bins, confidence, strategy)
    # np.digitize with right=True puts a value equal to an edge into the lower
    # bin; we special-case 0.0 so it lands in the first bin, not bin -1.
    idx = np.digitize(confidence, edges[1:-1], right=True)

    lowers, uppers, confs, accs, counts = [], [], [], [], []
    for b in range(len(edges) - 1):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        lowers.append(edges[b])
        uppers.append(edges[b + 1])
        confs.append(float(confidence[mask].mean()))
        accs.append(float(outcome[mask].mean()))
        counts.append(n)

    return ReliabilityCurve(
        bin_lower=np.asarray(lowers),
        bin_upper=np.asarray(uppers),
        bin_confidence=np.asarray(confs),
        bin_accuracy=np.asarray(accs),
        bin_count=np.asarray(counts, dtype=np.int64),
    )


def expected_calibration_error(
    confidence: np.ndarray,
    outcome: np.ndarray,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> float:
    """Expected Calibration Error: count-weighted mean of |accuracy - confidence|.

    This is the headline scalar. Zero means every bin's predicted confidence
    equals its observed accept rate.
    """
    curve = reliability_curve(confidence, outcome, n_bins, strategy)
    total = curve.bin_count.sum()
    if total == 0:
        return 0.0
    weights = curve.bin_count / total
    return float(np.sum(weights * np.abs(curve.gap)))


def maximum_calibration_error(
    confidence: np.ndarray,
    outcome: np.ndarray,
    n_bins: int = 15,
    strategy: str = "uniform",
) -> float:
    """Worst-case per-bin calibration gap. Sensitive to a single bad bin."""
    curve = reliability_curve(confidence, outcome, n_bins, strategy)
    if curve.bin_count.size == 0:
        return 0.0
    return float(np.max(np.abs(curve.gap)))


def brier_score(confidence: np.ndarray, outcome: np.ndarray) -> float:
    """Mean squared error between predicted confidence and the 0/1 outcome.

    A proper scoring rule: unlike ECE it penalises both miscalibration *and*
    lack of sharpness, so we report it alongside ECE rather than instead of it.
    """
    confidence, outcome = _validate(confidence, outcome)
    return float(np.mean((confidence - outcome) ** 2))
