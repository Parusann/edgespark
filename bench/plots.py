"""matplotlib plotting for real hardware runs (spec section 12).

The committed README figures are dependency-free SVG (``scripts/make_figures.py``).
This module is for interactive analysis on the GPU box: point it at a metrics
JSONL or a benchmark summary and get the same four charts as PNGs, plus the
per-step diagnostics (accept rate over time, ell distribution) that are useful
while tuning but too noisy for the README.

matplotlib is imported lazily and listed in ``requirements-dev``.
"""

from __future__ import annotations

from pathlib import Path

from edgespark.calibration import reliability_curve
from edgespark.utils.metrics_log import read_jsonl


def reliability_plot(pairs_by_precision: dict, out: str = "runs/plots/reliability.png"):
    """pairs_by_precision: {precision: (confidence, outcome, recalibrated)}."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(pairs_by_precision), figsize=(4 * len(pairs_by_precision), 4))
    if len(pairs_by_precision) == 1:
        axes = [axes]
    for ax, (prec, (conf, outcome, recal)) in zip(axes, pairs_by_precision.items(), strict=False):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
        raw = reliability_curve(conf, outcome)
        ax.plot(raw.bin_confidence, raw.bin_accuracy, "o-", color="#dc2626", label="raw")
        if recal is not None:
            rc = reliability_curve(recal, outcome)
            ax.plot(rc.bin_confidence, rc.bin_accuracy, "o-", color="#059669", label="recalibrated")
        ax.set_title(prec.upper())
        ax.set_xlabel("predicted confidence")
        ax.set_ylabel("observed acceptance")
        ax.legend()
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    return out


def accept_over_time(metrics_path: str, out: str = "runs/plots/accept_over_time.png"):
    import matplotlib.pyplot as plt

    records = read_jsonl(metrics_path)
    steps = [r["step"] for r in records]
    accepted = [r["accepted"] for r in records]
    ell = [r["ell"] for r in records]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(steps, accepted, label="accepted", color="#2563eb")
    ax.plot(steps, ell, label="ell", color="#f59e0b", alpha=0.6)
    ax.set_xlabel("generation step")
    ax.set_ylabel("tokens")
    ax.legend()
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    return out
