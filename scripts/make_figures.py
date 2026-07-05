"""Regenerate the committed SVG figures from the modelled results.

    python scripts/make_figures.py [--out docs/assets]

Every figure is drawn from ``bench.simulate`` so the repo's charts and its numbers
can never drift apart. The reliability diagrams in particular are produced by the
real calibration code (``edgespark.calibration``) run on simulated confidence
data, so the figure exercises the same path a hardware run would.

Pure numpy + the tiny SVG helper in ``bench.svgplot`` — no matplotlib, so this
runs on any machine and leaves text-diffable SVG in git rather than binary PNGs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import simulate
from bench.svgplot import (
    ACCENT,
    ACCENT2,
    BAD,
    GOOD,
    GRID,
    INK,
    MUTED,
    SERIES,
    Canvas,
    save,
)


def _axes(c: Canvas, x0, y0, w, h, *, xlabel="", ylabel="", ymax=1.0, yticks=5, xmax=1.0, xticks=5):
    c.rect(x0, y0, w, h, "#ffffff", stroke=GRID, sw=1)
    for i in range(yticks + 1):
        yy = y0 + h - h * i / yticks
        c.line(x0, yy, x0 + w, yy, stroke=GRID, sw=1)
        c.text(x0 - 8, yy + 4, f"{ymax * i / yticks:.2g}", size=11, fill=MUTED, anchor="end")
    if xlabel:
        c.text(x0 + w / 2, y0 + h + 34, xlabel, size=12, fill=INK, anchor="middle")
    if ylabel:
        c.text(x0 - 40, y0 + h / 2, ylabel, size=12, fill=INK, anchor="middle",
               style=f' transform="rotate(-90 {x0 - 40} {y0 + h / 2})"')


# --- Figure 1: reliability diagrams (the money plot) -------------------------

def reliability_figure(results, out: Path):
    precisions = ["fp16", "int8", "nf4"]
    pw, ph, gap, top, left = 240, 240, 46, 66, 60
    W = left + len(precisions) * (pw + gap)
    H = top + ph + 88
    c = Canvas(W, H)
    c.text(left, 30, "Confidence-head reliability: quantization vs. recalibration",
           size=17, fill=INK, weight="700")
    c.text(left, 50, "Perfectly calibrated = points on the diagonal. Red = raw quantized head, green = after temperature scaling.",
           size=12, fill=MUTED)

    for i, p in enumerate(precisions):
        cal = results["calibration"][p]
        x0 = left + i * (pw + gap)
        y0 = top
        c.rect(x0, y0, pw, ph, "#ffffff", stroke=GRID, sw=1)
        # grid
        for k in range(1, 5):
            c.line(x0 + pw * k / 4, y0, x0 + pw * k / 4, y0 + ph, stroke="#f1f5f9", sw=1)
            c.line(x0, y0 + ph * k / 4, x0 + pw, y0 + ph * k / 4, stroke="#f1f5f9", sw=1)
        # diagonal (ideal)
        c.line(x0, y0 + ph, x0 + pw, y0, stroke=MUTED, sw=1.2, dash="4 3")

        def to_px(xs, ys, x0=x0, y0=y0):
            return [(x0 + xv * pw, y0 + ph - yv * ph) for xv, yv in zip(xs, ys, strict=False)]

        raw = cal.curve_raw
        rec = cal.curve_recalibrated
        c.polyline(to_px(raw.bin_confidence, raw.bin_accuracy), stroke=BAD, sw=2.2)
        for xv, yv in to_px(raw.bin_confidence, raw.bin_accuracy):
            c.circle(xv, yv, 2.6, fill=BAD)
        c.polyline(to_px(rec.bin_confidence, rec.bin_accuracy), stroke=GOOD, sw=2.2)
        for xv, yv in to_px(rec.bin_confidence, rec.bin_accuracy):
            c.circle(xv, yv, 2.6, fill=GOOD)

        c.text(x0 + pw / 2, y0 - 12, p.upper(), size=14, fill=INK, weight="700", anchor="middle")
        c.text(x0 + 8, y0 + 18, f"ECE {cal.ece_raw:.3f}", size=12, fill=BAD, weight="600")
        c.text(x0 + 8, y0 + 36, f"→ {cal.ece_recalibrated:.3f}", size=12, fill=GOOD, weight="600")
        c.text(x0 + pw - 8, y0 + ph - 10, f"T={cal.temperature:.2f}", size=11, fill=MUTED, anchor="end")
        c.text(x0 + pw / 2, y0 + ph + 20, "predicted confidence", size=11, fill=MUTED, anchor="middle")

    c.text(left, H - 16, "observed acceptance rate  (y)   ·   predicted confidence  (x)   ·   n = 30,000 held-out positions per precision",
           size=11, fill=MUTED)
    save(c, out, "Reliability diagrams")


# --- Figure 2: throughput ----------------------------------------------------

def throughput_figure(results, out: Path):
    precisions = ["fp16", "int8", "nf4"]
    c = Canvas(720, 380)
    c.text(40, 30, "End-to-end throughput: EdgeSpark vs. vanilla quantized baseline",
           size=17, fill=INK, weight="700")
    c.text(40, 50, "DESIGN-TIME MODEL (not measured) · Qwen3-4B INT8 verifier · single stream · gated verification · output identical to baseline",
           size=12, fill=MUTED)

    x0, y0, w, h = 70, 80, 600, 220
    tok = [results["throughput"][(p, "code", "gated", True)] for p in precisions]
    tok_chat = [results["throughput"][(p, "chat", "gated", True)] for p in precisions]
    baseline = tok[0].tok_s_baseline
    ymax = 110
    _axes(c, x0, y0, w, h, ymax=ymax, yticks=5)
    c.text(x0 - 44, y0 + h / 2, "tokens / sec", size=12, fill=INK, anchor="middle",
           style=f' transform="rotate(-90 {x0 - 44} {y0 + h / 2})"')

    # baseline reference line
    by = y0 + h - h * baseline / ymax
    c.line(x0, by, x0 + w, by, stroke=ACCENT2, sw=2, dash="6 4")
    c.text(x0 + w - 4, by - 6, f"baseline {baseline:.0f} tok/s", size=11, fill=ACCENT2, anchor="end", weight="600")

    group_w = w / len(precisions)
    bw = 46
    for i, p in enumerate(precisions):
        gx = x0 + group_w * i + group_w / 2
        for j, (t, _label, col) in enumerate([(tok[i], "code", ACCENT), (tok_chat[i], "chat", GOOD)]):
            bx = gx - bw + j * bw
            bh = h * t.tok_s_edgespark / ymax
            c.rect(bx, y0 + h - bh, bw - 6, bh, col, rx=3)
            c.text(bx + (bw - 6) / 2, y0 + h - bh - 18, f"{t.tok_s_edgespark:.0f}", size=12, fill=INK, anchor="middle", weight="700")
            c.text(bx + (bw - 6) / 2, y0 + h - bh - 4, f"+{(t.speedup - 1) * 100:.0f}%", size=10, fill=col, anchor="middle", weight="600")
        c.text(gx, y0 + h + 20, f"{p.upper()} drafter", size=12, fill=INK, anchor="middle", weight="600")

    # legend
    c.rect(x0 + 6, y0 + 6, 12, 12, ACCENT, rx=2)
    c.text(x0 + 24, y0 + 16, "code", size=11, fill=INK)
    c.rect(x0 + 74, y0 + 6, 12, 12, GOOD, rx=2)
    c.text(x0 + 92, y0 + 16, "chat", size=11, fill=INK)
    c.text(40, 366, "Modelled projection (bench/simulate.py). On hardware: correctness + baselines measured; quantized run pending — see docs/RESULTS.md §0.", size=11, fill=MUTED)
    save(c, out, "Throughput")


# --- Figure 3: accepted-per-call, gated vs always-verify-all -----------------

def policy_figure(results, out: Path):
    precisions = ["fp16", "int8", "nf4"]
    c = Canvas(720, 380)
    c.text(40, 30, "Verification-length policy: gated vs. always-verify-all", size=17, fill=INK, weight="700")
    c.text(40, 50, "Accepted tokens per verifier call (τ). Gating stops before the low-survival tail wastes verifier time.",
           size=12, fill=MUTED)

    x0, y0, w, h = 70, 80, 600, 220
    ymax = 5.0
    _axes(c, x0, y0, w, h, ymax=ymax, yticks=5)
    c.text(x0 - 44, y0 + h / 2, "accepted / call (τ)", size=12, fill=INK, anchor="middle",
           style=f' transform="rotate(-90 {x0 - 44} {y0 + h / 2})"')

    group_w = w / len(precisions)
    bw = 46
    for i, p in enumerate(precisions):
        g = simulate.simulate_throughput(p, "code", policy="gated")
        a = simulate.simulate_throughput(p, "code", policy="always")
        gx = x0 + group_w * i + group_w / 2
        for j, (t, _label, col) in enumerate([(a, "always", MUTED), (g, "gated", ACCENT)]):
            bx = gx - bw + j * bw
            bh = h * t.tau / ymax
            c.rect(bx, y0 + h - bh, bw - 6, bh, col, rx=3)
            c.text(bx + (bw - 6) / 2, y0 + h - bh - 16, f"{t.tau:.2f}", size=12, fill=INK, anchor="middle", weight="700")
            c.text(bx + (bw - 6) / 2, y0 + h - bh - 2, f"ℓ={t.ell}", size=10, fill=col, anchor="middle")
            c.text(bx + (bw - 6) / 2, y0 + h - bh + 14 if bh > 40 else y0 + h - bh - 30,
                   f"+{(t.speedup - 1) * 100:.0f}%", size=10, fill="#ffffff" if bh > 40 else col, anchor="middle", weight="700")
        c.text(gx, y0 + h + 20, f"{p.upper()} drafter", size=12, fill=INK, anchor="middle", weight="600")

    c.rect(x0 + 6, y0 + 6, 12, 12, MUTED, rx=2)
    c.text(x0 + 24, y0 + 16, "always-verify-all (ℓ=5)", size=11, fill=INK)
    c.rect(x0 + 200, y0 + 6, 12, 12, ACCENT, rx=2)
    c.text(x0 + 218, y0 + 16, "confidence-gated ℓ", size=11, fill=INK)
    c.text(40, 366, "Modelled. On the RX 7900 XTX the per-position verify cost ≈ 0, so gating ties always-verify-all — see docs/RESULTS.md §0.", size=11, fill=MUTED)
    save(c, out, "Policy ablation")


# --- Figure 4: VRAM ----------------------------------------------------------

def vram_figure(results, out: Path):
    configs = [
        ("fp16 verifier\n+ INT8 drafter", results["vram"]["fp16_verifier"]),
        ("INT8 verifier\n+ INT8 drafter", results["vram"]["int8"]),
        ("INT8 verifier\n+ NF4 drafter", results["vram"]["nf4"]),
    ]
    c = Canvas(720, 380)
    c.text(40, 30, "VRAM footprint inside the 24 GB budget", size=17, fill=INK, weight="700")
    c.text(40, 50, "Verifier + drafter + KV cache + activations for Qwen3-4B at ~8k context on RX 7900 XTX.",
           size=12, fill=MUTED)

    x0, y0, w, h = 70, 80, 600, 220
    budget = 24 * 1024
    _axes(c, x0, y0, w, h, ymax=budget / 1024, yticks=6)
    c.text(x0 - 44, y0 + h / 2, "GB", size=12, fill=INK, anchor="middle",
           style=f' transform="rotate(-90 {x0 - 44} {y0 + h / 2})"')
    # budget line
    c.line(x0, y0, x0 + w, y0, stroke=BAD, sw=2, dash="6 4")
    c.text(x0 + w - 4, y0 - 6, "24 GB", size=11, fill=BAD, anchor="end", weight="600")

    parts = [("verifier", SERIES[0]), ("drafter", SERIES[4]), ("kv_cache", SERIES[5]), ("activations", ACCENT2)]
    group_w = w / len(configs)
    bw = 90
    for i, (label, v) in enumerate(configs):
        gx = x0 + group_w * i + group_w / 2 - bw / 2
        acc = 0.0
        for key, col in parts:
            seg = v[key]
            bh = h * seg / budget
            yy = y0 + h - (h * (acc + seg) / budget)
            c.rect(gx, yy, bw, bh, col, rx=2)
            if bh > 16:
                c.text(gx + bw / 2, yy + bh / 2 + 4, f"{seg / 1024:.1f}", size=10, fill="#ffffff", anchor="middle", weight="600")
            acc += seg
        total_y = y0 + h - h * v["total"] / budget
        c.text(gx + bw / 2, total_y - 8, f"{v['total'] / 1024:.1f} GB", size=12, fill=INK, anchor="middle", weight="700")
        for li, ln in enumerate(label.split("\n")):
            c.text(gx + bw / 2, y0 + h + 18 + li * 14, ln, size=11, fill=INK, anchor="middle")

    lx = x0 + 6
    for key, col in parts:
        c.rect(lx, y0 + 6, 12, 12, col, rx=2)
        c.text(lx + 16, y0 + 16, key.replace("_", " "), size=11, fill=INK)
        lx += 92
    save(c, out, "VRAM breakdown")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/assets")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = simulate.run_all()
    reliability_figure(results, out / "reliability_diagrams.svg")
    throughput_figure(results, out / "throughput.svg")
    policy_figure(results, out / "policy_ablation.svg")
    vram_figure(results, out / "vram_breakdown.svg")
    print(f"wrote 4 figures to {out}")


if __name__ == "__main__":
    main()
