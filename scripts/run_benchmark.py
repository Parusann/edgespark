"""One-command benchmark entry point (spec section 18, deliverable 5).

    python scripts/run_benchmark.py --hardware      # real run on the 7900 XTX
    python scripts/run_benchmark.py --smoke         # CPU plumbing check, no torch
    python scripts/run_benchmark.py --simulate      # regenerate modelled reference numbers

``--simulate`` writes ``runs/reference/summary.json``, the modelled reference the
README tables and figures quote, so anyone can reproduce the *shape* of the
result without the hardware, then replace it with a real ``--hardware`` run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _simulate_summary() -> dict:
    from bench import simulate

    r = simulate.run_all()
    throughput = []
    for (p, ps, pol, _cal), t in r["throughput"].items():
        if pol != "gated":
            continue
        throughput.append({
            "label": "edgespark", "precision": p, "prompt_set": ps,
            "policy": "gated", "ell": t.ell, "mean_tau": round(t.tau, 3),
            "tok_s": round(t.tok_s_edgespark, 1),
            "tok_s_baseline": round(t.tok_s_baseline, 1),
            "speedup": round(t.speedup, 3),
            "speedup_pct": round((t.speedup - 1) * 100, 1),
        })
    ablation = []
    for p in ("fp16", "int8", "nf4"):
        g = simulate.simulate_throughput(p, "code", policy="gated")
        a = simulate.simulate_throughput(p, "code", policy="always")
        ablation.append({
            "precision": p,
            "gated": {"ell": g.ell, "tau": round(g.tau, 3), "tok_s": round(g.tok_s_edgespark, 1)},
            "always": {"ell": a.ell, "tau": round(a.tau, 3), "tok_s": round(a.tok_s_edgespark, 1)},
        })
    calibration = {
        p: {
            "ece_raw": round(c.ece_raw, 4), "ece_recalibrated": round(c.ece_recalibrated, 4),
            "brier_raw": round(c.brier_raw, 4), "brier_recalibrated": round(c.brier_recalibrated, 4),
            "temperature": round(c.temperature, 3),
        }
        for p, c in r["calibration"].items()
    }
    nf4 = r["nf4_policy"]
    return {
        "source": "modelled (bench.simulate); reproduce with --hardware on RX 7900 XTX",
        "verifier": "Qwen3-4B (INT8)",
        "throughput": throughput,
        "policy_ablation": ablation,
        "calibration": calibration,
        "nf4_recalibration_effect": {
            "uncalibrated": {"ell": nf4["uncalibrated"].ell,
                             "speedup_pct": round((nf4["uncalibrated"].speedup - 1) * 100, 1)},
            "recalibrated": {"ell": nf4["calibrated"].ell,
                             "speedup_pct": round((nf4["calibrated"].speedup - 1) * 100, 1)},
        },
        "vram": {k: v for k, v in r["vram"].items()},
    }


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--hardware", action="store_true", help="real run on the target GPU")
    g.add_argument("--smoke", action="store_true", help="CPU plumbing check")
    g.add_argument("--simulate", action="store_true", help="regenerate modelled reference numbers")
    ap.add_argument("--out", default="runs/reference/summary.json")
    args = ap.parse_args()

    if args.smoke:
        from bench.harness import run_smoke

        report = run_smoke()
        print(report.to_json())
        return

    if args.simulate:
        summary = _simulate_summary()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {out}")
        for row in summary["throughput"]:
            print(f"  {row['precision']:5s} {row['prompt_set']:4s}  "
                  f"{row['tok_s']:6.1f} tok/s  (+{row['speedup_pct']:.0f}%)")
        return

    if args.hardware:
        from bench.harness import run_hardware
        from data.prompts import load_prompt_sets
        from edgespark.utils.config import BenchConfig, DrafterConfig, PolicyConfig

        bench_cfg = BenchConfig.from_yaml("configs/bench.yaml")
        report = run_hardware(
            bench_cfg,
            DrafterConfig.from_yaml("configs/drafter_qwen3_4b.yaml"),
            PolicyConfig.from_yaml("configs/policy.yaml"),
            load_prompt_sets(bench_cfg.prompt_sets),
        )
        report.save(args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
