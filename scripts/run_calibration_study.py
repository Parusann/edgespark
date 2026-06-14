"""Reproduce the confidence-head calibration study (spec section 9.4, Phase 4).

    python scripts/run_calibration_study.py [--out runs/calibration]

Measures ECE / Brier / reliability for fp16 vs INT8 vs NF4, fits temperature
recalibration on a held-out split, and reports the restored ECE. On the target
machine, point ``--from-log`` at a real metrics JSONL (predicted a_j vs observed
accept) instead of the model; the measurement/recalibration code is identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import simulate
from edgespark.calibration import expected_calibration_error
from edgespark.calibration.recalibrate import TemperatureScaler


def study_from_pairs(confidence: np.ndarray, outcome: np.ndarray) -> dict:
    """Measure + recalibrate a single (confidence, outcome) set from a real run."""
    cut = len(confidence) // 2
    scaler = TemperatureScaler().fit(confidence[:cut], outcome[:cut])
    recal = scaler.transform(confidence[cut:])
    return {
        "ece_raw": expected_calibration_error(confidence[cut:], outcome[cut:]),
        "ece_recalibrated": expected_calibration_error(recal, outcome[cut:]),
        "temperature": scaler.temperature,
        "n": int(len(confidence)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/calibration")
    ap.add_argument("--from-log", help="metrics JSONL with conf_profile + accepted from a real run")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.from_log:
        pairs = _pairs_from_log(args.from_log)
        result = {"real_run": study_from_pairs(*pairs)}
    else:
        result = {}
        for p in ("fp16", "int8", "nf4"):
            c = simulate.simulate_calibration(p)
            result[p] = {
                "ece_raw": round(c.ece_raw, 4),
                "ece_recalibrated": round(c.ece_recalibrated, 4),
                "brier_raw": round(c.brier_raw, 4),
                "brier_recalibrated": round(c.brier_recalibrated, 4),
                "temperature": round(c.temperature, 3),
            }

    (out / "calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nwrote {out / 'calibration.json'}")


def _pairs_from_log(path: str):
    from edgespark.utils.metrics_log import read_jsonl

    conf, acc = [], []
    for rec in read_jsonl(path):
        profile = rec.get("conf_profile", [])
        n_acc = rec.get("accepted", 0)
        for j, a in enumerate(profile):
            conf.append(a)
            acc.append(1.0 if j < n_acc else 0.0)
    return np.asarray(conf), np.asarray(acc)


if __name__ == "__main__":
    main()
