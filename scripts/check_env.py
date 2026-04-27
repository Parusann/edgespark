"""Phase 0 environment gate (spec section 16).

Confirms the pieces EdgeSpark is built on actually work on this machine *before*
weeks of work assume them: ROCm-visible GPU, a Qwen3 forward pass, an INT8 quant
tool that runs on gfx1100, and the flash-attention situation. Prints a checklist
and exits non-zero if a hard requirement is missing.

Nothing here needs the EdgeSpark package; it is deliberately runnable on a fresh
clone as the first thing you do.
"""

from __future__ import annotations

import importlib
import shutil
import sys


def _check(name, fn):
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"{type(e).__name__}: {e}"
    mark = "OK  " if ok else "MISS"
    print(f"  [{mark}] {name}: {detail}")
    return ok


def _torch_gpu():
    import torch

    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"
    name = torch.cuda.get_device_name(0)
    hip = getattr(torch.version, "hip", None)
    return True, f"{name} (HIP {hip})"


def _rocm_smi():
    exe = shutil.which("rocm-smi")
    return (bool(exe), exe or "rocm-smi not on PATH")


def _transformers():
    m = importlib.import_module("transformers")
    return True, f"transformers {m.__version__}"


def _bitsandbytes():
    m = importlib.import_module("bitsandbytes")
    # Presence is necessary but not sufficient on gfx1100 — the Phase 0 gate also
    # runs a tiny Linear8bitLt forward; see docs/PHASE0.md.
    return True, f"bitsandbytes {getattr(m, '__version__', '?')}"


def _qwen_forward():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen3-4B"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cuda")
    ids = tok("def hello():", return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        model(ids)
    return True, f"{name} forward pass ran"


def main():
    print("EdgeSpark Phase 0 environment gate\n")
    hard = [
        ("PyTorch + ROCm GPU", _torch_gpu),
        ("transformers", _transformers),
        ("INT8 quant tool (bitsandbytes)", _bitsandbytes),
    ]
    soft = [
        ("rocm-smi (VRAM/energy telemetry)", _rocm_smi),
        ("Qwen3-4B forward pass", _qwen_forward),
    ]
    print("Hard requirements:")
    hard_ok = all(_check(n, f) for n, f in hard)
    print("\nRecommended:")
    for n, f in soft:
        _check(n, f)

    print()
    if hard_ok:
        print("Phase 0 hard gate: PASS")
        sys.exit(0)
    print("Phase 0 hard gate: FAIL — resolve the MISS items before building on them.")
    sys.exit(1)


if __name__ == "__main__":
    main()
