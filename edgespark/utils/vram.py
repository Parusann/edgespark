"""VRAM accounting for the 24 GB budget (spec sections 6, 11).

Everything, verifier, drafter, KV cache, has to live inside 24 GB. We report
two numbers: what torch has allocated for tensors, and what the driver actually
reserved (via ``rocm-smi`` when present), because the gap between them is where
"it fit in my head" turns into an out-of-memory at token 4000.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class VramSnapshot:
    allocated_mb: float  # torch tensor allocations
    reserved_mb: float  # torch caching allocator reservation
    device_used_mb: float  # whole-device usage from rocm-smi, if available

    @property
    def headroom_mb(self) -> float:
        return max(0.0, 24 * 1024 - max(self.reserved_mb, self.device_used_mb))


def snapshot(device: int = 0) -> VramSnapshot:
    allocated = reserved = 0.0
    try:
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 2**20
            reserved = torch.cuda.memory_reserved(device) / 2**20
    except Exception:
        pass
    return VramSnapshot(allocated, reserved, _rocm_smi_used_mb(device))


def _rocm_smi_used_mb(device: int) -> float:
    """Best-effort whole-device VRAM usage via rocm-smi; 0.0 if unavailable."""
    exe = shutil.which("rocm-smi")
    if not exe:
        return 0.0
    try:
        out = subprocess.run(
            [exe, "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout
        import json

        data = json.loads(out)
        card = data.get(f"card{device}", {})
        for key, val in card.items():
            if "used" in key.lower():
                return float(val) / 2**20
    except Exception:
        return 0.0
    return 0.0


def peak_reset(device: int = 0) -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        pass


def peak_mb(device: int = 0) -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_reserved(device) / 2**20
    except Exception:
        pass
    return 0.0
