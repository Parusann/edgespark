"""Quantization pipeline for the drafter (spec section 9.3).

Two layers of functionality:

* **Real** quantization for the RX 7900 XTX: INT8 W8A8 and NF4 4-bit via
  bitsandbytes-ROCm. This is what ships.
* **Fake** (simulated) quantization: deterministic rounding of weights onto the
  INT8 / NF4 grid in plain torch. It reproduces the *numerical* effect of
  quantization without needing gfx1100 kernels, which is what makes the
  confidence-calibration study (section 9.4) runnable and unit-testable off the
  target GPU. Fake-quant is a standard tool, not a shortcut, it is how the
  degradation is isolated from kernel-level noise.

FP8 is intentionally absent: unsupported on RDNA3 (spec section 4).
"""

from edgespark.quantize.int8 import fake_quantize_int8, quantize_drafter_int8
from edgespark.quantize.nf4 import fake_quantize_nf4, quantize_drafter_nf4

__all__ = [
    "fake_quantize_int8",
    "quantize_drafter_int8",
    "fake_quantize_nf4",
    "quantize_drafter_nf4",
]
