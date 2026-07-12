"""Fake-quantization sanity checks (spec section 13).

torch-gated: these run on the ROCm box and any dev machine with torch, and are
skipped on the pure-numpy CI lane. They assert the simulators behave like real
quantizers, bounded error, exact grids, so the calibration study rests on a
quantizer that actually quantizes.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from edgespark.quantize import fake_quantize_int8, fake_quantize_nf4
from edgespark.quantize.nf4 import _NF4_LEVELS


def test_int8_fake_quant_is_bounded_and_close():
    w = torch.randn(64, 128)
    q = fake_quantize_int8(w)
    assert q.shape == w.shape
    # Per-channel INT8 error is at most half a step of the row's own scale.
    scale = w.abs().amax(dim=1, keepdim=True) / 127
    assert torch.all((q - w).abs() <= scale + 1e-6)


def test_int8_preserves_sign_of_large_weights():
    w = torch.randn(32, 32) * 3
    q = fake_quantize_int8(w)
    big = w.abs() > 0.5
    assert torch.all(torch.sign(q[big]) == torch.sign(w[big]))


def test_nf4_snaps_to_grid():
    w = torch.randn(16, 64)
    q = fake_quantize_nf4(w, block_size=64)
    levels = torch.tensor(_NF4_LEVELS)
    # Every dequantized value equals some grid level times its block absmax.
    scale = w.reshape(-1, 64).abs().amax(dim=1, keepdim=True)
    normed = (q.reshape(-1, 64) / scale.clamp_min(1e-8))
    nearest = torch.min((normed.unsqueeze(-1) - levels).abs(), dim=-1).values
    assert torch.all(nearest < 1e-4)


def test_nf4_error_exceeds_int8_error():
    # 4 bits should hurt more than 8 bits, the premise of the study.
    torch.manual_seed(0)
    w = torch.randn(128, 256)
    e8 = (fake_quantize_int8(w) - w).abs().mean()
    e4 = (fake_quantize_nf4(w) - w).abs().mean()
    assert e4 > e8
