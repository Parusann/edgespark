"""INT8 (W8A8) quantization of the drafter — the primary variant.

``quantize_drafter_int8`` swaps the backbone's ``nn.Linear`` layers for
bitsandbytes-ROCm 8-bit linears (verified on gfx1100 in Phase 0). By default the
confidence head is left in high precision — it is the thing under study, so we
quantize it last and separately.

``fake_quantize_int8`` is the simulator: symmetric per-channel INT8 rounding of a
weight tensor. It is a pure-torch function used by the calibration study to show
how the confidence head's outputs drift under 8-bit weights, reproducibly and on
any device.
"""

from __future__ import annotations


def fake_quantize_int8(weight, per_channel: bool = True):
    """Symmetric INT8 fake-quant of a 2-D weight tensor. Returns the dequantized weight.

    Per-channel (per output row) scales, which is what real INT8 kernels use and
    what keeps the error small enough that *proposals* survive while *calibration*
    does not — the exact asymmetry the study is about.
    """
    import torch

    qmax = 127
    if per_channel and weight.dim() == 2:
        scale = weight.abs().amax(dim=1, keepdim=True) / qmax
    else:
        scale = weight.abs().amax() / qmax
    scale = scale.clamp_min(1e-8)
    q = torch.clamp(torch.round(weight / scale), -qmax, qmax)
    return q * scale


def quantize_drafter_int8(drafter, config):
    """Replace eligible Linear layers with bitsandbytes 8-bit linears in place.

    ``config`` is a :class:`~edgespark.utils.config.QuantConfig`. Layers are
    selected by whether they belong to the backbone / markov / confidence head so
    the study can quantize each independently.
    """
    import bitsandbytes as bnb
    import torch

    def _should_quantize(qualified_name: str) -> bool:
        if qualified_name.startswith("confidence") and not config.quantize_confidence_head:
            return False
        if qualified_name.startswith("markov") and not config.quantize_markov_head:
            return False
        if qualified_name.startswith("backbone") and not config.quantize_backbone:
            return False
        return True

    replaced = 0
    for name, module in list(drafter.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, torch.nn.Linear):
                continue
            full = f"{name}.{child_name}".lstrip(".")
            if not _should_quantize(full):
                continue
            q = bnb.nn.Linear8bitLt(
                child.in_features, child.out_features,
                bias=child.bias is not None, has_fp16_weights=False,
            )
            q.weight.data = child.weight.data
            if child.bias is not None:
                q.bias.data = child.bias.data
            setattr(module, child_name, q)
            replaced += 1
    return drafter, {"replaced_linears": replaced, "scheme": "w8a8"}
