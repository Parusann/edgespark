"""NF4 (4-bit) quantization of the drafter, the aggressive variant.

NF4 is the interesting stress case for the calibration study (spec section 9.3):
4 bits is expected to damage the confidence head harder than INT8, so it is where
recalibration has the most to recover.

``quantize_drafter_nf4`` uses bitsandbytes ``Linear4bit`` with the NF4 datatype
and double quantization. ``fake_quantize_nf4`` simulates it: block-wise scaling
onto the 16-level NormalFloat-4 grid, in pure torch, for reproducible study.
"""

from __future__ import annotations

# The NF4 code points (Dettmers et al., 2023): 16 levels spaced by the quantiles
# of a unit normal, so they match the distribution of neural-net weights better
# than uniform 4-bit. Symmetric around 0 with an exact zero.
_NF4_LEVELS = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)


def fake_quantize_nf4(weight, block_size: int = 64):
    """Block-wise NF4 fake-quant of a weight tensor. Returns the dequantized weight.

    Each contiguous block of ``block_size`` weights is scaled by its own absmax,
    snapped to the nearest NF4 level, and scaled back, mirroring how the real
    NF4 kernel stores a per-block scale plus 4-bit codes.
    """
    import torch

    levels = torch.tensor(_NF4_LEVELS, dtype=weight.dtype, device=weight.device)
    flat = weight.reshape(-1)
    pad = (-flat.numel()) % block_size
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.reshape(-1, block_size)
    scale = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    normed = blocks / scale
    # Snap each value to the nearest NF4 level.
    idx = torch.argmin((normed.unsqueeze(-1) - levels).abs(), dim=-1)
    deq = levels[idx] * scale
    out = deq.reshape(-1)[: weight.numel()]
    return out.reshape(weight.shape)


def quantize_drafter_nf4(drafter, config):
    """Replace eligible Linear layers with bitsandbytes NF4 4-bit linears in place."""
    import bitsandbytes as bnb
    import torch

    def _should_quantize(name: str) -> bool:
        if name.startswith("confidence") and not config.quantize_confidence_head:
            return False
        if name.startswith("markov") and not config.quantize_markov_head:
            return False
        if name.startswith("backbone") and not config.quantize_backbone:
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
            q = bnb.nn.Linear4bit(
                child.in_features, child.out_features,
                bias=child.bias is not None,
                compute_dtype=getattr(torch, config.compute_dtype),
                quant_type="nf4", compress_statistics=config.double_quant,
            )
            q.weight = bnb.nn.Params4bit(child.weight.data, requires_grad=False, quant_type="nf4")
            if child.bias is not None:
                q.bias.data = child.bias.data
            setattr(module, child_name, q)
            replaced += 1
    return drafter, {"replaced_linears": replaced, "scheme": "nf4"}
