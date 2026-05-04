"""The controlled inference loop and the exact acceptance rule it depends on."""

from edgespark.loop.acceptance import (
    AcceptResult,
    greedy_accept,
    speculative_accept,
    verify_block,
)
from edgespark.loop.reference import (
    DecodeResult,
    StepRecord,
    speculative_decode,
)

__all__ = [
    "AcceptResult",
    "greedy_accept",
    "speculative_accept",
    "verify_block",
    "DecodeResult",
    "StepRecord",
    "speculative_decode",
]
