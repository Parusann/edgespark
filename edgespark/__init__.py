"""EdgeSpark: quantized speculative decoding for a single consumer AMD GPU.

The public surface is intentionally small. The verifier decides every token; the
drafter and the verification-length policy only affect *speed*. See
``edgespark.loop.acceptance`` for the exact rule that makes that guarantee hold.
"""

__version__ = "0.6.0"

__all__ = ["__version__"]
