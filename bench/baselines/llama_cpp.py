"""llama.cpp baselines and sanity oracle (spec sections 6, 9.6, 12).

llama.cpp has excellent HIP/ROCm support and *built-in* speculative decoding
(``--model-draft``). EdgeSpark uses it two ways, and neither is as the drafter
host:

* **Baseline 1** — vanilla quantized verifier, no speculation. The tokens/sec
  floor EdgeSpark has to beat.
* **Baseline 2** — vanilla quantized verifier + a separate draft model via
  ``--model-draft``. This is *ordinary* speculative decoding with no hidden-state
  conditioning, no Markov head, no confidence head — exactly the thing EdgeSpark's
  semi-AR drafter is supposed to improve on.

It also serves as an independent tokens/sec oracle: if EdgeSpark and llama.cpp
disagree wildly on baseline throughput, something is wrong with our timing.

This module shells out to a prebuilt ``llama-cli`` / ``llama-server`` and parses
its timing output; it does not vendor llama.cpp.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class LlamaResult:
    tok_s: float
    eval_ms_per_tok: float
    command: str


_TIMING_RE = re.compile(r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return proc.stdout + proc.stderr


def _parse_tok_s(output: str) -> tuple[float, float]:
    matches = _TIMING_RE.findall(output)
    if not matches:
        raise RuntimeError("could not parse llama.cpp eval timing from output")
    ms, toks = matches[-1]
    ms, toks = float(ms), int(toks)
    per_tok = ms / toks
    return 1000.0 / per_tok, per_tok


def baseline_no_speculation(model_gguf: str, prompt: str, n_predict: int = 256,
                            ngl: int = 999, binary: str = "llama-cli") -> LlamaResult:
    """Baseline 1: plain decode on the quantized verifier."""
    exe = shutil.which(binary) or binary
    cmd = [exe, "-m", model_gguf, "-p", prompt, "-n", str(n_predict), "-ngl", str(ngl), "-no-cnv"]
    out = _run(cmd)
    tok_s, per_tok = _parse_tok_s(out)
    return LlamaResult(tok_s=tok_s, eval_ms_per_tok=per_tok, command=" ".join(cmd))


def baseline_vanilla_speculative(model_gguf: str, draft_gguf: str, prompt: str,
                                 n_predict: int = 256, draft: int = 5, ngl: int = 999,
                                 binary: str = "llama-cli") -> LlamaResult:
    """Baseline 2: llama.cpp built-in speculative decoding with a separate draft model."""
    exe = shutil.which(binary) or binary
    cmd = [exe, "-m", model_gguf, "-md", draft_gguf, "--draft", str(draft),
           "-p", prompt, "-n", str(n_predict), "-ngl", str(ngl), "-no-cnv"]
    out = _run(cmd)
    tok_s, per_tok = _parse_tok_s(out)
    return LlamaResult(tok_s=tok_s, eval_ms_per_tok=per_tok, command=" ".join(cmd))
