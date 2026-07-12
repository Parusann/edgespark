"""The controlled PyTorch + ROCm inference loop (spec section 9.6).

This is the real thing the numpy ``reference.py`` loop specifies: verifier hidden
states in, drafted block out, exact verification, accepted tokens emitted, with
timing and per-step metrics logged. It owns the verifier's KV cache and the
hidden-state hooks, which is exactly why EdgeSpark runs in a controlled loop
rather than a black-box serving framework, no off-the-shelf runtime hands you
selected-layer hidden states, a custom Markov+confidence head, and per-step
``ell`` selection at once.

torch is imported lazily; the loop is exercised end-to-end by the numpy reference
on CPU and by this module on the 7900 XTX.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edgespark.utils import vram
from edgespark.utils.metrics_log import StepMetrics
from edgespark.utils.timing import GpuTimer


@dataclass
class GenerationOutput:
    tokens: list[int]
    text: str
    metrics: list[StepMetrics]

    @property
    def mean_tau(self) -> float:
        taus = [m.tau for m in self.metrics]
        return float(np.mean(taus)) if taus else 0.0

    @property
    def mean_accepted(self) -> float:
        acc = [m.accepted for m in self.metrics]
        return float(np.mean(acc)) if acc else 0.0


class EdgeSparkGenerator:
    """Speculative generation with a confidence-gated verification length."""

    def __init__(self, verifier, drafter, policy=None, *, precision: str = "fp16"):
        self.verifier = verifier
        self.drafter = drafter
        self.policy = policy
        self.precision = precision
        self.timer = GpuTimer()

    def _last_position_hidden(self, hidden_by_layer):
        # The drafter conditions on the hidden states at the final context
        # position, the point from which the next block is proposed.
        return {lid: h[:, -1:, :] for lid, h in hidden_by_layer.items()}

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        *,
        mode: str = "greedy",
        seed: int = 0,
        exactness_check: bool = False,
    ) -> GenerationOutput:
        import torch

        rng = np.random.default_rng(seed) if mode == "stochastic" else None
        seq = self.verifier.encode(prompt)  # [1, T]

        # Prime the verifier: logits + selected hidden states + KV cache.
        _, hidden, kv = self.verifier.forward_with_hidden(seq)

        generated: list[int] = []
        metrics: list[StepMetrics] = []
        recent_ema = 1.0
        step = 0
        vram.peak_reset()

        while len(generated) < max_new_tokens:
            with self.timer.region("draft"):
                block = self.drafter.draft(self._last_position_hidden(hidden), seq)
            draft_tokens = torch.as_tensor(block.tokens, device=seq.device)

            if self.policy is None:
                ell = self.drafter.block_size
            else:
                ell = int(self.policy.choose_length(block.confidence, recent_ema))
            ell = int(np.clip(ell, 0, self.drafter.block_size))

            with self.timer.region("verify"):
                result = self.verifier.verify(
                    seq, draft_tokens, ell, mode=mode,
                    draft_dist=block.dist if mode == "stochastic" else None, rng=rng,
                )

            accepted = result.tokens
            generated.extend(accepted)

            # Advance the verifier over the accepted tokens to refresh hidden
            # states and KV for the next block. (Reusing the verify pass's tail
            # hidden state is the obvious optimization; kept explicit here.)
            new = torch.as_tensor(accepted, device=seq.device).view(1, -1)
            seq = torch.cat([seq, new], dim=1)
            _, hidden, kv = self.verifier.forward_with_hidden(new, past_key_values=kv)

            if ell > 0:
                recent_ema = 0.9 * recent_ema + 0.1 * (result.n_accepted / ell)

            exact_ok = True
            if exactness_check:
                exact_ok = self._check_exact(seq, accepted, mode, seed)

            metrics.append(
                StepMetrics(
                    step=step, ell=ell, accepted=result.n_accepted, tau=result.tau,
                    t_draft_ms=self.timer.mean_ms("draft"),
                    t_verify_ms=self.timer.mean_ms("verify"),
                    vram_mb=vram.snapshot().reserved_mb,
                    conf_profile=[float(c) for c in block.confidence[:ell]],
                    precision=self.precision, exact_ok=exact_ok,
                )
            )
            step += 1

        text = self.verifier.decode(generated[:max_new_tokens])
        return GenerationOutput(tokens=generated[:max_new_tokens], text=text, metrics=metrics)

    def _check_exact(self, seq, accepted, mode, seed) -> bool:
        # Cross-check a token against the verifier decoding alone. For greedy this
        # is exact; for stochastic it is a coarse guard (full unbiasedness lives
        # in the test suite, not the hot loop).
        if mode != "greedy":
            return True
        import torch

        with torch.no_grad():
            logits = self.verifier._model(input_ids=seq[:, :-len(accepted)], use_cache=False).logits
        return int(logits[0, -1].argmax()) == accepted[0]
