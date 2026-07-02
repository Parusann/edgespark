"""Hardware demo engine: drives the real ROCm stack behind the dashboard.

Selected by ``python demo/server.py --hardware``. Streams the same event shape as
the CPU toy engine so ``app.js`` does not care which is running:

    {tokens: [str], accepted: int, ell: int, conf: [float], tok_s: float, step: int}

Imports torch/transformers lazily.
"""

from __future__ import annotations

import time


class HardwareDemo:
    def __init__(self, seed: int = 0):
        from edgespark.drafter import EdgeSparkDrafter
        from edgespark.loop.generate import EdgeSparkGenerator
        from edgespark.policy import ThresholdPolicy
        from edgespark.utils.config import DrafterConfig, PolicyConfig
        from edgespark.verifier import Verifier, VerifierConfig

        cfg = DrafterConfig.from_yaml("configs/drafter_qwen3_4b.yaml")
        pol = PolicyConfig.from_yaml("configs/policy.yaml")
        self.verifier = Verifier(VerifierConfig(model_name=cfg.target_model)).load()
        self.drafter = EdgeSparkDrafter(cfg, verifier_hidden=self.verifier.hidden_size)
        self.gen = EdgeSparkGenerator(
            self.verifier, self.drafter,
            ThresholdPolicy(theta=pol.theta, ema_gain=pol.ema_gain), precision="int8",
        )
        self.seed = seed
        self.prompt = "Write a Python function that returns the nth Fibonacci number."

    def vanilla(self, n_tokens: int, start: int = 0):

        # For a live stream we decode incrementally rather than in one shot.
        import torch

        seq = self.verifier.encode(self.prompt)
        _, _, kv = self.verifier.forward_with_hidden(seq)
        t0 = time.perf_counter()
        for i in range(n_tokens):
            logits, _, kv = self.verifier.forward_with_hidden(seq[:, -1:], past_key_values=kv)
            nxt = logits[0, -1].argmax().view(1, 1)
            seq = torch.cat([seq, nxt], dim=1)
            dt = time.perf_counter() - t0
            yield {"tokens": [self.verifier.decode([int(nxt)])], "accepted": 0, "ell": 0,
                   "tok_s": (i + 1) / dt if dt else 0.0}

    def edgespark(self, n_tokens: int, start: int = 0):
        out = self.gen.generate(self.prompt, n_tokens, mode="greedy", seed=self.seed)
        cursor = 0
        for m in out.metrics:
            toks = out.tokens[cursor:cursor + m.tau]
            cursor += m.tau
            yield {"tokens": [self.verifier.decode([t]) for t in toks],
                   "accepted": m.accepted, "ell": m.ell,
                   "conf": [round(c, 2) for c in m.conf_profile],
                   "tok_s": 1000.0 / (m.t_draft_ms + m.t_verify_ms) * m.tau, "step": m.step}
