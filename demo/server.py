"""EdgeSpark live demo server (spec section 9.8).

The report's ideal demo: *same local model, same machine, faster*. This serves a
side-by-side dashboard, vanilla decode vs EdgeSpark, streaming tokens with live
panels for tokens/sec, VRAM, tokens accepted speculatively this step, and a
draft-depth / accepted-length strip.

Two engines behind one Server-Sent-Events endpoint:

* **hardware** (``--hardware``): drives the real ``EdgeSparkGenerator`` on the
  7900 XTX.
* **cpu** (default): drives the numpy reference loop with correlated toy LMs, and
  times each round with the same latency model as ``bench.simulate``. No torch,
  no GPU, so the demo runs on any machine and still tells the throughput story.

Stdlib only (``http.server``), so ``python demo/server.py`` just works.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench import simulate
from edgespark.loop.acceptance import verify_block
from edgespark.policy import ThresholdPolicy
from edgespark.utils import ToyCategoricalLM

STATIC = Path(__file__).parent / "dashboard"
# A small, moderately-peaked toy vocabulary: sharp enough that the confidence
# proxy supports a realistic verification length, so the CPU demo reproduces the
# ~+40% throughput story rather than a pessimistic toy artefact.
VOCAB = 16
_TEMP = 0.5
# A whimsical readable "vocabulary" so the stream shows words, not token ids.
WORDS = (
    "the model drafts tokens ahead while the verifier checks them in one pass so "
    "generation runs faster without changing what it would have said on its own "
    "edge spark keeps the guarantee exact and only speed moves quantized drafter "
    "confidence gates how many tokens are worth verifying on a single stream today"
).split()


def _word(tok: int) -> str:
    return WORDS[tok % len(WORDS)]


class CpuDemo:
    """Toy-LM engine that runs anywhere and mimics the measured latency model."""

    def __init__(self, seed: int = 0):
        self.verifier = ToyCategoricalLM(vocab_size=VOCAB, seed=seed, temperature=_TEMP)
        # A strong drafter (low logit noise) so the demo shows the real throughput
        # story, not a pessimistic toy. Exactness holds at any drafter quality.
        self.drafter = ToyCategoricalLM.perturbed(
            self.verifier, noise=0.2, seed=seed + 1, temperature=_TEMP
        )
        self.policy = ThresholdPolicy(theta=0.45)

    def vanilla(self, n_tokens: int, start: int = 0):
        cur = start
        t = 0.0
        for _ in range(n_tokens):
            cur = int(self.verifier.dist(cur).argmax())
            t += simulate._T_DECODE_MS
            yield {"tokens": [_word(cur)], "accepted": 0, "ell": 0,
                   "elapsed_ms": t, "tok_s": 1000.0 * (_ + 1) / t}

    def edgespark(self, n_tokens: int, start: int = 0):
        cur, emitted, t, step = start, 0, 0.0, 0
        block = 5
        while emitted < n_tokens:
            draft_tokens, draft_dist = [], []
            c = cur
            for _ in range(block):
                d = self.drafter.dist(c)
                c = int(d.argmax())
                draft_tokens.append(c)
                draft_dist.append(d)
            draft_tokens = np.asarray(draft_tokens)
            conf = np.asarray([draft_dist[j][draft_tokens[j]] for j in range(block)])
            ell = self.policy.choose_length(conf, 1.0)
            target = self.verifier.block_target_dist(cur, draft_tokens)
            res = verify_block(target, draft_tokens, ell, mode="greedy")
            t += simulate._PRECISION["int8"]["t_draft_ms"] + simulate._t_verify(ell)
            emitted += len(res.tokens)
            cur = res.tokens[-1]
            yield {
                "tokens": [_word(x) for x in res.tokens],
                "accepted": res.n_accepted, "ell": ell,
                "conf": [round(float(x), 2) for x in conf[:ell]],
                "elapsed_ms": t, "tok_s": 1000.0 * emitted / t, "step": step,
            }
            step += 1


class Handler(BaseHTTPRequestHandler):
    engine_factory = staticmethod(lambda seed: CpuDemo(seed))

    def log_message(self, *args):  # quiet
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/stream":
            return self._stream(parse_qs(parsed.query))
        return self._static(parsed.path)

    def _static(self, path: str):
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (STATIC / rel).resolve()
        if not str(target).startswith(str(STATIC.resolve())) or not target.exists():
            self.send_error(404)
            return
        ctype = {".html": "text/html", ".js": "application/javascript", ".css": "text/css"}.get(
            target.suffix, "text/plain"
        )
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self, query):
        variant = query.get("variant", ["edgespark"])[0]
        n = int(query.get("n", ["96"])[0])
        seed = int(query.get("seed", ["0"])[0])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        engine = self.engine_factory(seed)
        gen = engine.vanilla(n) if variant == "vanilla" else engine.edgespark(n)
        try:
            for event in gen:
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
                # Pace to real elapsed time so both columns race believably.
                time.sleep(min(0.12, event["elapsed_ms"] / 1000.0 / max(n, 1)))
            self.wfile.write(b"event: done\ndata: {}\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--hardware", action="store_true", help="drive the real ROCm stack")
    args = ap.parse_args()
    if args.hardware:
        from demo.hardware_engine import HardwareDemo

        Handler.engine_factory = staticmethod(lambda seed: HardwareDemo(seed))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"EdgeSpark demo on http://127.0.0.1:{args.port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
