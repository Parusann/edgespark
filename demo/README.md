# EdgeSpark live demo

Side-by-side streaming, vanilla decode vs. EdgeSpark, with live tokens/sec,
speedup, a draft-depth / accepted-length strip, and the standing reminder that
the two columns emit the *same* tokens (spec §9.8).

## Run it

```bash
# CPU / any machine, toy LMs, timed with the measured latency model:
python demo/server.py
# open http://127.0.0.1:8000

# On the RX 7900 XTX with the real stack:
python demo/server.py --hardware
```

The CPU mode needs nothing but Python + numpy, so the demo works on a laptop
before the ROCm stack is stood up; `--hardware` swaps in the real
`EdgeSparkGenerator` behind the identical event stream.

## What you are looking at

- **Left column** decodes one token per verifier forward pass.
- **Right column** drafts a block, gates the verification length from the
  confidence head, verifies in one pass, and emits the accepted prefix plus the
  verifier's own next token.
- The **draft-depth strip** shows, per round, how many tokens were verified (`ℓ`),
  how many drafts were accepted (blue), the verifier token (green), and the
  positions the policy chose not to verify (grey).

Because the verifier owns every token, the right column is the same sequence the
left one produces, just fewer verifier passes to get there.
