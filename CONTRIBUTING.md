# Contributing

Thanks for looking at EdgeSpark. A few notes to make changes painless.

## Setup

```bash
pip install numpy pyyaml pytest ruff        # the pure-numpy core + tooling
# for the ROCm / neural stack (RX 7900 XTX):
pip install -r requirements-rocm.txt
```

## The one rule

**Exactness is not negotiable.** EdgeSpark's whole guarantee is that it never
changes the verifier's output. Any change that touches the acceptance rule
(`edgespark/loop/acceptance.py`), the verifier's `verify()`, or the inference loop
must keep `tests/test_exactness.py` green. If you're changing how the drafter or
policy behaves, that's fine, those only affect speed, but run the suite:

```bash
pytest -q
ruff check edgespark bench data train scripts demo
```

The correctness-critical cores (acceptance, calibration, policy) are pure numpy
and run in CI without a GPU. Keep them that way, put torch behind lazy imports so
the suite stays runnable on any machine.

## Style

- Match the surrounding code: comments explain *why*, not *what*.
- `ruff` config lives in `pyproject.toml`; the formatter owns line length.
- New numbers in the README or `docs/RESULTS.md` should come from a script
  (`bench/simulate.py` for modelled, `--hardware` for measured), never hand-typed.

## Regenerating artifacts

```bash
python scripts/make_figures.py              # docs/assets/*.svg
python scripts/run_benchmark.py --simulate  # runs/reference/summary.json
```
