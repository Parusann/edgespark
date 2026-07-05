# Paper

**EdgeSpark: Quantized Semi-Autoregressive Speculative Decoding and
Confidence-Head Calibration on a Single Consumer AMD GPU** — the technical
preprint. [`edgespark.pdf`](edgespark.pdf) is the compiled version.

## Build

The paper uses standard LaTeX packages (pgfplots, natbib, newtx, amsthm), so it
compiles anywhere:

```bash
# Tectonic (single self-contained binary, no TeX install):
tectonic edgespark.tex

# or a full TeX distribution:
latexmk -pdf edgespark.tex        # runs pdflatex + bibtex as needed

# or Overleaf: upload edgespark.tex + refs.bib, set the compiler to pdfLaTeX.
```

Figures are native pgfplots/TikZ drawn from the project's own data (no external
image files), so there is nothing to regenerate — the reliability curves in
Figure 2 are the same numbers `scripts/make_figures.py` produces.

## A note on honesty

Every number in the paper is labelled **measured** (on the RX 7900 XTX) or
**modelled** (design-time). The evidence ledger in Table 2 and Section 6 makes the
split explicit, and Section 8 states plainly what the paper does *not* establish
(no trained drafter; INT8/NF4 and the calibration study pending a
bitsandbytes-capable stack). This mirrors the repository's `docs/RESULTS.md` §0.
