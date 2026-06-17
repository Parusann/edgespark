"""A tiny dependency-free SVG chart primitive set.

matplotlib is the right tool for interactive analysis on the GPU box
(``bench/plots.py``), but the figures committed to the repo and embedded in the
README should render anywhere — including GitHub's markdown viewer — with no build
step and no binary blobs in git history. So the committed figures are hand-built
SVG. This module is the minimal primitive layer; ``scripts/make_figures.py`` uses
it to draw the four key charts from ``bench.simulate``.

Pure numpy/stdlib. The palette is chosen to read on both light and dark GitHub
themes (mid-tone fills, explicit stroke on text via CSS is avoided).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape

# Palette — colourblind-friendly, legible on light and dark backgrounds.
INK = "#1f2430"
MUTED = "#6b7280"
GRID = "#e5e7eb"
ACCENT = "#2563eb"     # EdgeSpark blue
ACCENT2 = "#f59e0b"    # amber (baseline / secondary)
GOOD = "#059669"       # green (recalibrated)
BAD = "#dc2626"        # red (miscalibrated)
FILL_BG = "#ffffff"
SERIES = ["#2563eb", "#059669", "#dc2626", "#f59e0b", "#7c3aed", "#0891b2"]


@dataclass
class Canvas:
    width: int
    height: int
    parts: list[str] = field(default_factory=list)

    def _add(self, s: str) -> None:
        self.parts.append(s)

    def rect(self, x, y, w, h, fill, *, rx=0, opacity=1.0, stroke="none", sw=1.0):
        self._add(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def line(self, x1, y1, x2, y2, stroke=GRID, sw=1.0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self._add(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>'
        )

    def polyline(self, points, stroke=ACCENT, sw=2.0, fill="none"):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        self._add(f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def circle(self, x, y, r, fill=ACCENT, stroke="none", sw=1.0):
        self._add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def text(self, x, y, s, *, size=13, fill=INK, anchor="start", weight="normal", style=""):
        self._add(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif"{style}>{escape(s)}</text>'
        )

    def render(self, title: str = "") -> str:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" '
            f'width="{self.width}" height="{self.height}" role="img" aria-label="{escape(title)}">'
        )
        bg = f'<rect width="{self.width}" height="{self.height}" fill="{FILL_BG}"/>'
        return head + bg + "".join(self.parts) + "</svg>"


def save(canvas: Canvas, path, title: str = "") -> None:
    from pathlib import Path

    Path(path).write_text(canvas.render(title), encoding="utf-8")
