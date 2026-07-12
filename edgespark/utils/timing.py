"""Timers for the two costs that decide whether speculation pays off.

The per-token latency model is ``(T_draft + T_verify) / tau`` (spec Appendix B),
so ``T_draft`` and ``T_verify`` have to be measured honestly. On GPU that means
synchronising around the region, wall-clock around an async kernel launch is a
classic way to report numbers that are far too good. ``GpuTimer`` handles the
sync; ``WallTimer`` is the CPU fallback used in the numpy reference path.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class WallTimer:
    """Accumulates elapsed wall-clock milliseconds under a named region."""

    totals_ms: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def region(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            dt = (time.perf_counter() - start) * 1e3
            self.totals_ms[name] = self.totals_ms.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1

    def mean_ms(self, name: str) -> float:
        n = self.counts.get(name, 0)
        return self.totals_ms.get(name, 0.0) / n if n else 0.0

    def reset(self) -> None:
        self.totals_ms.clear()
        self.counts.clear()


class GpuTimer:
    """CUDA/HIP event timer that synchronises before reading elapsed time.

    Falls back to :class:`WallTimer` semantics when torch or a GPU is absent, so
    the same benchmark code runs on the CPU reference path without branching.
    """

    def __init__(self) -> None:
        self._wall = WallTimer()
        try:
            import torch

            self._torch = torch
            self._gpu = torch.cuda.is_available()
        except Exception:  # torch not installed (e.g. the CPU test box)
            self._torch = None
            self._gpu = False

    @contextmanager
    def region(self, name: str):
        if not self._gpu:
            with self._wall.region(name):
                yield
            return
        torch = self._torch
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            torch.cuda.synchronize()
            dt = start.elapsed_time(end)  # milliseconds, already synced
            self._wall.totals_ms[name] = self._wall.totals_ms.get(name, 0.0) + dt
            self._wall.counts[name] = self._wall.counts.get(name, 0) + 1

    def mean_ms(self, name: str) -> float:
        return self._wall.mean_ms(name)

    def reset(self) -> None:
        self._wall.reset()
