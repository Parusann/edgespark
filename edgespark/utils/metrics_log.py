"""JSON-lines metrics logging (spec section 15).

One record per generation step. Append-only JSONL so a long run can be streamed
to disk and analysed later without holding it all in 32 GB of RAM, and so a
crashed run still leaves everything up to the crash on disk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StepMetrics:
    """The exact record shape from spec section 15."""

    step: int
    ell: int
    accepted: int
    tau: int
    t_draft_ms: float
    t_verify_ms: float
    vram_mb: float
    conf_profile: list[float]
    precision: str
    exact_ok: bool
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        extra = d.pop("extra")
        d.update(extra)
        return json.dumps(d)


class MetricsLogger:
    """Context-managed JSONL writer."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def __enter__(self) -> MetricsLogger:
        self._fh = self.path.open("w", encoding="utf-8")
        return self

    def log(self, record: StepMetrics) -> None:
        assert self._fh is not None, "use MetricsLogger as a context manager"
        self._fh.write(record.to_json() + "\n")

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a metrics file back into a list of dicts for offline analysis."""
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
