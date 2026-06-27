"""A running model-cost tally, so a runaway loop is visible immediately.

Accumulates token usage across the whole tool loop (not just the first call) and
converts to dollars at the configured per-MTok rates. Persisted, so the total
carries across restarts; a budget warning fires once when crossed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _default_path() -> Path:
    return Path(os.environ.get("VISION_VAR_DIR", "var")) / "cost.json"


@dataclass
class CostSnapshot:
    input_tokens: int
    output_tokens: int
    total_usd: float


class CostTally:
    def __init__(
        self,
        input_per_mtok: float,
        output_per_mtok: float,
        budget_warn_usd: float = 0.0,
        path: str | Path | None = None,
    ) -> None:
        self._in_rate = input_per_mtok
        self._out_rate = output_per_mtok
        self._budget = budget_warn_usd
        self.path = Path(path) if path else _default_path()
        self._input = 0
        self._output = 0
        self._warned = False
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self._input = int(data.get("input_tokens", 0))
                self._output = int(data.get("output_tokens", 0))
            except (json.JSONDecodeError, OSError, ValueError):
                pass

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "input_tokens": self._input,
                    "output_tokens": self._output,
                    "total_usd": round(self.total_usd, 4),
                },
                indent=2,
            )
        )

    def add(self, usage: Any) -> None:
        # cache reads/writes are cheaper in reality; counting them at the input
        # rate keeps the guardrail conservative (never under-reports).
        self._input += (
            getattr(usage, "input_tokens", 0)
            + getattr(usage, "cache_read_input_tokens", 0)
            + getattr(usage, "cache_creation_input_tokens", 0)
        )
        self._output += getattr(usage, "output_tokens", 0)
        self._save()

    @property
    def total_usd(self) -> float:
        return self._input / 1e6 * self._in_rate + self._output / 1e6 * self._out_rate

    def over_budget(self) -> bool:
        return self._budget > 0 and self.total_usd >= self._budget

    def should_warn(self) -> bool:
        """True once, the first time the budget is crossed."""
        if self.over_budget() and not self._warned:
            self._warned = True
            return True
        return False

    def snapshot(self) -> CostSnapshot:
        return CostSnapshot(self._input, self._output, round(self.total_usd, 4))
