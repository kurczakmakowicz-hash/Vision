"""A visible audit trail — a plain log of what Vision did and why.

When something surprises you, this is how you find out what happened: which tools
ran, what confirmations were asked, what injection was flagged, what the
heartbeat surfaced. Append-only JSON lines.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _default_path() -> Path:
    return Path(os.environ.get("VISION_VAR_DIR", "var")) / "audit.log"


def _short(value: Any, limit: int = 300) -> Any:
    try:
        value = value.model_dump()
    except AttributeError:
        pass
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


class AuditLog:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else _default_path()

    def _write(self, kind: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def tool_run(self, name: str, args: Any, output: str, is_error: bool) -> None:
        self._write("tool_run", name=name, args=_short(args), output=_short(output), error=is_error)

    def confirmation(self, name: str, args: Any, decision: Any) -> None:
        self._write(
            "confirmation",
            name=name,
            args=_short(args),
            allowed=getattr(decision, "allowed", None),
            reason=getattr(decision, "reason", ""),
        )

    def injection(self, name: str, flags: list[str]) -> None:
        self._write("injection_flagged", tool=name, flags=flags)

    def heartbeat_surface(self, check: str, summary: str, level: str) -> None:
        self._write("heartbeat_surface", check=check, summary=summary, level=level)
