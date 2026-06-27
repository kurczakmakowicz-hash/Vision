"""Durable heartbeat state: per-check next-due times + the held-notice queue.

Persisting next-due to disk is what stops a restart from resetting every timer or
firing everything at once on boot. Held notices are never delivered-and-lost: a
surfaced item lands here first and is shown when the user is back.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


def _default_path() -> Path:
    return Path(os.environ.get("VISION_VAR_DIR", "var")) / "heartbeat_state.json"


class HeartbeatState:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else _default_path()
        self._data: dict[str, Any] = {"checks": {}, "held": []}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        self._data.setdefault("checks", {})
        self._data.setdefault("held", [])

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2))

    # --- per-check schedule ---------------------------------------------------

    def get_next_due(self, name: str) -> float | None:
        entry = self._data["checks"].get(name)
        return entry.get("next_due") if entry else None

    def set_next_due(self, name: str, when: float) -> None:
        self._data["checks"].setdefault(name, {})["next_due"] = when
        self._save()

    def is_running(self, name: str) -> bool:
        entry = self._data["checks"].get(name)
        return bool(entry and entry.get("running"))

    def set_running(self, name: str, value: bool) -> None:
        self._data["checks"].setdefault(name, {})["running"] = bool(value)
        self._save()

    # --- held notices ---------------------------------------------------------

    def add_held(self, *, check: str, summary: str, level: str, ts: float) -> dict[str, Any]:
        item = {
            "id": uuid.uuid4().hex[:8],
            "check": check,
            "summary": summary,
            "level": level,
            "ts": ts,
            "delivered": False,
            "dismissed": False,
        }
        self._data["held"].append(item)
        self._save()
        return item

    def held(self, include_dismissed: bool = False) -> list[dict[str, Any]]:
        return [
            h for h in self._data["held"] if include_dismissed or not h.get("dismissed")
        ]

    def undelivered(self) -> list[dict[str, Any]]:
        """Items surfaced while no interface was attached — shown on return."""
        return [
            h
            for h in self._data["held"]
            if not h.get("delivered") and not h.get("dismissed")
        ]

    def mark_delivered(self, item_id: str) -> None:
        for h in self._data["held"]:
            if h["id"] == item_id:
                h["delivered"] = True
        self._save()

    def dismiss(self, item_id: str) -> bool:
        found = False
        for h in self._data["held"]:
            if h["id"] == item_id and not h.get("dismissed"):
                h["dismissed"] = True
                found = True
        if found:
            self._save()
        return found
