"""Scheduled checks: each is a small unit (name + interval + handler).

A handler runs, looks at something, and decides whether the result is worth
surfacing — returning a :class:`Notice` (or ``None`` for "nothing to say", which
is most of the time, by design). Handlers register by name and are resolved from
config, so *what to check* and *how often* live in config, not code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

# calm   → accumulates in the calm log, shown when the user looks
# interrupt → delivered now (unless quiet hours), else held for return
# critical  → delivered now even during quiet hours
Level = str


@dataclass
class Notice:
    summary: str
    level: Level = "calm"


@dataclass
class Check:
    name: str
    interval_seconds: int
    handler: Callable[[], "Notice | None | Awaitable[Notice | None]"]


_HANDLERS: dict[str, Callable[[], Any]] = {}


def check_handler(name: str) -> Callable[[Callable], Callable]:
    def decorator(fn: Callable) -> Callable:
        _HANDLERS[name] = fn
        return fn

    return decorator


def build_checks(check_configs: list[Any]) -> list[Check]:
    """Resolve config check entries to registered handlers (skip unknown ones)."""
    checks: list[Check] = []
    for cc in check_configs:
        handler = _HANDLERS.get(cc.handler)
        if handler is None:
            print(
                f"[heartbeat] no check handler named '{cc.handler}' "
                f"(for '{cc.name}') — skipping"
            )
            continue
        checks.append(Check(cc.name, cc.interval_seconds, handler))
    return checks


# --- built-in handlers --------------------------------------------------------


def _trigger_path() -> Path:
    return Path(os.environ.get("VISION_VAR_DIR", "var")) / "trigger"


@check_handler("watch_trigger")
def watch_trigger() -> Notice | None:
    """Surface a notice if a trigger file appears, then consume it (fires once).

    A hands-on way to verify the heartbeat: ``echo "ping" > var/trigger`` and the
    next tick surfaces "ping". With no file, it returns nothing — quiet by default.
    """
    path = _trigger_path()
    if not path.exists():
        return None
    try:
        message = path.read_text().strip() or "Heartbeat trigger fired."
    except OSError:
        message = "Heartbeat trigger fired."
    try:
        path.unlink()
    except OSError:
        pass
    return Notice(summary=message, level="interrupt")
