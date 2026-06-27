"""The one place surfaced items reach the user.

A Protocol so the heartbeat doesn't care which machine it's on: a console sink
today, a push/webhook sink later — the same loop, a different sink.
"""

from __future__ import annotations

from typing import Any, Protocol


class NotificationSink(Protocol):
    def deliver(self, item: dict[str, Any]) -> None: ...


class NullSink:
    """No interface attached — items stay held for catch-up on return."""

    def deliver(self, item: dict[str, Any]) -> None:
        pass


class ConsoleSink:
    """Prints a surfaced item into the text REPL."""

    def deliver(self, item: dict[str, Any]) -> None:
        icon = "‼️" if item.get("level") == "critical" else "🔔"
        print(
            f"\n[vision {icon}] {item['summary']}  "
            f"(id {item['id']} — /dismiss {item['id']} to clear)\n",
            end="",
            flush=True,
        )
