"""The hard confirmation gate — between the model choosing a tool and the tool
running. Any consequential action (sends, spends, deletes, overwrites, or
anything flagged) stops and gets an explicit yes first, stating plainly what it
will do. It covers typed, spoken, and heartbeat-initiated actions alike, because
they all flow through the same ``run_turn``.

Confirmation is per-action and never generalizes. A heartbeat-initiated action
can't block forever waiting on a human, so it resolves to the safe default
(deny) rather than hang.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Decision:
    allowed: bool
    description: str
    reason: str = ""


def describe_action(spec: Any, args: Any) -> str:
    """A plain-language statement of what the tool is about to do."""
    try:
        detail = args.model_dump()
    except AttributeError:
        detail = args
    summary = (spec.description or spec.name).split(".")[0].strip()
    return f"{spec.name} — {summary} (with {detail})"


class Asker(Protocol):
    async def ask(self, description: str, origin: str, timeout: float) -> bool: ...


class ConsoleAsker:
    """Ask the user at the console for typed/spoken turns. For heartbeat-initiated
    actions there's no human watching, so it returns the safe default (deny)."""

    async def ask(self, description: str, origin: str, timeout: float) -> bool:
        if origin == "heartbeat":
            return False  # never block the heartbeat on a human; do nothing
        loop = asyncio.get_running_loop()
        prompt = f"\n[confirm] Vision wants to: {description}\n          Allow? [y/N] "
        try:
            answer = await loop.run_in_executor(None, input, prompt)
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().lower() in ("y", "yes")


class Gate:
    def __init__(
        self,
        consequential_tools: list[str] | set[str],
        asker: Asker,
        *,
        timeout: float = 120,
    ) -> None:
        self._consequential = set(consequential_tools)
        self._asker = asker
        self._timeout = timeout

    def requires(self, spec: Any) -> bool:
        """A tool is gated if it self-flags, or it's on the consequential list."""
        return bool(getattr(spec, "requires_confirmation", False)) or (
            spec.name in self._consequential
        )

    async def confirm(self, spec: Any, args: Any, origin: str) -> Decision:
        description = describe_action(spec, args)
        allowed = await self._asker.ask(description, origin, self._timeout)
        if allowed:
            reason = ""
        elif origin == "heartbeat":
            reason = "auto-denied (heartbeat — no confirmation possible)"
        else:
            reason = "declined by the user"
        return Decision(allowed=allowed, description=description, reason=reason)
