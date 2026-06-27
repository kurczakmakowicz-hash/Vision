"""Provider seam — the only contract the brain depends on for "thinking".

One small surface: "send this conversation (plus available tools), get back a
streaming reply and, once consumed, a normalized final message." Everything else
in Vision calls this and never touches a vendor SDK. Swapping models = adding one
file next to ``anthropic.py`` that implements :class:`Provider`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class Usage:
    """Token usage for one provider call (drives the Tier 6 cost tally)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class ToolUse:
    """A normalized request from the model to call one tool (Tier 2+)."""

    id: str
    name: str
    input: dict[str, Any]


@runtime_checkable
class FinalMessage(Protocol):
    """The completed assistant turn, normalized across providers."""

    @property
    def stop_reason(self) -> str:
        """e.g. ``"end_turn"``, ``"tool_use"``, ``"refusal"``."""

    @property
    def usage(self) -> Usage: ...

    @property
    def assistant_content(self) -> Any:
        """Opaque content to append to the conversation as the assistant turn.

        The core never inspects this; it only moves it back into history so the
        next request replays the full assistant turn (text, tool_use, and — for
        models that produce them — thinking blocks with their signatures intact).
        """

    def text(self) -> str:
        """Concatenated visible text of the turn."""

    def tool_use_blocks(self) -> list[ToolUse]:
        """Tool calls the model requested this turn (empty unless tool_use)."""


@dataclass
class Reply:
    """A streaming reply. Consume ``text_deltas``, then ``await final()``."""

    text_deltas: AsyncIterator[str]
    _final: Callable[[], Awaitable[FinalMessage]] = field(repr=False)

    async def final(self) -> FinalMessage:
        return await self._final()


@runtime_checkable
class ConversationView(Protocol):
    """The slice of a conversation a provider needs — keeps the seam free of
    any dependency on :mod:`vision.core.conversation`."""

    @property
    def messages(self) -> list[dict[str, Any]]: ...

    def system_prompt(self) -> str: ...


class Provider(Protocol):
    """Send a conversation, get a streaming :class:`Reply`."""

    async def respond(
        self, conversation: ConversationView, tools: list[dict[str, Any]]
    ) -> Reply:
        """``tools`` is a list of already-rendered tool schemas (empty in Tier 1)."""
        ...
