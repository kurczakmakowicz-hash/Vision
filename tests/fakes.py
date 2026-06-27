"""Test doubles that implement the seam protocols without any network or audio.

These let us drive the real agent loop deterministically — proving the loop,
tool dispatch, gate, etc. — without an API key or a microphone.
"""

from __future__ import annotations

from typing import Any

from vision.seams.provider.base import Reply, ToolUse, Usage


class FakeFinalMessage:
    """A :class:`FinalMessage` whose content is plain dicts (no SDK objects)."""

    def __init__(
        self,
        content: list[dict[str, Any]],
        stop_reason: str = "end_turn",
        usage: Usage | None = None,
        tool_uses: list[ToolUse] | None = None,
    ) -> None:
        self._content = content
        self._stop_reason = stop_reason
        self._usage = usage or Usage(input_tokens=10, output_tokens=5)
        self._tool_uses = tool_uses or []

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    @property
    def usage(self) -> Usage:
        return self._usage

    @property
    def assistant_content(self) -> Any:
        return self._content

    def text(self) -> str:
        return "".join(
            b["text"] for b in self._content if b.get("type") == "text"
        )

    def tool_use_blocks(self) -> list[ToolUse]:
        return list(self._tool_uses)


class FakeProvider:
    """Replays a scripted list of ``(deltas, final_message)`` turns.

    Records, per call, a snapshot of the messages and tools it was handed — so a
    test can assert the full history is replayed each turn.
    """

    def __init__(self, scripted: list[tuple[list[str], FakeFinalMessage]]) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict[str, Any]] = []

    async def respond(self, conversation: Any, tools: list[dict[str, Any]]) -> Reply:
        self.calls.append(
            {"messages": list(conversation.messages), "tools": list(tools)}
        )
        deltas, final = self._scripted.pop(0)

        async def _text_deltas():
            for d in deltas:
                yield d

        async def _final() -> FakeFinalMessage:
            return final

        return Reply(text_deltas=_text_deltas(), _final=_final)
