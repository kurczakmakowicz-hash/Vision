"""Test doubles that implement the seam protocols without any network or audio.

These let us drive the real agent loop deterministically — proving the loop,
tool dispatch, gate, etc. — without an API key or a microphone.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from pydantic import BaseModel

from vision.seams.provider.base import Reply, ToolUse, Usage
from vision.seams.stt.base import Transcript
from vision.tools.registry import Registry, ToolSpec


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


# --- voice doubles ------------------------------------------------------------


class FakeSTT:
    """Yields scripted final transcripts, ignoring the audio frames."""

    def __init__(self, finals: list[str]) -> None:
        self._finals = finals

    async def transcribe(self, frames: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        async for _ in frames:  # drain (a real STT would consume these)
            pass
        for text in self._finals:
            yield Transcript(text=text, is_final=True)


class FakeTTS:
    """Turns each text chunk into bytes and records the chunks it was given."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def speak(self, text_chunks: AsyncIterator[str]) -> AsyncIterator[bytes]:
        async for chunk in text_chunks:
            self.spoken.append(chunk)
            yield chunk.encode("utf-8")


class FakePlayback:
    """Collects written bytes; tracks clear() for barge-in assertions."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.cleared = 0

    def write(self, data: bytes) -> None:
        self.buf += data

    def clear(self) -> None:
        self.cleared += 1
        self.buf = bytearray()

    def is_empty(self) -> bool:
        return len(self.buf) == 0


async def empty_frames() -> AsyncIterator[bytes]:
    """An audio source that yields nothing (FakeSTT supplies the transcript)."""
    return
    yield  # pragma: no cover — makes this an async generator


# --- shared tool helpers (used by Tier 2 and Tier 3 tests) --------------------


class EchoIn(BaseModel):
    text: str


def make_echo_registry() -> Registry:
    reg = Registry()
    reg.register(
        ToolSpec(
            name="echo",
            description="Echo the text back.",
            input_model=EchoIn,
            handler=lambda a: f"echo:{a.text}",
        )
    )
    return reg


def tool_use_turn(*tool_uses: ToolUse) -> FakeFinalMessage:
    return FakeFinalMessage(
        content=[{"type": "text", "text": "(calling tools)"}],
        stop_reason="tool_use",
        tool_uses=list(tool_uses),
    )
