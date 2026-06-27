"""Anthropic implementation of the :class:`Provider` seam.

The only file that imports the ``anthropic`` SDK. It also owns the model-param
rules for ``claude-opus-4-8`` (adaptive thinking + effort; no temperature/top_p/
top_k, no ``budget_tokens``, no last-assistant-turn prefill) and translates the
SDK's exceptions into our seam errors so the core stays vendor-agnostic.
"""

from __future__ import annotations

import os
from typing import Any

import anthropic

from vision.core.errors import ProviderAuthError, ProviderUnavailable
from vision.seams.provider.base import (
    ConversationView,
    FinalMessage,
    Reply,
    ToolUse,
    Usage,
)

# Streaming is mandatory (large max_tokens otherwise risks SDK HTTP timeouts) and
# is also what the voice layer taps in Tier 3.
_MAX_TOKENS = 16000


def _translate(exc: BaseException) -> BaseException:
    """Map an anthropic SDK exception onto our seam errors.

    Unknown exceptions are returned unchanged so genuine bugs aren't swallowed.
    """
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return ProviderAuthError("the model rejected the API key — check ANTHROPIC_API_KEY")
    if isinstance(
        exc,
        (
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        ),
    ):
        return ProviderUnavailable(str(exc) or type(exc).__name__)
    if isinstance(exc, anthropic.APIStatusError):
        return ProviderUnavailable(f"API error {getattr(exc, 'status_code', '?')}")
    if isinstance(exc, anthropic.APIError):
        return ProviderUnavailable(str(exc) or type(exc).__name__)
    return exc


class _AnthropicFinalMessage:
    """Normalizes a completed SDK message to the :class:`FinalMessage` contract."""

    def __init__(self, msg: Any) -> None:
        self._msg = msg

    @property
    def stop_reason(self) -> str:
        return self._msg.stop_reason or "end_turn"

    @property
    def usage(self) -> Usage:
        u = self._msg.usage
        return Usage(
            input_tokens=getattr(u, "input_tokens", 0) or 0,
            output_tokens=getattr(u, "output_tokens", 0) or 0,
            cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )

    @property
    def assistant_content(self) -> Any:
        # Append the raw SDK content blocks back into history verbatim — preserves
        # tool_use blocks and any thinking-block signatures for replay.
        return self._msg.content

    def text(self) -> str:
        return "".join(
            b.text for b in self._msg.content if getattr(b, "type", None) == "text"
        )

    def tool_use_blocks(self) -> list[ToolUse]:
        return [
            ToolUse(id=b.id, name=b.name, input=b.input)
            for b in self._msg.content
            if getattr(b, "type", None) == "tool_use"
        ]


class AnthropicProvider:
    """Streaming Anthropic provider. Satisfies :class:`Provider` structurally."""

    def __init__(self, model: str, effort: str, api_key: str | None = None) -> None:
        self._model = model
        self._effort = effort
        # Use a placeholder if no key so construction never raises — the failure
        # then surfaces per-turn as a clean ProviderAuthError and the REPL lives on.
        key = api_key or os.environ.get("ANTHROPIC_API_KEY") or "MISSING_ANTHROPIC_API_KEY"
        self._client = anthropic.AsyncAnthropic(api_key=key)

    def _build_kwargs(
        self, conversation: ConversationView, tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(
            model=self._model,
            max_tokens=_MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": self._effort},
            system=conversation.system_prompt(),
            messages=conversation.messages,
        )
        if tools:
            kwargs["tools"] = tools
        return kwargs

    async def respond(
        self, conversation: ConversationView, tools: list[dict[str, Any]]
    ) -> Reply:
        kwargs = self._build_kwargs(conversation, tools)
        try:
            manager = self._client.messages.stream(**kwargs)
            stream = await manager.__aenter__()
        except Exception as exc:  # noqa: BLE001 — translated, not swallowed
            raise _translate(exc) from exc

        closed = False

        async def _close() -> None:
            nonlocal closed
            if not closed:
                closed = True
                try:
                    await manager.__aexit__(None, None, None)
                except Exception:  # noqa: BLE001 — cleanup must not mask the real error
                    pass

        async def _text_deltas():
            try:
                async for text in stream.text_stream:
                    yield text
            except Exception as exc:  # noqa: BLE001
                await _close()
                raise _translate(exc) from exc

        async def _final() -> FinalMessage:
            try:
                msg = await stream.get_final_message()
            except Exception as exc:  # noqa: BLE001
                await _close()
                raise _translate(exc) from exc
            await _close()
            return _AnthropicFinalMessage(msg)

        return Reply(text_deltas=_text_deltas(), _final=_final)
