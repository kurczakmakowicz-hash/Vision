"""The text REPL — Tier 1's interface, and the one that stays alive forever.

It's how every future change gets debugged without talking to the computer, and
the graceful fallback when audio misbehaves. Input runs in a thread executor so
the event loop stays free (the heartbeat task shares it from Tier 5 on).
"""

from __future__ import annotations

import asyncio

from vision.config import Config
from vision.core.agent import run_turn
from vision.core.conversation import Conversation
from vision.core.errors import ProviderUnavailable
from vision.seams.provider.base import Provider
from vision.tools.registry import Registry

_QUIT = {"/quit", "/exit", "/q"}
_PROMPT = "you › "


async def _read_line(loop: asyncio.AbstractEventLoop) -> str:
    """Read a line without blocking the event loop."""
    return await loop.run_in_executor(None, input, _PROMPT)


async def run(
    conversation: Conversation,
    provider: Provider,
    registry: Registry | None = None,
    config: Config | None = None,
) -> None:
    loop = asyncio.get_running_loop()
    print("Vision is listening. Type a message, /voice to talk, or /quit to leave.\n")

    while True:
        try:
            user_input = await _read_line(loop)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        text = user_input.strip()
        if not text:
            continue
        if text.lower() in _QUIT:
            print("Goodbye.")
            return
        if text.lower() == "/voice":
            from vision.voice.launch import start_voice

            await start_voice(conversation, provider, registry, config or Config())
            continue

        conversation.append_user_text(user_input)

        print("vision › ", end="", flush=True)
        try:
            await run_turn(
                conversation, provider, on_text=_print_delta, registry=registry
            )
        except ProviderUnavailable as exc:
            # Drop the half-written prompt line and explain, but keep going.
            print(f"\n[vision couldn't reach the model: {exc}. Try again.]")
            continue
        print()  # newline after the streamed reply


def _print_delta(delta: str) -> None:
    print(delta, end="", flush=True)
