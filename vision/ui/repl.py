"""The text REPL — Tier 1's interface, and the one that stays alive forever.

It's how every future change gets debugged without talking to the computer, and
the graceful fallback when audio misbehaves. Input runs in a thread executor so
the event loop stays free (the heartbeat task shares it from Tier 5 on).
"""

from __future__ import annotations

import asyncio

from typing import Any

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
    heartbeat_state: Any | None = None,
    sink: Any | None = None,
    *,
    gate: Any | None = None,
    audit: Any | None = None,
    cost: Any | None = None,
    killswitch: Any | None = None,
) -> None:
    loop = asyncio.get_running_loop()
    print(
        "Vision is listening. Type a message, /voice to talk, /notices to see what "
        "came up, or /quit to leave.\n"
    )

    # Catch up on anything surfaced while the interface was closed.
    if heartbeat_state is not None and sink is not None:
        for item in heartbeat_state.undelivered():
            sink.deliver(item)
            heartbeat_state.mark_delivered(item["id"])

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

            await start_voice(
                conversation, provider, registry, config or Config(),
                gate=gate, audit=audit, cost=cost,
            )
            continue
        if text.lower() == "/notices":
            _show_notices(heartbeat_state)
            continue
        if text.lower().startswith("/dismiss"):
            _dismiss(heartbeat_state, text)
            continue
        if text.lower() == "/kill":
            _toggle_kill(killswitch, engage=True)
            continue
        if text.lower() == "/resume":
            _toggle_kill(killswitch, engage=False)
            continue
        if text.lower() == "/cost":
            _show_cost(cost)
            continue

        conversation.append_user_text(user_input)

        print("vision › ", end="", flush=True)
        try:
            await run_turn(
                conversation, provider, on_text=_print_delta,
                registry=registry, gate=gate, audit=audit, cost=cost,
            )
        except ProviderUnavailable as exc:
            # Drop the half-written prompt line and explain, but keep going.
            print(f"\n[vision couldn't reach the model: {exc}. Try again.]")
            continue
        print()  # newline after the streamed reply
        if cost is not None and cost.should_warn():
            print(f"[vision 💸 cost warning: ${cost.total_usd:.2f} so far this run]")


def _print_delta(delta: str) -> None:
    print(delta, end="", flush=True)


def _show_notices(state: Any | None) -> None:
    if state is None:
        print("[no heartbeat running]")
        return
    items = state.held()
    if not items:
        print("No notices.")
        return
    for h in items:
        print(f"  {h['id']}  [{h['level']}]  {h['summary']}")


def _dismiss(state: Any | None, text: str) -> None:
    parts = text.split()
    if state is None or len(parts) < 2:
        print("Usage: /dismiss <id>")
        return
    print(f"Dismissed {parts[1]}." if state.dismiss(parts[1]) else f"No notice with id {parts[1]}.")


def _toggle_kill(killswitch: Any | None, *, engage: bool) -> None:
    if killswitch is None:
        print("[no kill switch]")
        return
    if engage:
        killswitch.engage()
        print("Kill switch ENGAGED — proactive behavior is paused. Chat still works.")
    else:
        killswitch.release()
        print("Kill switch released — proactive behavior resumed.")


def _show_cost(cost: Any | None) -> None:
    if cost is None:
        print("[no cost tally]")
        return
    s = cost.snapshot()
    print(f"Model cost so far: ${s.total_usd:.4f}  ({s.input_tokens} in / {s.output_tokens} out tokens)")
