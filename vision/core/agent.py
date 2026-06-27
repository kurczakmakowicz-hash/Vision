"""The agent loop — the one entry point every turn flows through.

Tier 1: a single provider call, streamed out, appended to history. Tier 2 wraps
this in a ``while`` loop and adds the tool branch (validate → gate → execute →
return results), so the same ``run_turn`` serves typed, spoken, and
heartbeat-initiated turns. The shape here is deliberately the inner body of that
future loop.
"""

from __future__ import annotations

from typing import Callable

from vision.core.conversation import Conversation
from vision.seams.provider.base import FinalMessage, Provider


async def run_turn(
    conversation: Conversation,
    provider: Provider,
    on_text: Callable[[str], None],
) -> FinalMessage:
    """Run one turn: stream the reply out via ``on_text``, record it, return it.

    ``on_text`` receives text deltas as they arrive — stdout in Tier 1, the TTS
    sink in Tier 3. No tools yet, so the model always stops with ``end_turn``.
    """
    reply = await provider.respond(conversation, tools=[])

    async for delta in reply.text_deltas:
        on_text(delta)

    final = await reply.final()
    conversation.append_assistant(final.assistant_content)
    return final
