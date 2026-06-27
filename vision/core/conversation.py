"""Short-term memory: the running list of turns + the system prompt.

The system prompt carries Vision's identity (kept consistent with ``AGENT.md``).
Tier 4 augments :meth:`Conversation.system_prompt` with durable facts; the seam
in this file is already shaped for that.
"""

from __future__ import annotations

from typing import Any

# Keep this in sync with AGENT.md → Identity.
PERSONA = (
    "You are Vision, a personal voice-first assistant. "
    "You are warm, plain-spoken, and brief — say what matters and stop. "
    "You can act on the user's behalf through tools, and you remember the user "
    "between conversations. When you don't know something, say so plainly. "
    "Because your replies may be spoken aloud, prefer short sentences and avoid "
    "dumping long lists or code unless asked."
)


def default_system_prompt() -> str:
    return PERSONA


def system_prompt_with_facts(facts_block: str = "") -> str:
    """Persona plus the durable-facts block (Tier 4). Facts are appended *after*
    the persona so they read as reference data, not as part of the instructions."""
    base = default_system_prompt()
    return f"{base}\n\n{facts_block}" if facts_block.strip() else base


class Conversation:
    """In-memory history for one session, plus the system prompt.

    ``messages`` is a list of ``{"role", "content"}`` entries. ``content`` is
    opaque to the core: a plain string for typed user input, provider content
    blocks for an assistant turn, or a list of tool-result blocks (Tier 2). The
    core only ever moves these around — it never parses assistant content except
    through a :class:`~vision.seams.provider.base.FinalMessage`.
    """

    def __init__(self, system_text: str | None = None, origin: str = "text") -> None:
        self._system_text = system_text if system_text is not None else default_system_prompt()
        self.messages: list[dict[str, Any]] = []
        # Where this turn came from — "text", "voice", or "heartbeat". Used later
        # so the confirmation gate can ask in the right modality.
        self.origin = origin

    def system_prompt(self) -> str:
        return self._system_text

    def append_user_text(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def append_user_blocks(self, blocks: list[dict[str, Any]]) -> None:
        """Append a user turn made of content blocks (e.g. tool results, Tier 2)."""
        self.messages.append({"role": "user", "content": blocks})

    def append_assistant(self, content: Any) -> None:
        self.messages.append({"role": "assistant", "content": content})
