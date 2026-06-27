"""Wires the pieces together and owns the event loop.

Tier 1 builds the provider + conversation and runs the REPL. Later tiers start
the heartbeat as a second task on this same loop and attach voice as another
way in/out — without changing the brain.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from vision.config import load_config
from vision.core.conversation import Conversation, system_prompt_with_facts
from vision.memory.store import FactStore
from vision.seams.provider.anthropic import AnthropicProvider
from vision.tools.registry import discover_tools
from vision.ui import repl


async def main() -> None:
    load_dotenv()  # pulls ANTHROPIC_API_KEY (and later keys) from a git-ignored .env
    config = load_config()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "[setup] ANTHROPIC_API_KEY isn't set. Copy .env.example to .env and add "
            "your key. Vision will still start, but turns will fail until it's set.\n"
        )

    provider = AnthropicProvider(model=config.model, effort=config.effort)
    registry = discover_tools()

    # Walk into every conversation already knowing the durable facts.
    facts_block = FactStore().facts_block()
    conversation = Conversation(
        system_text=system_prompt_with_facts(facts_block), origin="text"
    )

    await repl.run(conversation, provider, registry, config)
