"""Wires the pieces together and owns the event loop.

Tier 1 builds the provider + conversation and runs the REPL. Later tiers start
the heartbeat as a second task on this same loop and attach voice as another
way in/out — without changing the brain.
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from vision.config import load_config
from vision.core.conversation import Conversation, system_prompt_with_facts
from vision.heartbeat.checks import build_checks
from vision.heartbeat.loop import HeartbeatLoop
from vision.heartbeat.quiet import is_quiet
from vision.heartbeat.sink import ConsoleSink
from vision.heartbeat.state import HeartbeatState
from vision.memory.store import FactStore
from vision.rails.audit import AuditLog
from vision.rails.cost import CostTally
from vision.rails.gate import ConsoleAsker, Gate
from vision.rails.killswitch import KillSwitch
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

    # The rails: gate, audit trail, cost tally, kill switch.
    audit = AuditLog()
    cost = CostTally(
        config.cost.input_per_mtok,
        config.cost.output_per_mtok,
        config.cost.budget_warn_usd,
    )
    gate = Gate(
        config.rails.consequential_tools,
        ConsoleAsker(),
        timeout=config.heartbeat.confirm_timeout_seconds,
    )
    kill = KillSwitch(default=config.rails.kill_switch)

    # The heartbeat runs as a separate task on this same loop (relocatable later).
    hb_state = HeartbeatState()
    sink = ConsoleSink()
    qh = config.heartbeat.quiet_hours
    heartbeat = HeartbeatLoop(
        build_checks(config.heartbeat.checks),
        hb_state,
        sink,
        interval_seconds=config.heartbeat.interval_seconds,
        is_quiet=lambda now: is_quiet(qh.start, qh.end, now),
        is_paused=lambda: kill.engaged,  # the kill switch halts proactive behavior
    )
    stop = asyncio.Event()
    hb_task = asyncio.create_task(heartbeat.run(stop))

    try:
        await repl.run(
            conversation, provider, registry, config, hb_state, sink,
            gate=gate, audit=audit, cost=cost, killswitch=kill,
        )
    finally:
        stop.set()
        await hb_task
