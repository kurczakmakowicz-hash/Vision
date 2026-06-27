"""The agent loop — the one entry point every turn flows through.

This is the hand-rolled manual tool-use loop (not the SDK's ``tool_runner``),
because the confirmation gate (Tier 6) must sit *between* the model choosing a
tool and the tool running — and ``tool_runner`` auto-executes. The shape:

    while the model keeps asking for tools:
        stream the reply out, record it
        for each requested tool:  validate → GATE → execute → collect result
        send all results back in ONE user message

The ``gate``/``audit``/``cost`` parameters are wired in by later tiers; until
then they're ``None`` and the gate is a pass-through. With no ``registry`` the
loop is exactly Tier 1 (one streamed reply, no tools).
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from pydantic import ValidationError

from vision.core.conversation import Conversation
from vision.seams.provider.base import FinalMessage, Provider, ToolUse
from vision.tools.registry import Registry, ToolSpec


async def run_turn(
    conversation: Conversation,
    provider: Provider,
    on_text: Callable[[str], None],
    *,
    registry: Registry | None = None,
    gate: Any | None = None,
    audit: Any | None = None,
    cost: Any | None = None,
) -> FinalMessage:
    """Run one turn to completion, including any chain of tool calls."""
    tools = registry.api_schemas() if registry else []

    while True:
        reply = await provider.respond(conversation, tools=tools)

        async for delta in reply.text_deltas:
            on_text(delta)

        final = await reply.final()
        if cost is not None:
            cost.add(final.usage)  # running model-cost tally (Tier 6)
        conversation.append_assistant(final.assistant_content)

        if final.stop_reason != "tool_use" or registry is None:
            return final

        # The model may have asked for several tools at once.
        tool_results = [
            await _run_one_tool(tu, registry, gate, audit, conversation)
            for tu in final.tool_use_blocks()
        ]
        # ALL results for this assistant turn go back in ONE user message.
        conversation.append_user_blocks(tool_results)
        # Loop: the model sees the results and may answer or call more tools.


async def _run_one_tool(
    tu: ToolUse,
    registry: Registry,
    gate: Any | None,
    audit: Any | None,
    conversation: Conversation,
) -> dict[str, Any]:
    spec = registry.get(tu.name)
    if spec is None:
        return _error_result(tu.id, f"There is no tool named '{tu.name}'.")

    # 1. Validate inputs (typed, not freeform). A bad input becomes a plain-language
    #    error TO THE MODEL — it can correct itself; the app never crashes.
    try:
        args = spec.input_model.model_validate(tu.input)
    except ValidationError as exc:
        return _error_result(tu.id, _friendly_validation(spec, exc))

    # 2. THE GATE — between tool-choice and tool-execution. Wired in Tier 6; until
    #    then ``gate`` is None and this is a pass-through.
    if gate is not None and spec.requires_confirmation:
        decision = await gate.confirm(spec, args, origin=conversation.origin)
        if audit is not None:
            audit.confirmation(spec.name, args, decision)
        if not getattr(decision, "allowed", False):
            return _error_result(tu.id, "The user declined this action.")

    # 3. Execute. Any failure becomes an error result — never an exception out.
    try:
        result = spec.handler(args)
        if inspect.isawaitable(result):
            result = await result
        output = str(result)
        is_error = False
    except Exception as exc:  # noqa: BLE001 — tool failures are data for the model
        output = f"The tool failed: {exc}"
        is_error = True

    if audit is not None:
        audit.tool_run(spec.name, args, output, is_error=is_error)

    # 4. (Tier 6) external-content tools get an injection scan here.

    return _error_result(tu.id, output) if is_error else _result_block(tu.id, output)


def _result_block(tool_use_id: str, content: str) -> dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


def _error_result(tool_use_id: str, content: str) -> dict[str, Any]:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": True,
    }


def _friendly_validation(spec: ToolSpec, exc: ValidationError) -> str:
    problems = "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or '(input)'}: {e['msg']}"
        for e in exc.errors()
    )
    return f"Invalid input for '{spec.name}': {problems}."
