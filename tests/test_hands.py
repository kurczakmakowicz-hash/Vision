"""Tier 2 verification: the agent can call tools, chain them, and survive a
failing one — and the first real tools register with valid schemas."""

from __future__ import annotations

from pydantic import BaseModel

from tests.fakes import FakeFinalMessage, FakeProvider
from vision.core.agent import run_turn
from vision.core.conversation import Conversation
from vision.seams.provider.base import ToolUse
from vision.tools.registry import Registry, ToolSpec, discover_tools


class EchoIn(BaseModel):
    text: str


def _boom(_args: EchoIn) -> str:
    raise RuntimeError("kaboom")


def _make_registry() -> Registry:
    reg = Registry()
    reg.register(
        ToolSpec(
            name="echo",
            description="Echo the text back.",
            input_model=EchoIn,
            handler=lambda a: f"echo:{a.text}",
        )
    )
    reg.register(
        ToolSpec(
            name="boom",
            description="Always fails.",
            input_model=EchoIn,
            handler=_boom,
        )
    )
    return reg


def _tool_use_turn(*tool_uses: ToolUse) -> FakeFinalMessage:
    return FakeFinalMessage(
        content=[{"type": "text", "text": "(calling tools)"}],
        stop_reason="tool_use",
        tool_uses=list(tool_uses),
    )


async def test_tool_is_called_and_result_fed_back_to_the_model():
    reg = _make_registry()
    provider = FakeProvider(
        [
            ([], _tool_use_turn(ToolUse("t1", "echo", {"text": "hi"}))),
            (["done"], FakeFinalMessage([{"type": "text", "text": "done"}])),
        ]
    )
    conv = Conversation()
    conv.append_user_text("please echo hi")

    final = await run_turn(conv, provider, on_text=lambda d: None, registry=reg)

    assert final.stop_reason == "end_turn"
    # messages: user(text), assistant(tool_use), user(tool_result), assistant(text)
    tool_result_turn = conv.messages[2]
    assert tool_result_turn["role"] == "user"
    block = tool_result_turn["content"][0]
    assert block["type"] == "tool_result"
    assert block["content"] == "echo:hi"
    assert not block.get("is_error")
    # the second provider call was handed the tool result
    second_call_msgs = provider.calls[1]["messages"]
    assert any(
        isinstance(m["content"], list)
        and m["content"][0].get("type") == "tool_result"
        for m in second_call_msgs
    )


async def test_multiple_tool_calls_in_one_turn_all_run():
    reg = _make_registry()
    provider = FakeProvider(
        [
            (
                [],
                _tool_use_turn(
                    ToolUse("t1", "echo", {"text": "a"}),
                    ToolUse("t2", "echo", {"text": "b"}),
                ),
            ),
            (["done"], FakeFinalMessage([{"type": "text", "text": "done"}])),
        ]
    )
    conv = Conversation()
    conv.append_user_text("echo a and b")

    await run_turn(conv, provider, on_text=lambda d: None, registry=reg)

    results = conv.messages[2]["content"]
    assert len(results) == 2  # both results returned in one user message
    assert {r["content"] for r in results} == {"echo:a", "echo:b"}


async def test_failing_tool_returns_plain_error_and_loop_continues():
    reg = _make_registry()
    provider = FakeProvider(
        [
            ([], _tool_use_turn(ToolUse("t1", "boom", {"text": "x"}))),
            (["recovered"], FakeFinalMessage([{"type": "text", "text": "recovered"}])),
        ]
    )
    conv = Conversation()
    conv.append_user_text("trigger the failure")

    final = await run_turn(conv, provider, on_text=lambda d: None, registry=reg)

    assert final.stop_reason == "end_turn"  # did not crash; recovered
    block = conv.messages[2]["content"][0]
    assert block["is_error"] is True
    assert "kaboom" in block["content"]


async def test_invalid_tool_input_becomes_an_error_result():
    reg = _make_registry()
    provider = FakeProvider(
        [
            ([], _tool_use_turn(ToolUse("t1", "echo", {}))),  # missing required 'text'
            (["ok"], FakeFinalMessage([{"type": "text", "text": "ok"}])),
        ]
    )
    conv = Conversation()
    conv.append_user_text("bad call")

    await run_turn(conv, provider, on_text=lambda d: None, registry=reg)

    block = conv.messages[2]["content"][0]
    assert block["is_error"] is True
    assert "Invalid input" in block["content"]


async def test_unknown_tool_becomes_an_error_result():
    reg = _make_registry()
    provider = FakeProvider(
        [
            ([], _tool_use_turn(ToolUse("t1", "nope", {}))),
            (["ok"], FakeFinalMessage([{"type": "text", "text": "ok"}])),
        ]
    )
    conv = Conversation()
    conv.append_user_text("call missing tool")

    await run_turn(conv, provider, on_text=lambda d: None, registry=reg)

    block = conv.messages[2]["content"][0]
    assert block["is_error"] is True
    assert "no tool named" in block["content"]


def test_first_tools_register_with_valid_schemas_and_flags():
    reg = discover_tools()
    names = {s.name for s in reg.specs()}
    assert {
        "add_reminder",
        "list_reminders",
        "delete_reminder",
        "read_file",
        "edit_file",
    } <= names

    # consequential actions are flagged; safe ones are not
    assert reg.get("edit_file").requires_confirmation
    assert reg.get("delete_reminder").requires_confirmation
    assert not reg.get("read_file").requires_confirmation
    assert reg.get("read_file").external_content  # file content = outside world

    for schema in reg.api_schemas():
        assert schema["input_schema"]["type"] == "object"
        assert schema["name"] and schema["description"]


def test_reminders_tool_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_VAR_DIR", str(tmp_path))
    from vision.tools.reminders import (
        AddReminderInput,
        DeleteReminderInput,
        ListRemindersInput,
        add_reminder,
        delete_reminder,
        list_reminders,
    )

    assert "buy milk" in add_reminder(AddReminderInput(text="buy milk"))
    listed = list_reminders(ListRemindersInput())
    assert "buy milk" in listed and "#1" in listed
    assert "Deleted reminder #1" in delete_reminder(DeleteReminderInput(id=1))
    assert "No reminders saved" in list_reminders(ListRemindersInput())
