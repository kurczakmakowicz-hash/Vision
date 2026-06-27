"""Tier 1 verification: the brain streams, remembers, and degrades gracefully."""

from __future__ import annotations

import pytest

from tests.fakes import FakeFinalMessage, FakeProvider
from vision.config import load_config
from vision.core.agent import run_turn
from vision.core.conversation import Conversation
from vision.core.errors import ProviderUnavailable


async def test_streams_deltas_and_appends_assistant_turn():
    conv = Conversation()
    provider = FakeProvider(
        [(["Hel", "lo ", "there"], FakeFinalMessage([{"type": "text", "text": "Hello there"}]))]
    )
    out: list[str] = []

    conv.append_user_text("hi")
    final = await run_turn(conv, provider, on_text=out.append)

    assert "".join(out) == "Hello there"          # streamed token-by-token
    assert final.stop_reason == "end_turn"
    assert final.text() == "Hello there"
    assert conv.messages[-1]["role"] == "assistant"  # recorded to history


async def test_history_is_replayed_so_it_remembers_earlier_turns():
    conv = Conversation()
    provider = FakeProvider(
        [
            ([], FakeFinalMessage([{"type": "text", "text": "Nice to meet you, Kasia."}])),
            ([], FakeFinalMessage([{"type": "text", "text": "Your name is Kasia."}])),
        ]
    )

    conv.append_user_text("My name is Kasia")
    await run_turn(conv, provider, on_text=lambda d: None)
    conv.append_user_text("What's my name?")
    await run_turn(conv, provider, on_text=lambda d: None)

    # On the second turn the provider was handed the full prior history.
    roles_on_2nd_call = [m["role"] for m in provider.calls[1]["messages"]]
    assert roles_on_2nd_call == ["user", "assistant", "user"]
    assert len(conv.messages) == 4  # user, assistant, user, assistant


async def test_provider_unavailable_is_raised_for_the_ui_to_catch():
    class Boom:
        async def respond(self, conversation, tools):
            raise ProviderUnavailable("network down")

    conv = Conversation()
    conv.append_user_text("hi")
    with pytest.raises(ProviderUnavailable):
        await run_turn(conv, Boom(), on_text=lambda d: None)


def test_config_falls_back_to_defaults_when_file_missing(tmp_path):
    cfg = load_config(tmp_path / "does-not-exist.toml")
    assert cfg.model == "claude-opus-4-8"
    assert cfg.effort == "high"


def test_config_reads_toml_and_ignores_future_sections(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        'model = "claude-opus-4-8"\n'
        'effort = "medium"\n'
        "[voice]\n"               # a section a later tier adds — must not break Tier 1
        'ptt_key = "space"\n'
    )
    cfg = load_config(p)
    assert cfg.effort == "medium"


def test_anthropic_provider_constructs_without_a_key(monkeypatch):
    # Proves the REPL can start (and degrade per-turn) even with no key set.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from vision.seams.provider.anthropic import AnthropicProvider

    provider = AnthropicProvider(model="claude-opus-4-8", effort="high")
    assert provider is not None
