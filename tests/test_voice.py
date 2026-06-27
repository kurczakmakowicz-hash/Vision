"""Tier 3 verification (the parts that don't need a microphone): the chunker, the
clearable playback buffer, and the voice session reusing the SAME brain + tools.

Live audio (mic, Deepgram, ElevenLabs, speaker, keys) can't run headless; those
adapters are import-lazy and the seams keep them swappable.
"""

from __future__ import annotations

import asyncio

from tests.fakes import (
    FakeFinalMessage,
    FakePlayback,
    FakeProvider,
    FakeSTT,
    FakeTTS,
    empty_frames,
    make_echo_registry,
    tool_use_turn,
)
from vision.core.conversation import Conversation
from vision.seams.provider.base import ToolUse
from vision.voice.chunker import chunk_text
from vision.voice.playback import Playback
from vision.voice.session import VoiceSession


# --- chunker ------------------------------------------------------------------


async def test_chunker_splits_on_boundaries_including_mid_delta():
    async def src():
        for d in ["Hello", " there", ". How ", "are you", "?", " bye"]:
            yield d

    chunks = [c async for c in chunk_text(src())]
    assert chunks[0] == "Hello there."        # boundary arrived mid-delta
    assert "How are you?" in chunks
    assert chunks[-1] == "bye"                 # tail flushed without a boundary


async def test_chunker_flushes_long_runons_without_punctuation():
    async def src():
        yield "word " * 60  # > max_chars, no boundary

    chunks = [c async for c in chunk_text(src())]
    assert chunks  # something was emitted rather than buffering forever


# --- playback buffer (barge-in primitive) ------------------------------------


def test_playback_buffer_pull_and_underrun():
    pb = Playback()
    pb.write(b"abcd")
    pb.write(b"ef")
    assert not pb.is_empty()
    assert pb._pull(3) == b"abc"
    assert pb._pull(3) == b"def"
    assert pb._pull(2) == b"\x00\x00"  # underrun → silence
    assert pb.is_empty()


def test_playback_clear_drops_queued_audio():
    pb = Playback()
    pb.write(b"xyzw")
    pb.clear()
    assert pb.is_empty()
    assert pb._pull(2) == b"\x00\x00"


# --- voice session reuses the same brain -------------------------------------


async def test_voice_turn_transcribes_runs_and_speaks():
    stt = FakeSTT(["what's on my list"])
    provider = FakeProvider(
        [(["You have ", "two items."], FakeFinalMessage([{"type": "text", "text": "You have two items."}]))]
    )
    tts = FakeTTS()
    pb = FakePlayback()
    conv = Conversation(origin="voice")
    seen: list[str] = []

    session = VoiceSession(
        conv, provider, registry=None, stt=stt, tts=tts, playback=pb, on_transcript=seen.append
    )
    final = await session.handle_utterance(empty_frames())

    assert seen == ["what's on my list"]                       # transcript shown
    assert final is not None and final.stop_reason == "end_turn"
    assert conv.messages[0] == {"role": "user", "content": "what's on my list"}
    assert "two items" in pb.buf.decode("utf-8")               # reply was spoken


async def test_voice_turn_runs_tools_through_the_same_loop():
    stt = FakeSTT(["remember to call mom"])
    reg = make_echo_registry()
    provider = FakeProvider(
        [
            ([], tool_use_turn(ToolUse("t1", "echo", {"text": "call mom"}))),
            (["Saved."], FakeFinalMessage([{"type": "text", "text": "Saved."}])),
        ]
    )
    tts = FakeTTS()
    pb = FakePlayback()
    conv = Conversation(origin="voice")

    session = VoiceSession(conv, provider, registry=reg, stt=stt, tts=tts, playback=pb)
    await session.handle_utterance(empty_frames())

    # the tool ran via the same run_turn the text path uses
    assert any(
        isinstance(m["content"], list) and m["content"][0].get("type") == "tool_result"
        for m in conv.messages
    )
    assert "Saved" in pb.buf.decode("utf-8")


async def test_silent_utterance_does_nothing():
    session = VoiceSession(
        Conversation(), FakeProvider([]), registry=None,
        stt=FakeSTT([""]), tts=FakeTTS(), playback=FakePlayback(),
    )
    assert await session.handle_utterance(empty_frames()) is None


async def test_barge_in_clears_playback_and_cancels_speech():
    pb = FakePlayback()
    pb.write(b"\x00\x00")
    session = VoiceSession(
        Conversation(), FakeProvider([]), registry=None,
        stt=FakeSTT([]), tts=FakeTTS(), playback=pb,
    )

    async def _forever():
        await asyncio.sleep(100)

    session._speak_task = asyncio.create_task(_forever())
    session.barge_in()

    assert pb.is_empty()
    assert pb.cleared >= 1
    await asyncio.sleep(0)  # let the cancellation land
    assert session._speak_task.cancelled() or session._speak_task.done()
