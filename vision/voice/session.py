"""VoiceSession — ties ears → the same brain → mouth, with barge-in.

The whole point: a spoken turn is just a typed turn that arrives as transcribed
audio and leaves as spoken audio. It runs the *same* :func:`run_turn` with the
*same* registry, so voice never forks the agent logic. ``handle_utterance``,
``_speak``, and ``barge_in`` are pure orchestration over the seams and are unit
-tested with fakes; ``run`` wires real hardware for live use.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Callable

from vision.core.agent import run_turn
from vision.core.conversation import Conversation
from vision.seams.provider.base import FinalMessage, Provider
from vision.seams.stt.base import STT
from vision.seams.tts.base import TTS
from vision.tools.registry import Registry
from vision.voice.chunker import chunk_text


class VoiceSession:
    def __init__(
        self,
        conversation: Conversation,
        provider: Provider,
        registry: Registry | None,
        stt: STT,
        tts: TTS,
        playback: Any,  # Playback (or a fake with write/clear/is_empty)
        *,
        ptt_key: str = "space",
        input_device: str | int | None = None,
        on_transcript: Callable[[str], None] | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> None:
        self.conversation = conversation
        self.provider = provider
        self.registry = registry
        self.stt = stt
        self.tts = tts
        self.playback = playback
        self.ptt_key = ptt_key
        self.input_device = input_device
        self._on_transcript = on_transcript or (lambda t: None)
        self._on_text = on_text or (lambda d: None)

        self._speak_task: asyncio.Task | None = None
        self._utterance_task: asyncio.Task | None = None
        self._mic = None

    # --- one spoken turn -------------------------------------------------------

    async def handle_utterance(self, frames: AsyncIterator[bytes]) -> FinalMessage | None:
        """Transcribe ``frames``, run the turn, speak the reply. Returns the final
        message, or ``None`` if nothing was heard."""
        text = await self._final_transcript(frames)
        if not text.strip():
            return None

        self._on_transcript(text)  # show what it thought it heard
        self.conversation.append_user_text(text)

        text_q: asyncio.Queue[str | None] = asyncio.Queue()

        def on_text(delta: str) -> None:
            self._on_text(delta)
            text_q.put_nowait(delta)

        self._speak_task = asyncio.create_task(self._speak(text_q))
        try:
            final = await run_turn(
                self.conversation, self.provider, on_text=on_text, registry=self.registry
            )
        finally:
            text_q.put_nowait(None)  # end the speech stream

        try:
            await self._speak_task
        except asyncio.CancelledError:
            pass  # barge-in cancelled speech
        self._speak_task = None
        return final

    async def _final_transcript(self, frames: AsyncIterator[bytes]) -> str:
        parts: list[str] = []
        async for tr in self.stt.transcribe(frames):
            if tr.is_final and tr.text.strip():
                parts.append(tr.text.strip())
        return " ".join(parts)

    async def _speak(self, text_q: asyncio.Queue[str | None]) -> None:
        async def deltas() -> AsyncIterator[str]:
            while True:
                d = await text_q.get()
                if d is None:
                    return
                yield d

        async for audio in self.tts.speak(chunk_text(deltas())):
            self.playback.write(audio)

    def barge_in(self) -> None:
        """Stop talking and listen: drop queued audio and cancel in-flight work."""
        self.playback.clear()
        for task in (self._speak_task, self._utterance_task):
            if task is not None and not task.done():
                task.cancel()

    # --- live hardware loop (not unit-tested; needs a mic, speaker, and keys) ---

    async def run(self) -> None:
        from vision.voice.ptt import PushToTalk

        self.playback.start()
        ptt = PushToTalk(self.ptt_key, on_down=self._on_down, on_up=self._on_up)
        ptt.start()
        print(f"Voice mode: hold [{self.ptt_key}] to talk. Press Ctrl-C to return.\n")
        try:
            await asyncio.Event().wait()  # run until cancelled
        finally:
            ptt.stop()
            self.barge_in()
            self.playback.stop()

    def _on_down(self) -> None:
        from vision.voice.mic import Microphone

        self.barge_in()  # interrupt any reply in progress
        self._mic = Microphone(self.input_device)
        self._mic.start()
        self._utterance_task = asyncio.create_task(
            self.handle_utterance(self._mic.frames())
        )

    def _on_up(self) -> None:
        if self._mic is not None:
            self._mic.stop()  # ends frames() → STT finalizes → reply → speak
            self._mic = None
