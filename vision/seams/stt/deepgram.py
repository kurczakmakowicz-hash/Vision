"""Deepgram implementation of the STT seam (streaming websocket).

Targets ``deepgram-sdk`` v3 async live transcription. The SDK is event-callback
based, so we bridge its callbacks to our async-iterator contract via a queue.
Lazily imported — the package imports fine without the ``voice`` extra. Needs
``DEEPGRAM_API_KEY``. Live behavior requires a key + audio, so it isn't unit
-tested here; the seam keeps it swappable.
"""

from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

from vision.seams.stt.base import Transcript

_SAMPLE_RATE = 16_000


class DeepgramSTT:
    def __init__(
        self,
        api_key: str | None = None,
        language: str = "en-US",
        model: str = "nova-2",
    ) -> None:
        self._api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")
        self._language = language
        self._model = model

    async def transcribe(self, frames: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents

        dg = DeepgramClient(self._api_key)
        connection = dg.listen.asyncwebsocket.v("1")
        queue: asyncio.Queue[Transcript | None] = asyncio.Queue()

        async def on_transcript(_client, result, **_kw):
            alt = result.channel.alternatives[0]
            if alt.transcript:
                await queue.put(Transcript(text=alt.transcript, is_final=result.is_final))

        async def on_close(_client, *_a, **_k):
            await queue.put(None)

        connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
        connection.on(LiveTranscriptionEvents.Close, on_close)

        await connection.start(
            LiveOptions(
                model=self._model,
                language=self._language,
                encoding="linear16",
                sample_rate=_SAMPLE_RATE,
                channels=1,
            )
        )

        async def pump() -> None:
            async for frame in frames:
                await connection.send(frame)
            await connection.finish()

        pump_task = asyncio.create_task(pump())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            pump_task.cancel()
