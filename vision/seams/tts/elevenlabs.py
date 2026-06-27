"""ElevenLabs implementation of the TTS seam.

Synthesizes one clause at a time (``output_format="pcm_16000"`` to match the
playback path), so speaking starts on the first clause while the rest of the
reply is still being written. The SDK call is synchronous, so it's run in a
thread executor. Lazily imported; needs ``ELEVENLABS_API_KEY`` and a voice id.
"""

from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

_DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"  # ElevenLabs "George" — a sane default
_MODEL = "eleven_flash_v2_5"            # low-latency model, good for conversation


class ElevenLabsTTS:
    def __init__(
        self,
        voice_id: str | None = None,
        api_key: str | None = None,
        model_id: str = _MODEL,
    ) -> None:
        self._voice_id = voice_id or _DEFAULT_VOICE
        self._api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self._model_id = model_id

    async def speak(self, text_chunks: AsyncIterator[str]) -> AsyncIterator[bytes]:
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=self._api_key)
        loop = asyncio.get_running_loop()

        async for chunk in text_chunks:
            text = chunk.strip()
            if not text:
                continue

            def synth(t: str = text) -> bytes:
                stream = client.text_to_speech.convert(
                    voice_id=self._voice_id,
                    model_id=self._model_id,
                    output_format="pcm_16000",
                    text=t,
                )
                return b"".join(stream)

            audio = await loop.run_in_executor(None, synth)
            if audio:
                yield audio
