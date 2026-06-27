"""Deepgram implementation of the STT seam — prerecorded REST endpoint.

For push-to-talk we capture the whole clip while the key is held, then send it
once to Deepgram and get the transcript back. That's simpler and more robust than
a streaming websocket for PTT, and — by using plain HTTP via httpx (already a
dependency) instead of the deepgram SDK — it's immune to the SDK's frequent major
rewrites. Needs ``DEEPGRAM_API_KEY``. Lazily imports nothing vendor-specific.
"""

from __future__ import annotations

import os
from typing import AsyncIterator

import httpx

from vision.seams.stt.base import Transcript

_ENDPOINT = "https://api.deepgram.com/v1/listen"
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
        # PTT: drain the whole held-key clip, then transcribe it in one request.
        audio = bytearray()
        async for frame in frames:
            audio += frame
        if not audio:
            return

        params = {
            "model": self._model,
            "language": self._language,
            "encoding": "linear16",      # raw PCM16 from the mic
            "sample_rate": str(_SAMPLE_RATE),
            "channels": "1",
            "punctuate": "true",
            "smart_format": "true",
        }
        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "application/octet-stream",
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                _ENDPOINT, params=params, headers=headers, content=bytes(audio)
            )
            resp.raise_for_status()
            data = resp.json()

        try:
            transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
        except (KeyError, IndexError, TypeError):
            transcript = ""
        if transcript.strip():
            yield Transcript(text=transcript, is_final=True)
