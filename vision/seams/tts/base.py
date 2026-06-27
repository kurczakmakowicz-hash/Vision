"""TTS seam — "give me text, get back audio to play."

Takes an async stream of clause-sized text chunks and yields PCM audio bytes, so
playback can begin before the whole reply is written.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol


class TTS(Protocol):
    def speak(self, text_chunks: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Yield PCM16/16kHz/mono audio bytes for each incoming text chunk."""
        ...
