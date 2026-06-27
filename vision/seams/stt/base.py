"""STT seam — "give me audio, get back text."

One small surface so the transcriber can change in one place. Push-to-talk feeds
PCM frames while a key is held; the adapter streams them to the transcriber and
yields :class:`Transcript` objects (interim and final).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol


@dataclass(frozen=True)
class Transcript:
    text: str
    is_final: bool


class STT(Protocol):
    def transcribe(self, frames: AsyncIterator[bytes]) -> AsyncIterator[Transcript]:
        """Consume PCM16/16kHz/mono frames; yield transcripts as they're recognized."""
        ...
