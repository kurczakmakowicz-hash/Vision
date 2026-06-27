"""Turn a stream of text deltas into clause-sized chunks for TTS.

Because the brain streams and TTS streams, we can start speaking the first clause
while the rest of the reply is still being written. Feeding raw tokens to TTS
sounds choppy, so buffer until a sentence/clause boundary (or a max length) and
flush. Boundaries can land mid-delta, so we split at the *last* boundary in the
running buffer. Pure logic, no audio: fully testable.
"""

from __future__ import annotations

from typing import AsyncIterator

_BOUNDARIES = ".!?;:\n"
_MAX_CHARS = 180  # flush long run-ons even without punctuation


async def chunk_text(
    deltas: AsyncIterator[str],
    *,
    boundaries: str = _BOUNDARIES,
    max_chars: int = _MAX_CHARS,
) -> AsyncIterator[str]:
    """Yield clause-sized chunks as deltas arrive; flush the remainder at the end."""
    buf = ""

    async for delta in deltas:
        if not delta:
            continue
        buf += delta
        cut = max((buf.rfind(b) for b in boundaries), default=-1)
        if cut >= 0:
            head, buf = buf[: cut + 1], buf[cut + 1 :]
            chunk = head.strip()
            if chunk:
                yield chunk
        elif len(buf) >= max_chars:
            chunk = buf.strip()
            if chunk:
                yield chunk
            buf = ""

    tail = buf.strip()
    if tail:
        yield tail
