"""Microphone capture → an async stream of PCM frames.

Push-to-talk: the session opens the mic on key-down and closes it on key-up, so
audio is only captured while the user holds the key. That half-duplex gating is
what keeps Vision from transcribing its own speech. The PortAudio callback runs
on its own thread and hands frames to the event loop via ``call_soon_threadsafe``.
Device I/O is lazily imported (needs the ``voice`` extra).
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"
_BLOCKSIZE = 1600  # 100ms at 16kHz

_SENTINEL = b""  # end-of-capture marker


class Microphone:
    def __init__(self, input_device: str | int | None = None) -> None:
        self._input_device = input_device or None
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._stream = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        import sounddevice as sd

        self._loop = asyncio.get_running_loop()

        def callback(indata, _frames, _time, _status):  # PortAudio thread
            data = bytes(indata)
            # Hand off to the loop thread — never touch the loop from here directly.
            self._loop.call_soon_threadsafe(self._queue.put_nowait, data)

        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=_BLOCKSIZE,
            callback=callback,
            device=self._input_device,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, _SENTINEL)

    async def frames(self) -> AsyncIterator[bytes]:
        """Yield captured frames until :meth:`stop` is called."""
        while True:
            data = await self._queue.get()
            if data == _SENTINEL:
                return
            yield data
