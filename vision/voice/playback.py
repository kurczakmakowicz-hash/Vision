"""Audio playback as a clearable ring buffer behind a sounddevice callback.

This is *why* playback isn't a blocking ``play()``: barge-in (the user starting a
new turn mid-reply) must be able to stop speech instantly — ``clear()`` drops the
queued audio and the callback immediately plays silence. The buffer logic is pure
and tested; only ``start()``/``stop()`` touch the audio device (lazy import).
"""

from __future__ import annotations

import threading
from collections import deque

SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "int16"
_BYTES_PER_FRAME = 2  # int16 mono


class Playback:
    def __init__(self, output_device: str | int | None = None) -> None:
        self._chunks: deque[bytes] = deque()
        self._lock = threading.Lock()
        self._stream = None
        self._output_device = output_device or None

    def write(self, data: bytes) -> None:
        if data:
            with self._lock:
                self._chunks.append(bytes(data))

    def clear(self) -> None:
        """Drop all queued audio — the barge-in primitive."""
        with self._lock:
            self._chunks.clear()

    def is_empty(self) -> bool:
        with self._lock:
            return not self._chunks

    def _pull(self, nbytes: int) -> bytes:
        """Pull exactly ``nbytes`` from the queue, padding with silence on underrun."""
        out = bytearray()
        with self._lock:
            while self._chunks and len(out) < nbytes:
                chunk = self._chunks[0]
                need = nbytes - len(out)
                if len(chunk) <= need:
                    out += chunk
                    self._chunks.popleft()
                else:
                    out += chunk[:need]
                    self._chunks[0] = chunk[need:]
        if len(out) < nbytes:
            out += b"\x00" * (nbytes - len(out))
        return bytes(out)

    # --- device-backed (lazy: only needs the `voice` extra when actually used) --

    def start(self) -> None:
        import numpy as np
        import sounddevice as sd

        def callback(outdata, frames, _time, _status):  # runs on PortAudio's thread
            data = self._pull(frames * _BYTES_PER_FRAME)
            outdata[:] = np.frombuffer(data, dtype=np.int16).reshape(frames, CHANNELS)

        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=callback,
            device=self._output_device,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.clear()
