"""Push-to-talk key listener.

A global hold-to-talk key via pynput, so it works without putting the terminal in
raw mode — the text REPL keeps working alongside it. Key events arrive on a
listener thread; they're marshaled onto the event loop with
``call_soon_threadsafe``. Lazily imports pynput (needs the ``voice`` extra).
"""

from __future__ import annotations

import asyncio
from typing import Callable


def _resolve_key(name: str):
    from pynput import keyboard

    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    return getattr(keyboard.Key, name)  # e.g. "space" → Key.space


class PushToTalk:
    """Calls ``on_down``/``on_up`` (on the event loop) when the PTT key is
    pressed/released."""

    def __init__(
        self,
        key: str,
        on_down: Callable[[], None],
        on_up: Callable[[], None],
    ) -> None:
        self._key_name = key
        self._on_down = on_down
        self._on_up = on_up
        self._listener = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._down = False  # debounce key-repeat

    def start(self) -> None:
        from pynput import keyboard

        self._loop = asyncio.get_running_loop()
        target = _resolve_key(self._key_name)

        def on_press(k):
            if k == target and not self._down:
                self._down = True
                self._loop.call_soon_threadsafe(self._on_down)

        def on_release(k):
            if k == target and self._down:
                self._down = False
                self._loop.call_soon_threadsafe(self._on_up)

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
