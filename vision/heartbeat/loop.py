"""The heartbeat loop itself — separate from the conversation loop.

``tick()`` is one wake: for each check, if it's due and not already running,
reschedule it once (so a long sleep never causes a boot storm of catch-up runs)
and run it; route any notice through the quiet/calm surfacing policy. ``run()``
just calls ``tick()`` on an interval until told to stop. The clock and the
quiet/pause predicates are injected, so ``tick()`` is fully testable.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Callable

from vision.heartbeat.checks import Check, Notice
from vision.heartbeat.sink import NotificationSink, NullSink
from vision.heartbeat.state import HeartbeatState


class HeartbeatLoop:
    def __init__(
        self,
        checks: list[Check],
        state: HeartbeatState,
        sink: NotificationSink | None = None,
        *,
        interval_seconds: float = 60,
        is_quiet: Callable[[float], bool] | None = None,
        is_paused: Callable[[], bool] | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.checks = list(checks)
        self.state = state
        self.sink = sink or NullSink()
        self.interval_seconds = interval_seconds
        self._is_quiet = is_quiet
        self._is_paused = is_paused
        self._now = now

    async def tick(self) -> None:
        if self._is_paused is not None and self._is_paused():
            return  # kill switch (Tier 6) halts proactive behavior

        now = self._now()
        for chk in self.checks:
            next_due = self.state.get_next_due(chk.name)

            if next_due is None:
                # First time we've seen this check: schedule forward, don't fire.
                self.state.set_next_due(chk.name, now + chk.interval_seconds)
                continue
            if now < next_due:
                continue
            if self.state.is_running(chk.name):
                continue  # a slow run is still going — skip, don't stack

            # Reschedule ONCE before running (overdue → fire once, not N times).
            self.state.set_next_due(chk.name, now + chk.interval_seconds)
            self.state.set_running(chk.name, True)
            try:
                notice = chk.handler()
                if inspect.isawaitable(notice):
                    notice = await notice
            except Exception as exc:  # noqa: BLE001 — one bad check can't stop the loop
                print(f"[heartbeat] check '{chk.name}' failed: {exc}")
                notice = None
            finally:
                self.state.set_running(chk.name, False)

            if notice is not None:
                self._surface(chk.name, notice, now)

    def _surface(self, name: str, notice: Notice, now: float) -> None:
        item = self.state.add_held(
            check=name, summary=notice.summary, level=notice.level, ts=now
        )
        quiet = bool(self._is_quiet(now)) if self._is_quiet else False
        # Quiet by default: only interrupt/critical push, and interrupts wait out
        # quiet hours. Everything else accumulates in the calm log (held).
        if notice.level == "critical" or (notice.level == "interrupt" and not quiet):
            self.sink.deliver(item)
            self.state.mark_delivered(item["id"])

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await self.tick()
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.interval_seconds)
            except asyncio.TimeoutError:
                pass
