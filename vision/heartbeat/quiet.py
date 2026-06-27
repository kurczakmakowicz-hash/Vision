"""Quiet hours — a pure helper so non-urgent surfacing waits for waking hours.

Compares the local wall-clock time of ``now`` against a ``HH:MM``–``HH:MM``
window that may wrap past midnight. Pure and deterministic (tests build ``now``
from a local ``datetime``), so it's tz-independent under test.
"""

from __future__ import annotations

from datetime import datetime, time


def _parse_hhmm(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


def is_quiet(start: str, end: str, now_epoch: float) -> bool:
    now_t = datetime.fromtimestamp(now_epoch).time()
    start_t, end_t = _parse_hhmm(start), _parse_hhmm(end)
    if start_t <= end_t:                 # same-day window, e.g. 13:00–14:00
        return start_t <= now_t <= end_t
    return now_t >= start_t or now_t <= end_t  # wraps midnight, e.g. 22:00–07:00
