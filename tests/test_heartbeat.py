"""Tier 5 verification: the heartbeat surfaces quietly, holds for catch-up,
survives restarts without a boot storm, respects quiet hours, and is dismissible.
All deterministic via an injected clock — no real sleeping."""

from __future__ import annotations

from datetime import datetime

from vision.heartbeat.checks import Check, Notice
from vision.heartbeat.loop import HeartbeatLoop
from vision.heartbeat.quiet import is_quiet
from vision.heartbeat.state import HeartbeatState


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class Handler:
    def __init__(self, notice: Notice | None = None) -> None:
        self.notice = notice
        self.runs = 0

    def __call__(self) -> Notice | None:
        self.runs += 1
        return self.notice


class RecordingSink:
    def __init__(self) -> None:
        self.delivered: list[dict] = []

    def deliver(self, item: dict) -> None:
        self.delivered.append(item)


# --- scheduling ---------------------------------------------------------------


async def test_first_boot_schedules_forward_without_firing(tmp_path):
    state = HeartbeatState(tmp_path / "s.json")
    h = Handler(Notice("x", "interrupt"))
    loop = HeartbeatLoop([Check("c", 10, h)], state, now=Clock(1000))
    await loop.tick()
    assert h.runs == 0                          # never fires everything on boot
    assert state.get_next_due("c") == 1010


async def test_due_check_fires_once_and_holds_notice(tmp_path):
    state = HeartbeatState(tmp_path / "s.json")
    state.set_next_due("c", 1000)
    h = Handler(Notice("ping", "interrupt"))
    loop = HeartbeatLoop([Check("c", 10, h)], state, now=Clock(1000))
    await loop.tick()
    assert h.runs == 1
    assert state.get_next_due("c") == 1010
    held = state.held()
    assert len(held) == 1 and held[0]["summary"] == "ping"


async def test_not_due_is_skipped(tmp_path):
    state = HeartbeatState(tmp_path / "s.json")
    state.set_next_due("c", 2000)
    h = Handler(Notice("x"))
    await HeartbeatLoop([Check("c", 10, h)], state, now=Clock(1000)).tick()
    assert h.runs == 0


async def test_overdue_fires_once_no_boot_storm(tmp_path):
    state = HeartbeatState(tmp_path / "s.json")
    state.set_next_due("c", 1000)  # the laptop was asleep for ages
    h = Handler(Notice("x"))
    await HeartbeatLoop([Check("c", 10, h)], state, now=Clock(10000)).tick()
    assert h.runs == 1                          # once, not once per missed interval
    assert state.get_next_due("c") == 10010


async def test_running_check_is_not_restacked(tmp_path):
    state = HeartbeatState(tmp_path / "s.json")
    state.set_next_due("c", 1000)
    state.set_running("c", True)
    h = Handler(Notice("x"))
    await HeartbeatLoop([Check("c", 10, h)], state, now=Clock(1000)).tick()
    assert h.runs == 0


async def test_restart_resumes_schedule(tmp_path):
    path = tmp_path / "s.json"
    HeartbeatState(path).set_next_due("c", 1500)
    reborn = HeartbeatState(path)               # fresh process tomorrow
    assert reborn.get_next_due("c") == 1500
    h = Handler(Notice("x"))
    await HeartbeatLoop([Check("c", 10, h)], reborn, now=Clock(1000)).tick()
    assert h.runs == 0                           # 1000 < 1500, not due — no refire


# --- surfacing policy ---------------------------------------------------------


async def test_quiet_hours_holds_interrupt_for_later(tmp_path):
    state = HeartbeatState(tmp_path / "s.json")
    state.set_next_due("c", 1000)
    sink = RecordingSink()
    loop = HeartbeatLoop(
        [Check("c", 10, Handler(Notice("ping", "interrupt")))],
        state, sink, is_quiet=lambda _now: True, now=Clock(1000),
    )
    await loop.tick()
    assert sink.delivered == []                  # not pushed during quiet hours
    assert len(state.held()) == 1                # but held
    assert len(state.undelivered()) == 1         # pending for catch-up


async def test_interrupt_delivered_when_not_quiet(tmp_path):
    state = HeartbeatState(tmp_path / "s.json")
    state.set_next_due("c", 1000)
    sink = RecordingSink()
    loop = HeartbeatLoop(
        [Check("c", 10, Handler(Notice("ping", "interrupt")))],
        state, sink, is_quiet=lambda _now: False, now=Clock(1000),
    )
    await loop.tick()
    assert len(sink.delivered) == 1
    assert state.undelivered() == []             # delivered, not pending


async def test_critical_delivered_even_during_quiet_hours(tmp_path):
    state = HeartbeatState(tmp_path / "s.json")
    state.set_next_due("c", 1000)
    sink = RecordingSink()
    loop = HeartbeatLoop(
        [Check("c", 10, Handler(Notice("fire", "critical")))],
        state, sink, is_quiet=lambda _now: True, now=Clock(1000),
    )
    await loop.tick()
    assert len(sink.delivered) == 1


async def test_calm_notice_is_held_not_pushed(tmp_path):
    state = HeartbeatState(tmp_path / "s.json")
    state.set_next_due("c", 1000)
    sink = RecordingSink()
    loop = HeartbeatLoop(
        [Check("c", 10, Handler(Notice("fyi", "calm")))],
        state, sink, is_quiet=lambda _now: False, now=Clock(1000),
    )
    await loop.tick()
    assert sink.delivered == []                  # quiet by default
    assert len(state.held()) == 1


async def test_kill_switch_halts_all_checks(tmp_path):
    state = HeartbeatState(tmp_path / "s.json")
    state.set_next_due("c", 1000)
    h = Handler(Notice("x", "interrupt"))
    loop = HeartbeatLoop([Check("c", 10, h)], state, is_paused=lambda: True, now=Clock(1000))
    await loop.tick()
    assert h.runs == 0


# --- held queue + catch-up ----------------------------------------------------


def test_dismiss_clears_a_notice(tmp_path):
    state = HeartbeatState(tmp_path / "s.json")
    item = state.add_held(check="c", summary="x", level="calm", ts=1)
    assert len(state.held()) == 1
    assert state.dismiss(item["id"]) is True
    assert state.held() == []
    assert state.dismiss("nope") is False


def test_undelivered_then_marked_delivered(tmp_path):
    state = HeartbeatState(tmp_path / "s.json")
    item = state.add_held(check="c", summary="x", level="interrupt", ts=1)
    assert len(state.undelivered()) == 1
    state.mark_delivered(item["id"])
    assert state.undelivered() == []


# --- quiet-hours helper + built-in trigger check ------------------------------


def test_quiet_hours_window_wraps_midnight():
    night = datetime(2026, 6, 27, 23, 30).timestamp()
    day = datetime(2026, 6, 27, 12, 0).timestamp()
    assert is_quiet("22:00", "07:00", night) is True
    assert is_quiet("22:00", "07:00", day) is False
    lunch = datetime(2026, 6, 27, 13, 30).timestamp()
    assert is_quiet("13:00", "14:00", lunch) is True


def test_watch_trigger_fires_once(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_VAR_DIR", str(tmp_path))
    from vision.heartbeat.checks import watch_trigger

    assert watch_trigger() is None                       # nothing by default
    (tmp_path / "trigger").write_text("ping")
    notice = watch_trigger()
    assert notice is not None and notice.summary == "ping" and notice.level == "interrupt"
    assert watch_trigger() is None                       # consumed → fires once
