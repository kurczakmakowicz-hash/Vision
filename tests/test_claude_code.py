"""Tests for the Claude Code handoff tool — registration, the not-installed and
bad-folder guards, and JSON result parsing (subprocess mocked, so no API calls)."""

from __future__ import annotations

import json

from vision.tools import claude_code as cc
from vision.tools.claude_code import DelegateCodingTaskInput, delegate_coding_task
from vision.tools.registry import discover_tools


class FakeProc:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode

    async def communicate(self):
        return self._stdout, self._stderr

    def kill(self):  # pragma: no cover - only used on timeout
        pass


def _fake_exec(proc: FakeProc):
    async def _exec(*_args, **_kwargs):
        return proc

    return _exec


def test_tool_is_registered_and_gated():
    spec = discover_tools().get("delegate_coding_task")
    assert spec is not None
    assert spec.requires_confirmation  # it edits the user's code → confirmation gate


async def test_reports_when_claude_is_not_installed(monkeypatch, tmp_path):
    monkeypatch.setattr(cc.shutil, "which", lambda _name: None)
    out = await delegate_coding_task(
        DelegateCodingTaskInput(task="anything", project_dir=str(tmp_path))
    )
    assert "isn't installed" in out


async def test_reports_when_project_folder_missing(monkeypatch):
    monkeypatch.setattr(cc.shutil, "which", lambda _name: "/usr/bin/claude")
    out = await delegate_coding_task(
        DelegateCodingTaskInput(task="x", project_dir="/no/such/folder/anywhere")
    )
    assert "no folder" in out


async def test_parses_successful_result_with_cost(monkeypatch, tmp_path):
    monkeypatch.setattr(cc.shutil, "which", lambda _name: "/usr/bin/claude")
    payload = json.dumps(
        {"result": "Added greet() to app.py.", "is_error": False, "total_cost_usd": 0.14}
    ).encode()
    monkeypatch.setattr(cc.asyncio, "create_subprocess_exec", _fake_exec(FakeProc(stdout=payload)))

    out = await delegate_coding_task(
        DelegateCodingTaskInput(task="add greet()", project_dir=str(tmp_path))
    )
    assert "Claude Code finished the task" in out
    assert "Added greet()" in out
    assert "$0.14" in out


async def test_reports_an_error_result(monkeypatch, tmp_path):
    monkeypatch.setattr(cc.shutil, "which", lambda _name: "/usr/bin/claude")
    payload = json.dumps(
        {"result": "Ran out of turns.", "is_error": True, "subtype": "error_max_turns"}
    ).encode()
    monkeypatch.setattr(cc.asyncio, "create_subprocess_exec", _fake_exec(FakeProc(stdout=payload)))

    out = await delegate_coding_task(
        DelegateCodingTaskInput(task="big task", project_dir=str(tmp_path))
    )
    assert "couldn't finish" in out
