"""Delegate a real coding task to Claude Code — a specialist sub-agent.

When a job is big enough to deserve its own focus, Vision hands it off to Claude
Code (the coding agent), which works in a project folder and reports back. It can
edit the user's files, so it's gated (the user confirms each handoff). It runs
``claude`` headlessly and inherits Vision's ``ANTHROPIC_API_KEY`` from the
environment, so no separate Claude Code login is needed — only the CLI installed.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from vision.config import load_config
from vision.tools.registry import tool

_INSTALL_HINT = (
    "Claude Code isn't installed on this machine. Install it with "
    "`curl -fsSL https://claude.ai/install.sh | bash`, confirm `claude --version` "
    "works, then ask me again."
)


class DelegateCodingTaskInput(BaseModel):
    task: str = Field(
        description=(
            "The coding task to delegate, in plain language — e.g. 'fix the failing "
            "login test', 'add a dark-mode toggle to the settings page'."
        )
    )
    project_dir: str = Field(
        description="Absolute path to the project folder Claude Code should work in."
    )


@tool(
    name="delegate_coding_task",
    description=(
        "Hand a real coding task to Claude Code (a coding agent) to work on a "
        "project folder — fixing bugs, adding features, refactoring, writing tests. "
        "It can read and EDIT files in that folder. Use this when the user wants "
        "actual code changes made, not just advice. Always include which project "
        "folder it should work in."
    ),
    input_model=DelegateCodingTaskInput,
    requires_confirmation=True,  # it edits the user's code → gated
)
async def delegate_coding_task(args: DelegateCodingTaskInput) -> str:
    if shutil.which("claude") is None:
        return _INSTALL_HINT

    project = Path(args.project_dir).expanduser()
    if not project.is_dir():
        return f"There's no folder at {args.project_dir}. Tell me the project's full path."

    cc = load_config().claude_code
    cmd = [
        "claude",
        "-p",
        args.task,
        "--output-format",
        "json",
        "--permission-mode",
        cc.permission_mode,
        "--max-turns",
        str(cc.max_turns),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(project),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=cc.timeout_seconds)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        minutes = max(1, cc.timeout_seconds // 60)
        return f"Claude Code ran past the {minutes}-minute limit and was stopped."
    except Exception as exc:  # noqa: BLE001
        return f"Couldn't start Claude Code: {exc}"

    stdout = out.decode(errors="replace").strip()
    stderr = err.decode(errors="replace").strip()

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        if proc.returncode != 0:
            return f"Claude Code errored (exit {proc.returncode}): {(stderr or stdout)[-1500:]}"
        return f"Claude Code finished:\n{stdout[-3000:]}" if stdout else "Claude Code finished with no output."

    result = (data.get("result") or "").strip()
    cost = data.get("total_cost_usd")
    cost_note = f" (cost ~${cost:.2f})" if isinstance(cost, (int, float)) else ""

    if data.get("is_error"):
        detail = result or data.get("subtype") or "unknown error"
        return f"Claude Code couldn't finish the task{cost_note}: {detail[-1500:]}"
    return f"Claude Code finished the task{cost_note}. It reported:\n\n{result[-3000:]}"
