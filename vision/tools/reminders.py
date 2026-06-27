"""Reminders — a durable local-file to-do store. One of Vision's first
capabilities. Add/list are safe; delete overwrites data, so it's gated."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from vision.tools.registry import tool


def _store_path() -> Path:
    return Path(os.environ.get("VISION_VAR_DIR", "var")) / "reminders.json"


def _load() -> list[dict[str, Any]]:
    p = _store_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save(items: list[dict[str, Any]]) -> None:
    p = _store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items, indent=2))


def _next_id(items: list[dict[str, Any]]) -> int:
    return max((i["id"] for i in items), default=0) + 1


class AddReminderInput(BaseModel):
    text: str = Field(description="What to be reminded about.")
    when: str | None = Field(
        default=None, description="Optional human-readable time, e.g. 'tomorrow 9am'."
    )


@tool(
    name="add_reminder",
    description=(
        "Save a reminder so Vision can recall it later. Use this whenever the "
        "user asks to be reminded of something or wants to note a to-do."
    ),
    input_model=AddReminderInput,
)
def add_reminder(args: AddReminderInput) -> str:
    items = _load()
    item = {
        "id": _next_id(items),
        "text": args.text,
        "when": args.when,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    items.append(item)
    _save(items)
    suffix = f" (when: {args.when})" if args.when else ""
    return f"Saved reminder #{item['id']}: {args.text}{suffix}"


class ListRemindersInput(BaseModel):
    pass


@tool(
    name="list_reminders",
    description=(
        "List the user's saved reminders. Use this when the user asks what's on "
        "their list, what they need to do, or to recall a reminder."
    ),
    input_model=ListRemindersInput,
)
def list_reminders(args: ListRemindersInput) -> str:
    items = _load()
    if not items:
        return "No reminders saved."
    lines = []
    for i in items:
        suffix = f" (when: {i['when']})" if i.get("when") else ""
        lines.append(f"#{i['id']}: {i['text']}{suffix}")
    return "\n".join(lines)


class DeleteReminderInput(BaseModel):
    id: int = Field(description="The id of the reminder to delete.")


@tool(
    name="delete_reminder",
    description="Delete a saved reminder by its id.",
    input_model=DeleteReminderInput,
    requires_confirmation=True,  # deletes data → gated in Tier 6
)
def delete_reminder(args: DeleteReminderInput) -> str:
    items = _load()
    remaining = [i for i in items if i["id"] != args.id]
    if len(remaining) == len(items):
        return f"There's no reminder with id {args.id}."
    _save(remaining)
    return f"Deleted reminder #{args.id}."
