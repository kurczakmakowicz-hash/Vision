"""Read a text file from disk — Vision's first coding-help tool. Read-only and
safe, so it runs without confirmation. Its output is outside-world content, so it
is flagged ``external_content`` for the Tier 6 injection scan."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from vision.tools.registry import tool

_MAX_BYTES = 20_000


class ReadFileInput(BaseModel):
    path: str = Field(description="Path to the text file to read.")
    max_bytes: int = Field(
        default=_MAX_BYTES,
        description="Maximum number of bytes to read (truncates large files).",
    )


@tool(
    name="read_file",
    description=(
        "Read a text file from disk and return its contents. Use this for coding "
        "help — to look at a file the user mentions. Read-only and safe."
    ),
    input_model=ReadFileInput,
    external_content=True,
)
def read_file(args: ReadFileInput) -> str:
    p = Path(args.path).expanduser()
    if not p.exists():
        return f"There's no file at {args.path}."
    if not p.is_file():
        return f"{args.path} is not a file."
    limit = max(1, min(args.max_bytes, _MAX_BYTES))
    data = p.read_text(errors="replace")
    truncated = len(data) > limit
    body = data[:limit]
    note = "\n…(truncated)" if truncated else ""
    return f"--- {args.path} ---\n{body}{note}"
