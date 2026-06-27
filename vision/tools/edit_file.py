"""Edit a text file by exact-snippet replacement — Vision's code-editing tool.
It overwrites data, so it's flagged ``requires_confirmation`` (gated in Tier 6)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from vision.tools.registry import tool


class EditFileInput(BaseModel):
    path: str = Field(description="Path to the text file to edit.")
    old: str = Field(
        description="Exact text to replace. Must occur exactly once in the file."
    )
    new: str = Field(description="Replacement text.")


@tool(
    name="edit_file",
    description=(
        "Replace an exact snippet of text in a file with new text. Use this for "
        "code edits. The 'old' text must match exactly once, or the edit is "
        "rejected so you can make it unambiguous."
    ),
    input_model=EditFileInput,
    requires_confirmation=True,  # overwrites data → gated in Tier 6
)
def edit_file(args: EditFileInput) -> str:
    p = Path(args.path).expanduser()
    if not p.exists():
        return f"There's no file at {args.path}."
    if not p.is_file():
        return f"{args.path} is not a file."
    content = p.read_text()
    count = content.count(args.old)
    if count == 0:
        return f"The text to replace wasn't found in {args.path}."
    if count > 1:
        return (
            f"The text to replace appears {count} times in {args.path}; "
            "include more surrounding context so it matches exactly once."
        )
    p.write_text(content.replace(args.old, args.new, 1))
    return f"Edited {args.path}."
