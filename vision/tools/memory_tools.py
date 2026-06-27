"""Tools that let Vision manage its own long-term memory as it learns.

Store durable facts (preferences, identities, decisions), not the play-by-play of
one conversation — the short-term history already covers that. Facts are written
to the hand-editable store and take effect in the next session's system prompt.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from vision.memory.store import FactStore
from vision.tools.registry import tool


class RememberFactInput(BaseModel):
    text: str = Field(
        description="A single durable fact about the user, written as a plain statement."
    )


@tool(
    name="remember_fact",
    description=(
        "Save a durable fact about the user — a preference, an identity, or a "
        "decision — so Vision recalls it in future conversations. Use this for "
        "lasting facts, not passing chatter."
    ),
    input_model=RememberFactInput,
)
def remember_fact(args: RememberFactInput) -> str:
    fact = FactStore().add(args.text)
    return f"Saved (id {fact.id}): {fact.text}"


class UpdateFactInput(BaseModel):
    id: str = Field(description="The id of the fact to update.")
    text: str = Field(description="The corrected fact text.")


@tool(
    name="update_fact",
    description="Update a previously saved fact by its id (e.g. to correct it).",
    input_model=UpdateFactInput,
)
def update_fact(args: UpdateFactInput) -> str:
    ok = FactStore().update(args.id, args.text)
    return f"Updated fact {args.id}." if ok else f"There's no fact with id {args.id}."


class ForgetFactInput(BaseModel):
    id: str = Field(description="The id of the fact to remove.")


@tool(
    name="forget_fact",
    description="Remove a saved fact by its id.",
    input_model=ForgetFactInput,
    requires_confirmation=True,  # deletes data → gated in Tier 6
)
def forget_fact(args: ForgetFactInput) -> str:
    ok = FactStore().remove(args.id)
    return f"Removed fact {args.id}." if ok else f"There's no fact with id {args.id}."
