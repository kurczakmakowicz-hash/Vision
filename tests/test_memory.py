"""Tier 4 verification: facts persist across restarts, are user-editable, inject
into the system prompt as data, and the memory tools manage them."""

from __future__ import annotations

from vision.core.conversation import default_system_prompt, system_prompt_with_facts
from vision.memory.store import FactStore


def test_add_load_update_remove_roundtrip(tmp_path):
    store = FactStore(tmp_path / "facts.md")
    fact = store.add("User's name is Kasia")
    assert [f.text for f in store.load()] == ["User's name is Kasia"]

    assert store.update(fact.id, "User's name is Katarzyna")
    assert store.load()[0].text == "User's name is Katarzyna"

    assert store.remove(fact.id)
    assert store.load() == []
    assert store.update("nope", "x") is False
    assert store.remove("nope") is False


def test_facts_survive_a_restart(tmp_path):
    path = tmp_path / "facts.md"
    FactStore(path).add("Prefers metric units")

    # A brand-new store on the same file = a fresh process tomorrow.
    reborn = FactStore(path)
    block = reborn.facts_block()
    assert "Prefers metric units" in block


def test_hand_edited_file_is_respected(tmp_path):
    path = tmp_path / "facts.md"
    path.write_text(
        "# my notes\n\n"
        "- [id:a1] User has a dog named Burek.\n"
        "- [id:a2] Works in Europe/Warsaw timezone.\n"
        "not a fact line — ignored\n"
    )
    facts = FactStore(path).load()
    assert {f.id for f in facts} == {"a1", "a2"}
    assert any("Burek" in f.text for f in facts)


def test_facts_block_framed_as_data_not_instructions(tmp_path):
    store = FactStore(tmp_path / "facts.md")
    assert store.facts_block() == ""  # empty when there are no facts
    store.add("Likes morning meetings")
    block = store.facts_block()
    assert "<user_facts>" in block
    assert "not" in block.lower() and "instructions" in block.lower()
    assert "Likes morning meetings" in block


def test_system_prompt_injection():
    assert system_prompt_with_facts("") == default_system_prompt()
    block = "<user_facts>\n...\n- [id:x] User's name is Kasia.\n</user_facts>"
    prompt = system_prompt_with_facts(block)
    assert "Vision" in prompt          # persona preserved
    assert "Kasia" in prompt           # facts present
    assert prompt.index("Vision") < prompt.index("Kasia")  # facts come after persona


def test_memory_tools_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_MEMORY_FILE", str(tmp_path / "facts.md"))
    from vision.tools.memory_tools import (
        ForgetFactInput,
        RememberFactInput,
        UpdateFactInput,
        forget_fact,
        remember_fact,
        update_fact,
    )

    msg = remember_fact(RememberFactInput(text="Allergic to peanuts"))
    fact_id = msg.split("id ")[1].split(")")[0]
    assert "Allergic to peanuts" in FactStore(tmp_path / "facts.md").facts_block()

    assert "Updated" in update_fact(UpdateFactInput(id=fact_id, text="Allergic to tree nuts"))
    assert "tree nuts" in FactStore(tmp_path / "facts.md").facts_block()

    assert "Removed" in forget_fact(ForgetFactInput(id=fact_id))
    assert FactStore(tmp_path / "facts.md").facts_block() == ""


def test_memory_tools_register_with_forget_gated():
    from vision.tools.registry import discover_tools

    reg = discover_tools()
    assert {"remember_fact", "update_fact", "forget_fact"} <= {s.name for s in reg.specs()}
    assert reg.get("forget_fact").requires_confirmation       # deletes data → gated
    assert not reg.get("remember_fact").requires_confirmation
