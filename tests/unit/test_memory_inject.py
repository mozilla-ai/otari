"""Unit tests for the memory message helpers (inject_memory_facts, build_remember_messages)."""

from __future__ import annotations

from gateway.api.routes._helpers import (
    MEMORY_FACTS_HEADER,
    build_remember_messages,
    inject_memory_facts,
)


def test_inject_no_facts_is_noop() -> None:
    messages = [{"role": "user", "content": "hi"}]
    assert inject_memory_facts(messages, []) is messages


def test_inject_creates_system_message_when_absent() -> None:
    messages = [{"role": "user", "content": "hi"}]
    out = inject_memory_facts(messages, ["likes metric", "name is Dimitris"])
    assert out[0]["role"] == "system"
    assert MEMORY_FACTS_HEADER in out[0]["content"]
    assert "- likes metric" in out[0]["content"]
    assert "- name is Dimitris" in out[0]["content"]
    assert out[1] == {"role": "user", "content": "hi"}
    # input is not mutated
    assert messages == [{"role": "user", "content": "hi"}]


def test_inject_extends_existing_system_message() -> None:
    messages = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}]
    out = inject_memory_facts(messages, ["likes metric"])
    assert out[0]["content"].startswith("You are helpful.")
    assert "- likes metric" in out[0]["content"]
    assert len(out) == 2
    assert messages[0]["content"] == "You are helpful."  # not mutated


def test_build_remember_messages_user_and_assistant() -> None:
    messages = [{"role": "user", "content": "My name is Dimitris"}]
    out = build_remember_messages(messages, "Nice to meet you, Dimitris.")
    assert out == [
        {"role": "user", "content": "My name is Dimitris"},
        {"role": "assistant", "content": "Nice to meet you, Dimitris."},
    ]


def test_build_remember_messages_skips_empty_assistant() -> None:
    messages = [{"role": "user", "content": "hi"}]
    assert build_remember_messages(messages, "") == [{"role": "user", "content": "hi"}]


def test_build_remember_messages_empty_when_nothing() -> None:
    assert build_remember_messages([], "") == []
