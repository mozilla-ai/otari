"""Unit tests for the zero-tenant-change dispatch key derivation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from gateway.services.composite_key import derive_automation_key


def _req(**overrides: Any) -> Any:
    base: dict[str, Any] = {"session_label": None, "system": "sys", "tools": [{"name": "a"}, {"name": "b"}]}
    base.update(overrides)
    return SimpleNamespace(**base)


def test_explicit_automation_label_wins() -> None:
    assert derive_automation_key(_req(session_label="automation:9cb6")) == "automation:9cb6"


def test_no_tools_returns_none() -> None:
    # Without tools there is no choreography to compose; not automation-shaped.
    assert derive_automation_key(_req(tools=None)) is None
    assert derive_automation_key(_req(tools=[])) is None


def test_fingerprint_is_fp_prefixed() -> None:
    key = derive_automation_key(_req())
    assert key is not None and key.startswith("fp:")


def test_fingerprint_stable_for_same_prefix() -> None:
    assert derive_automation_key(_req()) == derive_automation_key(_req())


def test_fingerprint_ignores_tool_order() -> None:
    assert derive_automation_key(_req(tools=[{"name": "a"}, {"name": "b"}])) == derive_automation_key(
        _req(tools=[{"name": "b"}, {"name": "a"}])
    )


def test_fingerprint_differs_on_system_or_tools() -> None:
    base = derive_automation_key(_req())
    assert derive_automation_key(_req(system="other")) != base
    assert derive_automation_key(_req(tools=[{"name": "a"}, {"name": "c"}])) != base


def test_non_automation_label_falls_back_to_fingerprint() -> None:
    # A chat label is not an automation key; it fingerprints like any other
    # request (and gets filtered downstream by volume, not here).
    key = derive_automation_key(_req(session_label="chat:123"))
    assert key is not None and key.startswith("fp:")


def test_system_as_content_blocks_is_stable() -> None:
    blocks = [{"type": "text", "text": "hi"}]
    assert derive_automation_key(_req(system=blocks)) == derive_automation_key(_req(system=blocks))
