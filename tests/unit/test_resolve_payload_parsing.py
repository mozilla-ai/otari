"""Unit tests for `_parse_resolve_payload`, focused on the `memory_enabled` flag.

The platform is authoritative for per-workspace memory enablement and carries the
signal on the resolve response. The gateway must read it, and must default to
`True` when an older platform omits it so recall never silently stops.
"""

from __future__ import annotations

from typing import Any

import pytest

from gateway.api.routes._platform import _parse_resolve_payload


def _multi_attempt(**extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "request_id": "01HXY",
        "fallback_enabled": False,
        "attempts": [
            {
                "attempt_id": "01HX1",
                "position": 0,
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "sk-test",
                "api_base": None,
                "managed": False,
            }
        ],
    }
    payload.update(extra)
    return payload


def _single_attempt(**extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "provider": "openai",
        "model": "gpt-4o",
        "api_key": "sk-test",
        "api_base": None,
        "managed": False,
        "correlation_id": "01HXC",
    }
    payload.update(extra)
    return payload


@pytest.mark.parametrize("value", [True, False])
def test_multi_attempt_reads_memory_enabled(value: bool) -> None:
    route = _parse_resolve_payload(_multi_attempt(memory_enabled=value))
    assert route.memory_enabled is value


def test_multi_attempt_defaults_memory_enabled_true_when_absent() -> None:
    # Older platform omits the field: default to True so recall still runs and relies
    # on the memory endpoints' own empty-when-disabled behaviour (never silently off).
    route = _parse_resolve_payload(_multi_attempt())
    assert route.memory_enabled is True


@pytest.mark.parametrize("value", [True, False])
def test_single_attempt_reads_memory_enabled(value: bool) -> None:
    route = _parse_resolve_payload(_single_attempt(memory_enabled=value))
    assert route.memory_enabled is value


def test_single_attempt_defaults_memory_enabled_true_when_absent() -> None:
    route = _parse_resolve_payload(_single_attempt())
    assert route.memory_enabled is True
