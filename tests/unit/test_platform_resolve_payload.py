"""Unit tests for ``_parse_resolve_payload``'s handling of the platform's
optional caller-identity field (``user_id``).

Covers the three shapes a peer can send: the modern attempts-list shape with
the field present, the same shape with it absent (an older or non-otari.ai
peer), and the legacy single-attempt shape, which never carries it.
"""

from __future__ import annotations

from gateway.api.routes._platform import _parse_resolve_payload


def test_parse_resolve_payload_reads_user_id_from_attempts_shape() -> None:
    """The modern attempts-list shape carries user_id at the top level, alongside request_id."""
    payload = {
        "request_id": "req-1",
        "fallback_enabled": False,
        "user_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "attempts": [
            {
                "attempt_id": "a0",
                "position": 0,
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "sk-...",
                "managed": False,
            }
        ],
    }

    route = _parse_resolve_payload(payload)

    assert route.user_id == "3fa85f64-5717-4562-b3fc-2c963f66afa6"


def test_parse_resolve_payload_tolerates_missing_user_id() -> None:
    """A peer that predates the field, or simply omits it, must not break resolution."""
    payload = {
        "request_id": "req-2",
        "fallback_enabled": False,
        "attempts": [
            {
                "attempt_id": "a0",
                "position": 0,
                "provider": "openai",
                "model": "gpt-4o",
                "api_key": "sk-...",
                "managed": False,
            }
        ],
    }

    route = _parse_resolve_payload(payload)

    assert route.user_id is None


def test_parse_resolve_payload_legacy_shape_has_no_user_id() -> None:
    """The legacy single-attempt shape predates user_id entirely; no legacy mirror exists."""
    payload = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": "sk-...",
        "managed": False,
        "correlation_id": "corr-1",
        # A hypothetical peer stapling user_id onto the legacy shape is still
        # ignored: only the attempts-list branch reads it.
        "user_id": "ignored-on-legacy-shape",
    }

    route = _parse_resolve_payload(payload)

    assert route.user_id is None
