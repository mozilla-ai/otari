"""Unit tests for ``_parse_resolve_payload``'s handling of the optional
identity fields a peer sends alongside the routing plan: the caller identity
(``user_id``) and the tenancy pair (``workspace_id``, ``organization_id``).

Covers the shapes a peer can send: the modern attempts-list shape with the
fields present, the same shape with them absent (an older or non-otari.ai
peer), and the legacy single-attempt shape, which carries no ``user_id`` but
does mirror the tenancy pair. A last test pins the tenancy values as
recording-only: they must not reach the provider call.
"""

from __future__ import annotations

from typing import Any

from gateway.api.routes._platform import ResolvedAttempt, _parse_resolve_payload, default_attempt_kwargs


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


def test_parse_resolve_payload_reads_tenant_ids_from_attempts_shape() -> None:
    """The attempts-list shape carries the tenancy pair at the top level, next to request_id."""
    payload = {
        "request_id": "req-3",
        "fallback_enabled": False,
        "workspace_id": "ws-42",
        "organization_id": "org-7",
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

    assert route.workspace_id == "ws-42"
    assert route.organization_id == "org-7"


def test_parse_resolve_payload_reads_tenant_ids_from_legacy_shape() -> None:
    """Unlike user_id, the tenancy pair is read off the legacy single-attempt shape too.

    A peer old enough to still answer flat may nonetheless know its own tenant,
    and a record that ships untenanted is a record the miner throws away.
    """
    payload = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": "sk-...",
        "managed": False,
        "correlation_id": "corr-2",
        "workspace_id": "ws-42",
        "organization_id": "org-7",
    }

    route = _parse_resolve_payload(payload)

    assert route.workspace_id == "ws-42"
    assert route.organization_id == "org-7"


def test_parse_resolve_payload_tolerates_missing_tenant_ids() -> None:
    """Both shapes must resolve without the pair: no error, both fields None.

    This is the fail-open rule the observation stream runs under. An older peer
    that omits them costs the record, never the request.
    """
    attempts_payload = {
        "request_id": "req-4",
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
    legacy_payload = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": "sk-...",
        "managed": False,
        "correlation_id": "corr-3",
    }

    for payload in (attempts_payload, legacy_payload):
        route = _parse_resolve_payload(payload)
        assert route.workspace_id is None
        assert route.organization_id is None


def test_tenant_ids_do_not_reach_the_provider_call() -> None:
    """The tenancy pair is recording-only: it cannot alter an upstream request.

    That holds structurally, because the pair lives on the route and only a
    ResolvedAttempt feeds the provider kwargs. This pins it, so moving either
    field onto the attempt (where extra_params is forwarded to the provider
    client verbatim) fails here instead of silently leaking a tenant id
    upstream.
    """
    payload = {
        "request_id": "req-5",
        "fallback_enabled": False,
        "workspace_id": "ws-42",
        "organization_id": "org-7",
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
    assert "workspace_id" not in ResolvedAttempt.model_fields
    assert "organization_id" not in ResolvedAttempt.model_fields

    kwargs = default_attempt_kwargs(route.attempts[0], {"messages": [{"role": "user", "content": "hi"}]})
    flattened: dict[str, Any] = {**kwargs, **(kwargs.get("client_args") or {})}
    assert "workspace_id" not in flattened
    assert "organization_id" not in flattened
    assert "ws-42" not in str(kwargs)
    assert "org-7" not in str(kwargs)
