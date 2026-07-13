"""Route-level tests for the MCP-server / guardrail URL SSRF safety check.

The check (`gateway.services.url_safety.validate_mcp_url`) used to run
synchronously inside a Pydantic `model_validator` at request-body-parse time,
which surfaced as a 422. It now runs from the async request pipeline (DNS
resolution must be awaited, and Pydantic validators can't await), so a
rejected URL surfaces as 400 instead. These tests lock in that behaviour and
confirm the check still runs — no live MCP server or guardrails service is
needed since the safety check rejects before any network call to either.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

_UNSAFE_URL = "http://169.254.169.254/latest/meta-data"


def test_unsafe_mcp_server_url_returns_400(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "anthropic:claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "mcp_servers": [{"name": "evil", "url": _UNSAFE_URL}],
        },
        headers=api_key_header,
    )

    assert resp.status_code == 400, resp.text
    assert "link-local" in resp.json()["detail"]


def test_unsafe_guardrail_url_returns_400(
    client: TestClient,
    api_key_header: dict[str, str],
) -> None:
    resp = client.post(
        "/v1/messages",
        json={
            "model": "anthropic:claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "guardrails": [{"profile": "prompt-injection", "url": _UNSAFE_URL}],
        },
        headers=api_key_header,
    )

    assert resp.status_code == 400, resp.text
    assert "link-local" in resp.json()["detail"]


def test_safe_mcp_server_url_no_longer_validated_at_parse_time() -> None:
    """`McpServerConfig` construction itself does no I/O (and cannot raise on
    an unsafe URL) now that the check moved to the async pipeline; this is
    covered at the route level (this file) and the model no longer imports
    `url_safety` at all."""
    from gateway.models.mcp import McpServerConfig

    # Constructing with an unsafe URL does not raise -- the check runs later,
    # from `prepare_gateway_tools`, where it can be awaited.
    cfg = McpServerConfig(name="x", url=_UNSAFE_URL)
    assert cfg.url == _UNSAFE_URL
