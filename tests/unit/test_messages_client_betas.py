"""Unit tests for which betas the Messages route forwards.

Two never reach a provider: the gateway's own MCP client capability, and every
beta at all when the dispatched provider has no Anthropic Messages API, where
any-llm refuses the whole request and a model swap would otherwise stop working.
"""

from __future__ import annotations

from gateway.api.routes.messages import _serves_messages_natively, _split_client_betas
from gateway.services.mcp_loop_messages import MCP_CLIENT_BETA

ANTHROPIC = "anthropic:claude-sonnet-4-6"
OPEN_MODEL = "nebius:openai/gpt-oss-120b"


def test_a_native_provider_keeps_its_betas() -> None:
    kwargs = {"model": ANTHROPIC, "betas": ["code-execution-2025-08-25"]}

    forwarded, saw_mcp = _split_client_betas(kwargs)

    assert forwarded["betas"] == ["code-execution-2025-08-25"]
    assert saw_mcp is False


def test_a_provider_without_a_messages_api_has_its_betas_dropped() -> None:
    kwargs = {"model": OPEN_MODEL, "betas": ["code-execution-2025-08-25", "files-api-2025-04-14"]}

    forwarded, saw_mcp = _split_client_betas(kwargs)

    assert "betas" not in forwarded
    assert saw_mcp is False
    # The caller's own dict is left alone.
    assert kwargs["betas"] == ["code-execution-2025-08-25", "files-api-2025-04-14"]


def test_the_mcp_capability_is_consumed_and_the_rest_forwarded() -> None:
    kwargs = {"model": ANTHROPIC, "betas": [MCP_CLIENT_BETA, "code-execution-2025-08-25"]}

    forwarded, saw_mcp = _split_client_betas(kwargs)

    assert forwarded["betas"] == ["code-execution-2025-08-25"]
    assert saw_mcp is True


def test_the_mcp_capability_alone_leaves_no_betas_key() -> None:
    forwarded, saw_mcp = _split_client_betas({"model": ANTHROPIC, "betas": [MCP_CLIENT_BETA]})

    assert "betas" not in forwarded
    assert saw_mcp is True


def test_the_mcp_capability_is_still_reported_for_an_open_model() -> None:
    forwarded, saw_mcp = _split_client_betas({"model": OPEN_MODEL, "betas": [MCP_CLIENT_BETA]})

    assert "betas" not in forwarded
    assert saw_mcp is True


def test_a_request_with_no_betas_is_passed_through_untouched() -> None:
    kwargs = {"model": OPEN_MODEL}

    forwarded, saw_mcp = _split_client_betas(kwargs)

    assert forwarded is kwargs
    assert saw_mcp is False


def test_an_unknown_selector_is_not_stripped_on_a_guess() -> None:
    assert _serves_messages_natively("not-a-provider:some-model")
    assert _serves_messages_natively(None)
    assert _serves_messages_natively("")
