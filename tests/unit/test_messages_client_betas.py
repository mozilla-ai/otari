"""Unit tests for which betas the Messages route forwards.

Two never reach a provider: the gateway's own MCP client capability, and every
beta at all when the dispatched model is served with no Anthropic Messages API,
where any-llm refuses the whole request and a model swap would otherwise stop
working. Bedrock gets none either, since it refuses a beta it does not support.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from any_llm import AnyLLM, amessages

from gateway.api.routes.messages import _serves_messages_natively, _split_client_betas, _with_upstream_user
from gateway.services.inference import UPSTREAM_USER_FIELD
from gateway.services.mcp_loop_messages import MCP_CLIENT_BETA

ANTHROPIC = "anthropic:claude-sonnet-4-6"
OPEN_MODEL = "nebius:openai/gpt-oss-120b"
BEDROCK_CLAUDE = "bedrock:anthropic.claude-sonnet-4-5-20250929-v1:0"
BEDROCK_CLAUDE_PROFILE = "bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0"
BEDROCK_LLAMA = "bedrock:meta.llama3-70b-instruct-v1:0"
THINKING_BETA = "interleaved-thinking-2025-05-14"


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


def test_nativeness_follows_the_provider_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _serves_messages_natively(ANTHROPIC)
    assert not _serves_messages_natively(OPEN_MODEL)

    monkeypatch.setattr(AnyLLM.get_provider_class("nebius"), "SUPPORTS_MESSAGES_NATIVE", True)

    assert _serves_messages_natively(OPEN_MODEL)


@pytest.mark.parametrize("model", [ANTHROPIC, BEDROCK_CLAUDE, BEDROCK_CLAUDE_PROFILE])
def test_a_model_with_a_messages_api_of_its_own_is_native(model: str) -> None:
    assert _serves_messages_natively(model)


@pytest.mark.parametrize("model", [OPEN_MODEL, BEDROCK_LLAMA])
def test_a_bridged_model_is_not_native(model: str) -> None:
    assert not _serves_messages_natively(model)


@pytest.mark.parametrize("model", [BEDROCK_CLAUDE, BEDROCK_CLAUDE_PROFILE, BEDROCK_LLAMA])
def test_bedrock_drops_client_betas_for_every_model(model: str) -> None:
    forwarded, saw_mcp = _split_client_betas({"model": model, "betas": [MCP_CLIENT_BETA, THINKING_BETA]})

    assert "betas" not in forwarded
    assert saw_mcp is True


class _Stop(Exception):
    """Ends the call once the boto3 client has seen the request."""


class _BedrockRuntime:
    """A bedrock-runtime client that records the one call it receives."""

    def __init__(self) -> None:
        self.calls: dict[str, dict[str, Any]] = {}

    def invoke_model(self, **kwargs: Any) -> None:
        self.calls["invoke_model"] = kwargs
        raise _Stop

    def converse(self, **kwargs: Any) -> None:
        self.calls["converse"] = kwargs
        raise _Stop


async def _dispatch_to_bedrock(model: str) -> dict[str, dict[str, Any]]:
    """Send a Messages request with a beta and an end-user identifier through any-llm's Bedrock provider."""
    client = _BedrockRuntime()
    kwargs = {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "betas": [THINKING_BETA],
        UPSTREAM_USER_FIELD: "id",
    }
    provider_kwargs, _ = _split_client_betas(_with_upstream_user(kwargs))
    with pytest.raises(_Stop):
        await amessages(**provider_kwargs, client_args={"client": client, "region_name": "us-east-1"})
    return client.calls


@pytest.mark.asyncio
async def test_bedrock_claude_reaches_invoke_model_with_its_identifier_and_no_client_betas() -> None:
    calls = await _dispatch_to_bedrock(BEDROCK_CLAUDE_PROFILE)

    request = calls["invoke_model"]
    body = json.loads(request["body"])
    assert "anthropic_beta" not in body
    assert body["metadata"] == {"user_id": "id"}
    assert "requestMetadata" not in request


@pytest.mark.asyncio
async def test_other_bedrock_models_reach_converse_with_request_metadata_and_no_betas() -> None:
    calls = await _dispatch_to_bedrock(BEDROCK_LLAMA)

    request = calls["converse"]
    assert request["requestMetadata"] == {"otari_user": "id"}
    assert "anthropic_beta" not in json.dumps(request)
