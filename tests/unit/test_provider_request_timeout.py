"""``provider_request_timeout_seconds`` and the kwargs it is added to."""

from typing import Any

import httpx
import pytest
from any_llm import LLMProvider

from gateway.api.routes._pipeline import _with_provider_timeout
from gateway.core.config import GatewayConfig


def test_timeout_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_PROVIDER_REQUEST_TIMEOUT_SECONDS", "90")
    assert GatewayConfig().provider_request_timeout_seconds == 90.0


def test_a_separate_provider_keyword_is_honored() -> None:
    kwargs = {"provider": LLMProvider.ANTHROPIC, "model": "claude-sonnet-4-5"}
    assert _with_provider_timeout(kwargs, GatewayConfig())["timeout"] == httpx.Timeout(600.0, connect=5.0)


def test_connect_never_outlasts_the_whole_budget() -> None:
    config = GatewayConfig(provider_request_timeout_seconds=2)
    assert _with_provider_timeout({"model": "openai:gpt-4o"}, config)["timeout"] == httpx.Timeout(2.0)


@pytest.mark.parametrize(
    "client_args",
    [
        # The operator's own client timeout stands.
        {"timeout": 60},
        # A pre-built client, which any-llm cannot retime for Bedrock and rejects a timeout for.
        {"region_name": "us-east-1", "client": object()},
    ],
)
def test_a_client_with_its_own_timeout_is_left_alone(client_args: dict[str, Any]) -> None:
    kwargs = {"model": "bedrock:anthropic.claude-sonnet-4-5", "client_args": client_args}
    assert "timeout" not in _with_provider_timeout(kwargs, GatewayConfig())


@pytest.mark.parametrize("kwargs", [{"model": "no-provider"}, {"model": "not-a-provider:x"}, {}])
def test_an_unresolvable_selector_is_left_for_any_llm_to_report(kwargs: dict[str, str]) -> None:
    assert _with_provider_timeout(kwargs, GatewayConfig()) == kwargs
