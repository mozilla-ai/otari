"""Who runs the built-in tools on one provider, for a caller that leaves the choice to the deployment."""

import pytest

from gateway.core.config import GatewayConfig
from gateway.services.tools import provider_tools
from gateway.services.tools._registry import BUILTIN_TOOLS

_SEARCH = "web_search"
_FETCH = "web_fetch"
_CODE = "code_execution"


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OTARI_WEB_SEARCH_INTERCEPT", "OTARI_CODE_EXECUTION_EXECUTOR", "OTARI_SANDBOX_URL"):
        monkeypatch.delenv(name, raising=False)


def test_a_provider_runs_the_tools_it_has_words_for_itself() -> None:
    config = GatewayConfig()
    anthropic = provider_tools(config, "anthropic")
    assert anthropic.native == {_SEARCH, _FETCH, _CODE}
    assert anthropic.claimed == frozenset()
    openai = provider_tools(config, "openai")
    assert openai.native == {_SEARCH, _CODE}
    assert _FETCH not in openai.native


def test_a_provider_without_its_own_tools_runs_none() -> None:
    assert provider_tools(GatewayConfig(), "bedrock").native == frozenset()


def test_otari_runs_what_it_has_a_backend_for_on_a_provider_without_its_own() -> None:
    config = GatewayConfig(sandbox_url="http://sandbox.test")
    assert _CODE in provider_tools(config, "bedrock").claimed
    # A provider that runs code itself keeps it under ``auto``.
    assert _CODE in provider_tools(config, "anthropic").native


def test_the_deployment_executor_claims_code_from_a_provider_that_runs_it() -> None:
    config = GatewayConfig(sandbox_url="http://sandbox.test", code_execution_executor="otari")
    anthropic = provider_tools(config, "anthropic")
    assert _CODE in anthropic.claimed
    assert _CODE not in anthropic.native
    assert _SEARCH in anthropic.native


def test_gemini_runs_every_tool_in_its_own_format() -> None:
    answer = provider_tools(GatewayConfig(), "gemini")

    assert answer.native == frozenset(tool.name for tool in BUILTIN_TOOLS)
    assert answer.claimed == frozenset()


def test_gemini_tools_go_to_otari_where_the_deployment_always_runs_them() -> None:
    config = GatewayConfig(code_execution_executor="otari", sandbox_url="http://sandbox:8080")

    answer = provider_tools(config, "gemini")

    assert _CODE in answer.claimed
    assert _CODE not in answer.native
