"""``provider_request_timeout_seconds`` and the kwargs it is added to."""

import pytest
from any_llm import LLMProvider

from gateway.api.routes._pipeline import _with_provider_timeout
from gateway.core.config import GatewayConfig


def test_timeout_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTARI_PROVIDER_REQUEST_TIMEOUT_SECONDS", "90")
    assert GatewayConfig().provider_request_timeout_seconds == 90.0


def test_a_separate_provider_keyword_is_honored() -> None:
    kwargs = {"provider": LLMProvider.ANTHROPIC, "model": "claude-sonnet-4-5"}
    assert _with_provider_timeout(kwargs, GatewayConfig())["timeout"] == 600.0


@pytest.mark.parametrize("kwargs", [{"model": "no-provider"}, {"model": "not-a-provider:x"}, {}])
def test_an_unresolvable_selector_is_left_for_any_llm_to_report(kwargs: dict[str, str]) -> None:
    assert _with_provider_timeout(kwargs, GatewayConfig()) == kwargs
