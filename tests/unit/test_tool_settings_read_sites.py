"""The tool/guardrail read sites resolve from config (so a dashboard override
hot-applies), not only from the environment. Guards against a normalization that
mutates config but never reaches the request path (eng T3)."""

from typing import Any

import pytest

from gateway.api.routes._guardrail_targets import chat_targets
from gateway.api.routes._guardrails import RequestGuardrails
from gateway.api.routes._tools import _resolve_sandbox_purpose_hint, _resolve_web_search_purpose_hint
from gateway.core.config import GatewayConfig
from gateway.core.settings.tools import enrichment_enabled, synthesized_search_instance
from gateway.models.guardrails import GuardrailConfig
from gateway.services.tools import (
    WebToolCallCap,
    build_web_search_backends,
    resolve_in_loop_search,
    web_search_max_results_baseline,
)
from gateway.services.web_retrieval_policy import DomainPolicy


def test_web_search_engines_come_from_config() -> None:
    config = GatewayConfig(web_search_engines="google,bing", web_search_url="http://searxng:8080")

    synthesized = synthesized_search_instance(config)

    assert synthesized is not None
    assert synthesized.engines == ("google", "bing")


def test_web_search_max_results_reads_config_first(monkeypatch: pytest.MonkeyPatch) -> None:
    # A config value (a dashboard override applied to config) wins over the env var.
    monkeypatch.setenv("OTARI_WEB_SEARCH_MAX_RESULTS", "9")
    assert web_search_max_results_baseline(GatewayConfig(web_search_max_results=2)) == 2


def test_web_search_max_results_falls_back_to_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pure-env deployment: no config value, env still honored.
    monkeypatch.setenv("OTARI_WEB_SEARCH_MAX_RESULTS", "4")
    config = GatewayConfig()
    config.web_search_max_results = None
    assert web_search_max_results_baseline(config) == 4


def test_web_search_extract_reads_config_first() -> None:
    assert enrichment_enabled(GatewayConfig(web_search_extract=False)) is False


@pytest.mark.parametrize("trusted", [False, True])
def test_retrieval_proxy_trust_comes_only_from_deployment_config(
    monkeypatch: pytest.MonkeyPatch, trusted: bool
) -> None:
    monkeypatch.setenv("OTARI_WEB_RETRIEVAL_TRUST_ENV_PROXY", str(not trusted).lower())
    config = GatewayConfig(web_retrieval_trust_env_proxy=trusted, web_search_url="http://searxng:8080")
    search = resolve_in_loop_search(config, None)
    assert search is not None

    (backend,) = build_web_search_backends(
        config,
        search,
        search_tool_entry={"trust_env_proxy": not trusted, "web_retrieval_trust_env_proxy": not trusted},
        fetch=None,
        fetch_policy=DomainPolicy(),
        cap=WebToolCallCap(10),
        purpose_hint=None,
        web_provider_client=None,
        tally=None,
        trace=None,
    )

    assert backend._trust_env_proxy is trusted


def test_resolve_purpose_hints_from_config() -> None:
    config = GatewayConfig(sandbox_purpose_hint="sbx", web_search_purpose_hint="ws")
    assert _resolve_sandbox_purpose_hint(None, config) == "sbx"
    assert _resolve_web_search_purpose_hint(None, config) == "ws"


def test_sandbox_image_reads_config_first() -> None:
    config = GatewayConfig(sandbox_session_image="ghcr.io/acme/sandbox:2")
    assert config.effective_sandbox_image() == "ghcr.io/acme/sandbox:2"


def test_sandbox_image_falls_back_to_env_when_the_override_is_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clearing a dashboard override must land on the configured value, not on nothing."""
    monkeypatch.setenv("OTARI_SANDBOX_SESSION_IMAGE", "mzdotai/otari-sandbox-container:latest")
    config = GatewayConfig()
    config.sandbox_session_image = None
    assert config.effective_sandbox_image() == "mzdotai/otari-sandbox-container:latest"
    # And the cleared value is still what a workspace may pin.
    assert config.pinnable_sandbox_images() == ("mzdotai/otari-sandbox-container:latest",)


@pytest.mark.asyncio
async def test_request_guardrails_use_config_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import Response

    seen: dict[str, Any] = {}

    async def fake_run(
        guardrails: Any,
        input_text: str,
        *,
        default_url: str | None,
        credentials: Any = None,
        mandated: Any = None,
        in_process: Any = None,
        unsafe: Any = None,
    ) -> Any:
        seen["default_url"] = default_url

        class _V:
            blocked = False
            flagged = False
            results: list[Any] = []

        return _V()

    monkeypatch.setattr("gateway.api.routes._guardrails.run_input_guardrails", fake_run)
    config = GatewayConfig(guardrails_url="http://guardrails:8000")
    await RequestGuardrails(
        [GuardrailConfig(profile="prompt-injection", mode="monitor")], response=Response(), config=config
    ).admit(chat_targets([{"role": "user", "content": "hello"}]))
    assert seen["default_url"] == "http://guardrails:8000"
