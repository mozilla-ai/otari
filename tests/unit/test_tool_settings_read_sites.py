"""The tool/guardrail read sites resolve from config (so a dashboard override
hot-applies), not only from the environment. Guards against a normalization that
mutates config but never reaches the request path (eng T3)."""

from typing import Any

import pytest

from gateway.api.routes._helpers import apply_input_guardrails
from gateway.api.routes._tools import (
    _build_web_search_backend,
    _resolve_sandbox_purpose_hint,
    _resolve_web_search_purpose_hint,
)
from gateway.core.config import GatewayConfig
from gateway.models.guardrails import GuardrailConfig


def test_build_web_search_backend_reads_config_knobs() -> None:
    config = GatewayConfig(
        web_search_engines="google,bing",
        web_search_max_results=3,
        web_search_extract=False,
    )
    backend = _build_web_search_backend(base_url="http://searxng:8080", tool_entry={}, config=config)
    assert backend._engines == ("google", "bing")
    assert backend._max_results == 3
    assert backend._extract_content is False


def test_build_web_search_backend_config_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # A config value (a dashboard override applied to config) wins over the env var.
    monkeypatch.setenv("OTARI_WEB_SEARCH_MAX_RESULTS", "9")
    config = GatewayConfig(web_search_max_results=2)
    backend = _build_web_search_backend(base_url="http://x:8080", tool_entry={}, config=config)
    assert backend._max_results == 2


def test_build_web_search_backend_env_fallback_when_config_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # Pure-env deployment: no config value, env still honoured (byte-for-byte prior behavior).
    monkeypatch.setenv("OTARI_WEB_SEARCH_MAX_RESULTS", "4")
    config = GatewayConfig()
    config.web_search_max_results = None
    backend = _build_web_search_backend(base_url="http://x:8080", tool_entry={}, config=config)
    assert backend._max_results == 4


def test_resolve_purpose_hints_from_config() -> None:
    config = GatewayConfig(sandbox_purpose_hint="sbx", web_search_purpose_hint="ws")
    assert _resolve_sandbox_purpose_hint(None, config) == "sbx"
    assert _resolve_web_search_purpose_hint(None, config) == "ws"


@pytest.mark.asyncio
async def test_apply_input_guardrails_uses_config_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import Response

    seen: dict[str, Any] = {}

    async def fake_run(guardrails: Any, input_text: str, *, default_url: str | None) -> Any:
        seen["default_url"] = default_url

        class _V:
            blocked = False
            flagged = False
            results: list[Any] = []

        return _V()

    monkeypatch.setattr("gateway.api.routes._helpers.run_input_guardrails", fake_run)
    config = GatewayConfig(guardrails_url="http://guardrails:8000")
    await apply_input_guardrails(
        [GuardrailConfig(profile="prompt-injection", mode="monitor")],
        "hello",
        response=Response(),
        config=config,
    )
    assert seen["default_url"] == "http://guardrails:8000"
