"""Unit tests for the ``X-Otari-Protocol-Version`` header (issue #148).

``_parse_resolve_payload`` distinguishes the multi-attempt resolve shape from
the legacy single-attempt shape purely by the presence of an ``attempts`` key,
with no protocol version signal on either side of the wire. These tests pin
that every outgoing resolve/usage call now carries an explicit version header,
so a peer can branch on it instead of forever shape-sniffing.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from gateway.api.routes import _platform as platform_module
from gateway.api.routes._platform import _report_platform_usage, _resolve_platform_credentials


def _config() -> Any:
    cfg = MagicMock()
    cfg.platform = {
        "base_url": "https://platform.local",
        "resolve_timeout_ms": 5000,
        "usage_timeout_ms": 5000,
        "usage_max_retries": 3,
    }
    cfg.platform_token = "gw_test_token"
    return cfg


@pytest.mark.asyncio
async def test_resolve_credentials_sends_protocol_version_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_post(
        *, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        captured["headers"] = headers
        return httpx.Response(
            200,
            json={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "sk-...",
                "managed": False,
                "correlation_id": "corr-1",
            },
        )

    monkeypatch.setattr(platform_module, "_post_platform", fake_post)

    await _resolve_platform_credentials(_config(), user_token="tk_user", model_selector="openai/gpt-4o-mini")

    assert captured["headers"]["X-Otari-Protocol-Version"] == "1"
    assert captured["headers"]["X-Gateway-Token"] == "gw_test_token"
    assert captured["headers"]["X-User-Token"] == "tk_user"


@pytest.mark.asyncio
async def test_usage_report_sends_protocol_version_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_post(
        *, url: str, headers: dict[str, str], body: dict[str, Any], timeout_seconds: float
    ) -> httpx.Response:
        captured["headers"] = headers
        return httpx.Response(204)

    monkeypatch.setattr(platform_module, "_post_platform", fake_post)

    await _report_platform_usage(
        _config(),
        correlation_id="corr-1",
        outcome="error",
        usage=None,
        error_class="http_500",
        is_final_attempt=True,
    )

    assert captured["headers"]["X-Otari-Protocol-Version"] == "1"
    assert captured["headers"]["X-Gateway-Token"] == "gw_test_token"
