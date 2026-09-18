"""The verify path asks the bound key format where a key belongs before it looks it up.

A key another deployment minted is answered 421 naming that deployment, and a
key that claims the build's format and fails it is answered 401. Neither reaches
the database: the point of a routing hint and a checksum is a refusal that costs
no round trip (otari-ai#1665). The open-source adapter routes every key
``Local``, so nothing here changes for a standalone gateway.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException, status

from gateway.api.deps import _verify_and_update_api_key, misdirected_key_detail
from gateway.api.routes import _platform as platform_module
from gateway.api.routes._platform import _post_resolve
from gateway.metrics import REGISTRY
from gateway.ports.api_key_format_port import KeyRoute, Local, Malformed, Misdirected

EU_HOST = "api.eu.otari.example"


class RoutingProbe:
    """A key format that routes on a spelled-out prefix, standing in for a hosted one."""

    def mint(self) -> str:
        return "probe-" + "a" * 60

    def fingerprint(self, api_key: str) -> str:
        return api_key[:13]

    def route(self, presented: str) -> KeyRoute:
        if presented.startswith("elsewhere-"):
            return Misdirected(host=EU_HOST)
        if presented.startswith("broken-"):
            return Malformed()
        return Local()


def _db_that_must_not_be_read() -> Any:
    db = AsyncMock()
    db.execute.side_effect = AssertionError("the verify path looked the key up")
    return db


def _sample(labels: dict[str, str]) -> float:
    return REGISTRY.get_sample_value("gateway_auth_failures_total", labels) or 0.0


@pytest.mark.asyncio
async def test_a_misdirected_key_is_421_naming_the_host_with_no_lookup() -> None:
    db = _db_that_must_not_be_read()
    before = _sample({"reason": "misdirected_key"})

    with pytest.raises(HTTPException) as exc_info:
        await _verify_and_update_api_key(db, "elsewhere-" + "b" * 60, RoutingProbe())

    assert exc_info.value.status_code == status.HTTP_421_MISDIRECTED_REQUEST
    assert exc_info.value.detail == misdirected_key_detail(EU_HOST)
    assert EU_HOST in str(exc_info.value.detail)
    db.execute.assert_not_called()
    assert _sample({"reason": "misdirected_key"}) - before == 1.0


@pytest.mark.asyncio
async def test_a_malformed_key_is_401_with_no_lookup() -> None:
    """Gives the ``invalid_format`` auth-failure reason a trigger again."""
    db = _db_that_must_not_be_read()
    before_format = _sample({"reason": "invalid_format"})
    before_invalid = _sample({"reason": "invalid_key"})

    with pytest.raises(HTTPException) as exc_info:
        await _verify_and_update_api_key(db, "broken-" + "c" * 60, RoutingProbe())

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    # The same wording as an unknown key, so the response does not say which
    # check refused it.
    assert exc_info.value.detail == "Invalid API key"
    db.execute.assert_not_called()
    assert _sample({"reason": "invalid_format"}) - before_format == 1.0
    assert _sample({"reason": "invalid_key"}) == before_invalid


@pytest.mark.asyncio
async def test_a_local_key_reaches_the_lookup() -> None:
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    db: Any = AsyncMock()
    db.execute.return_value = result

    with pytest.raises(HTTPException) as exc_info:
        await _verify_and_update_api_key(db, "gw-legacy-" + "d" * 60, RoutingProbe())

    # Looked up and not found: the ordinary refusal, after one round trip.
    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    db.execute.assert_awaited_once()


def _hybrid_config() -> Any:
    return SimpleNamespace(
        platform={"base_url": "https://platform.local", "resolve_timeout_ms": 5000},
        platform_token="gw_test_token",
    )


@pytest.mark.asyncio
async def test_hybrid_mode_forwards_the_platforms_421_and_its_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """A user token for another region is the platform's call; the gateway relays it."""
    detail = misdirected_key_detail(EU_HOST)

    async def fake_post(**_: Any) -> httpx.Response:
        return httpx.Response(status.HTTP_421_MISDIRECTED_REQUEST, json={"detail": detail})

    monkeypatch.setattr(platform_module, "_post_platform", fake_post)

    with pytest.raises(HTTPException) as exc_info:
        await _post_resolve(
            _hybrid_config(),
            user_token="otr_tk_v1_eu_" + "e" * 49,
            path="/gateway/provider-keys/resolve",
            body={"model": "gpt-4o-mini"},
            client_error_detail="Authorization request rejected",
        )

    assert exc_info.value.status_code == status.HTTP_421_MISDIRECTED_REQUEST
    assert exc_info.value.detail == detail
