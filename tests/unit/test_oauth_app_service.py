"""Time is sampled once for each app grant exchange."""

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from gateway.core.settings.identity import OAuthAppConfig
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.identity_exceptions import OAuthAppGrantError
from gateway.models.identity import OAuthAppGrant
from gateway.models.tenancy import User
from gateway.ports.api_key_format_port import ApiKeyFormatPort
from gateway.repositories.identity.oauth_app_grant_repository import OAuthAppGrantRepository
from gateway.schemas.identity.oauth_apps import AppTokenRequest
from gateway.services.api_keys import ApiKeyService
from gateway.services.identity import OAuthAppService
from gateway.services.tenancy import OrganizationService, WorkspaceService


@pytest.mark.asyncio
async def test_an_unregistered_client_id_is_refused_without_reaching_the_log(caplog: pytest.LogCaptureFixture) -> None:
    grants = AsyncMock(spec=OAuthAppGrantRepository)
    service = OAuthAppService(
        cast(UnitOfWork, MagicMock(spec=UnitOfWork)),
        grants,
        AsyncMock(spec=ApiKeyService),
        cast(ApiKeyFormatPort, Mock()),
        AsyncMock(spec=OrganizationService),
        cast(WorkspaceService, Mock()),
        apps={"app": OAuthAppConfig(name="App", redirect_uris=["https://app.example/callback"])},
        ui_link=lambda path: path,
        issuer="https://otari.example",
    )
    forged = "x\nWARNING  Signed in admin [red]forged[/red]"
    request = AppTokenRequest(
        grant_type="authorization_code",
        client_id=forged,
        redirect_uri="https://app.example/callback",
        code="c" * 43,
        code_verifier="v" * 43,
    )
    with caplog.at_level(logging.WARNING, logger="gateway"), pytest.raises(OAuthAppGrantError) as refused:
        await service.exchange(request)
    assert refused.value.error == "invalid_client"
    assert "reason=unknown_client" in caplog.text
    assert "forged" not in caplog.text
    grants.find_bound.assert_not_called()


@pytest.mark.asyncio
async def test_redemption_uses_the_same_instant_for_lookup_and_atomic_consumption() -> None:
    now = datetime(2026, 11, 10, tzinfo=UTC)
    clock = Mock(side_effect=[now, now + timedelta(minutes=3)])
    grants = AsyncMock(spec=OAuthAppGrantRepository)
    grants.find_bound.return_value = OAuthAppGrant(
        code_hash="hash", user_id=uuid.uuid4(), workspace_id=uuid.uuid4(), expires_at=now + timedelta(seconds=1)
    )
    grants.consume.return_value = True
    keys = AsyncMock(spec=ApiKeyService)
    keys.issue_app_key.return_value = "issued-token"
    organizations = AsyncMock(spec=OrganizationService)
    organizations.hold_delegated_workspace_access.return_value = User(id=uuid.uuid4())
    service = OAuthAppService(
        cast(UnitOfWork, MagicMock(spec=UnitOfWork)),
        grants,
        keys,
        cast(ApiKeyFormatPort, Mock()),
        organizations,
        cast(WorkspaceService, Mock()),
        apps={"app": OAuthAppConfig(name="App", redirect_uris=["https://app.example/callback"])},
        ui_link=lambda path: path,
        issuer="https://otari.example",
        clock=clock,
    )
    request = AppTokenRequest(
        grant_type="authorization_code",
        client_id="app",
        redirect_uri="https://app.example/callback",
        code="c" * 43,
        code_verifier="v" * 43,
    )
    assert await service.exchange(request) == "issued-token"
    assert clock.call_count == 1
    assert grants.find_bound.call_args.kwargs["now"] == now
    grants.consume.assert_awaited_once_with("hash", now)
