"""How the app grant exchange admits clients and samples time."""

import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
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
from gateway.schemas.identity import AppAuthorizationRequest
from gateway.schemas.identity.oauth_apps import AppTokenRequest
from gateway.services.api_keys import ApiKeyService
from gateway.services.budgets import BudgetService
from gateway.services.identity import OAuthAppService
from gateway.services.tenancy import OrganizationService, WorkspaceService
from gateway.types.key_scope import DEFAULT_KEY_SCOPES, KeyScopeSet


def _service(
    grants: OAuthAppGrantRepository,
    *,
    open_apps: bool = False,
    spend_limits: bool = True,
    ui_link: Callable[[str], str] = lambda path: path,
) -> OAuthAppService:
    return OAuthAppService(
        cast(UnitOfWork, MagicMock(spec=UnitOfWork)),
        grants,
        AsyncMock(spec=ApiKeyService),
        cast(ApiKeyFormatPort, Mock()),
        AsyncMock(spec=OrganizationService),
        cast(WorkspaceService, Mock()),
        AsyncMock(spec=BudgetService),
        apps={"app": OAuthAppConfig(name="App", redirect_uris=["https://app.example/callback"])},
        open_apps=open_apps,
        spend_limits=spend_limits,
        ui_link=ui_link,
        issuer="https://otari.example",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("open_apps", [False, True])
async def test_an_unregistered_client_id_is_refused_without_reaching_the_log(
    caplog: pytest.LogCaptureFixture, open_apps: bool
) -> None:
    grants = AsyncMock(spec=OAuthAppGrantRepository)
    service = _service(grants, open_apps=open_apps)
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


def test_open_apps_admit_a_site_by_its_origin_and_mark_it_unverified() -> None:
    service = _service(AsyncMock(spec=OAuthAppGrantRepository), open_apps=True)
    details = service.details("https://someone.github.io", "https://someone.github.io/pizza/")
    assert details.name == "someone.github.io"
    assert not details.verified
    assert details.scopes == list(DEFAULT_KEY_SCOPES)
    assert service.details("app", "https://app.example/callback").verified


def test_an_open_app_may_be_a_site_with_a_name_longer_than_a_registration_allows() -> None:
    site = f"{'a' * 50}.{'b' * 50}.example"
    service = _service(AsyncMock(spec=OAuthAppGrantRepository), open_apps=True)
    assert service.details(f"https://{site}", f"https://{site}/").name == site


def test_the_manage_url_is_absolute_when_the_dashboard_has_no_address_of_its_own() -> None:
    grants = AsyncMock(spec=OAuthAppGrantRepository)
    assert _service(grants).manage_url() == "https://otari.example/keys"
    configured = _service(grants, ui_link=lambda path: "https://app.otari.example" + path)
    assert configured.manage_url() == "https://app.otari.example/keys"


@pytest.mark.asyncio
async def test_no_spending_limit_is_offered_or_kept_where_budgets_are_not_enforced() -> None:
    grants = AsyncMock(spec=OAuthAppGrantRepository)
    service = _service(grants, spend_limits=False)
    service.workspaces = AsyncMock(spec=WorkspaceService)
    assert not service.details("app", "https://app.example/callback").spend_limits
    await service.authorize(
        cast(User, Mock(id="user")),
        AppAuthorizationRequest(
            client_id="app",
            redirect_uri="https://app.example/callback",
            state="s" * 43,
            code_challenge="c" * 43,
            code_challenge_method="S256",
            workspace_id=uuid.uuid4(),
            scopes=list(DEFAULT_KEY_SCOPES),
            limit=Decimal("5"),
        ),
    )
    assert grants.issue.await_args is not None
    assert grants.issue.await_args.kwargs["spend_limit"] is None


@pytest.mark.parametrize(
    "client_id,redirect_uri",
    [
        # A registered client keeps its own callbacks even when open apps are on.
        ("app", "https://app.example/elsewhere"),
        ("https://someone.github.io", "https://other.github.io/"),
    ],
)
def test_open_apps_admit_no_callback_off_the_apps_own(client_id: str, redirect_uri: str) -> None:
    service = _service(AsyncMock(spec=OAuthAppGrantRepository), open_apps=True)
    with pytest.raises(OAuthAppGrantError):
        service.details(client_id, redirect_uri)


def test_without_open_apps_a_site_is_unknown() -> None:
    service = _service(AsyncMock(spec=OAuthAppGrantRepository))
    with pytest.raises(OAuthAppGrantError):
        service.details("https://someone.github.io", "https://someone.github.io/")


@pytest.mark.asyncio
async def test_redemption_uses_the_same_instant_for_lookup_and_atomic_consumption() -> None:
    now = datetime(2026, 11, 10, tzinfo=UTC)
    clock = Mock(side_effect=[now, now + timedelta(minutes=3)])
    grants = AsyncMock(spec=OAuthAppGrantRepository)
    grants.find_bound.return_value = OAuthAppGrant(
        code_hash="hash",
        user_id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        scopes=KeyScopeSet.checked(["inference:run"], is_service_key=False),
        expires_at=now + timedelta(seconds=1),
    )
    grants.consume.return_value = True
    keys = AsyncMock(spec=ApiKeyService)
    keys.issue_app_key.return_value = ("issued-token", "key-id")
    organizations = AsyncMock(spec=OrganizationService)
    organizations.hold_delegated_workspace_access.return_value = User(id=uuid.uuid4())
    service = OAuthAppService(
        cast(UnitOfWork, MagicMock(spec=UnitOfWork)),
        grants,
        keys,
        cast(ApiKeyFormatPort, Mock()),
        organizations,
        cast(WorkspaceService, Mock()),
        AsyncMock(spec=BudgetService),
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
    assert await service.exchange(request) == (
        "issued-token",
        KeyScopeSet.checked(["inference:run"], is_service_key=False),
    )
    assert clock.call_count == 1
    assert grants.find_bound.call_args.kwargs["now"] == now
    grants.consume.assert_awaited_once_with("hash", now)
