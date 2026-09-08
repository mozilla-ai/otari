"""Connections: OAuth grants an application's users give for third-party apps (standalone mode only).

Two routers. ``router`` is the application-facing API under
``/v1/connections``, authenticated with the application's API key (or the
master key) and scoped to that key's workspace; every call names the user
with the same string the application puts in ``user`` on a completion. The
application never handles an OAuth code or a token: it gets a link to send the
browser to, and later asks what the user has connected.

``callback_router`` is the one path a provider's redirect lands on. It is
unauthenticated on purpose: the browser arriving there belongs to the user,
not to the application, and proves nothing but the ``state`` it carries,
which Otari minted and bound to an end user, a provider and a return URL. It
finishes the exchange and sends the browser to that return URL with the
outcome in the query, so the application's page can refresh what it shows.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps import get_config, get_db, verify_api_key_or_master_key
from gateway.api.routes._public_auth import throttle_public_auth
from gateway.core.config import CONNECTED_APP_PROVIDERS, GatewayConfig
from gateway.models.entities import APIKey
from gateway.services.tenancy.connected_account_service import (
    MAX_EXTERNAL_USER_ID,
    AccessToken,
    AuthorizePublic,
    AuthorizeRequest,
    ConnectedAccountPublic,
    ConnectedAccountService,
    ConnectedAccountsPublic,
    ConnectedAccountUpdate,
    ConnectedAppPublic,
    with_query,
)
from gateway.services.tenancy.errors import TenancyError
from gateway.services.workspace_scope import resolve_workspace_id

# The credential on the router, as the other management routers declare it, so
# a route added here starts gated; ``get_workspace_id`` reads the same cached
# dependency to learn which workspace the caller may see.
router = APIRouter(
    prefix="/v1/connections", tags=["connections"], dependencies=[Depends(verify_api_key_or_master_key)]
)
callback_router = APIRouter(tags=["connections"])

ProviderPath = Annotated[
    str,
    Path(description="Which app.", pattern=f"^({'|'.join(CONNECTED_APP_PROVIDERS)})$"),
]
UserQuery = Annotated[
    str,
    Query(min_length=1, max_length=MAX_EXTERNAL_USER_ID, description="Your application's id for the user."),
]


def get_connected_account_service(
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> ConnectedAccountService:
    """Build the service on the request's session."""
    return ConnectedAccountService(db, config)


async def get_workspace_id(
    credential: Annotated[tuple[APIKey | None, bool], Depends(verify_api_key_or_master_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> uuid.UUID:
    """The workspace the caller's key belongs to, which scopes every connection it can see.

    An API key names its workspace; the master key falls back to the
    deployment's default workspace, the same rule the request plane applies
    (``workspace_scope.resolve_workspace_id``).
    """
    api_key, _ = credential
    return await resolve_workspace_id(db, api_key)


ServiceDep = Annotated[ConnectedAccountService, Depends(get_connected_account_service)]
WorkspaceDep = Annotated[uuid.UUID, Depends(get_workspace_id)]


@router.get("/apps")
async def list_connected_apps(
    service: ServiceDep, workspace_id: WorkspaceDep, user: UserQuery | None = None
) -> list[ConnectedAppPublic]:
    """The apps this deployment can connect and the scopes each asks for; with ``user``, how many accounts they hold."""
    return await service.list_apps(workspace_id, user)


@router.post("/{provider}/authorize")
async def authorize_connection(
    service: ServiceDep, workspace_id: WorkspaceDep, provider: ProviderPath, body: AuthorizeRequest
) -> AuthorizePublic:
    """Start connecting an account for one of your users: the link to send their browser to.

    Otari runs the consent flow and the callback. The browser ends up at
    ``return_url`` with ``connection=ok&provider=…&connection_id=…`` or
    ``connection=error&provider=…&reason=…`` appended. The link is good for ten
    minutes and one exchange.
    """
    return await service.authorize(workspace_id, body.user, provider, scopes=body.scopes, return_url=body.return_url)


@router.get("")
async def list_connections(
    service: ServiceDep, workspace_id: WorkspaceDep, user: UserQuery, provider: str | None = None
) -> ConnectedAccountsPublic:
    """A user's connected accounts, optionally for one provider. Tokens are never included."""
    return await service.list_accounts(workspace_id, user, provider)


@router.get("/{provider}/token")
async def get_connection_token(
    service: ServiceDep, workspace_id: WorkspaceDep, provider: ProviderPath, user: UserQuery
) -> AccessToken:
    """The live credential for a user's account with an app, for code that calls the app itself.

    Refreshed first when it expires within a minute. This is the one place a
    token leaves Otari, and it goes only to the application whose users these
    are: the same application that registered the OAuth client. A user with no
    account for the app answers 404.
    """
    token = await service.access_token_for_provider(workspace_id, user, provider)
    if token is None:
        raise _NoConnection(provider, user)
    return token


@router.get("/{connection_id}")
async def get_connection(
    service: ServiceDep, workspace_id: WorkspaceDep, connection_id: uuid.UUID, user: UserQuery
) -> ConnectedAccountPublic:
    """One connected account."""
    return await service.get_account(workspace_id, user, connection_id)


@router.patch("/{connection_id}")
async def update_connection(
    service: ServiceDep,
    workspace_id: WorkspaceDep,
    connection_id: uuid.UUID,
    user: UserQuery,
    body: ConnectedAccountUpdate,
) -> ConnectedAccountPublic:
    """Set or clear the user's own label for an account ("Work Slack")."""
    return await service.update(workspace_id, user, connection_id, body)


@router.delete("/{connection_id}", status_code=status.HTTP_204_NO_CONTENT)
async def disconnect_connection(
    service: ServiceDep, workspace_id: WorkspaceDep, connection_id: uuid.UUID, user: UserQuery
) -> None:
    """Disconnect an account: revoke the grant at the provider when it supports that, then forget the tokens."""
    await service.disconnect(workspace_id, user, connection_id)


class _NoConnection(TenancyError):
    status_code = status.HTTP_404_NOT_FOUND

    def __init__(self, provider: str, user: str) -> None:
        super().__init__(f"user {user!r} has no connected {provider} account")


@callback_router.get("/connected-accounts/{provider}/callback", include_in_schema=False)
async def connection_callback(
    request: Request,
    service: ServiceDep,
    provider: ProviderPath,
    state: Annotated[str, Query(min_length=1, max_length=128)],
    code: Annotated[str | None, Query(max_length=2048)] = None,
    error: Annotated[str | None, Query(max_length=200)] = None,
) -> RedirectResponse:
    """Where the provider sends the browser; finishes the flow and bounces to the return URL.

    Throttled per client address like the other public routes: there is no
    legitimate caller here at a rate worth exempting. Every outcome is a
    redirect, never an error page, so the application's page always regains
    the browser and can show what happened.
    """
    throttle_public_auth(request)
    if error or not code:
        target = await service.return_url_for_state(state)
        return RedirectResponse(
            with_query(target, connection="error", provider=provider, reason=error or "no_code"),
            status_code=status.HTTP_302_FOUND,
        )
    try:
        account, target = await service.complete(state=state, code=code)
    except TenancyError as failure:
        target = await service.return_url_for_state(state)
        reason = type(failure).__name__.removeprefix("ConnectedAccount").removesuffix("Error").lower() or "failed"
        return RedirectResponse(
            with_query(target, connection="error", provider=provider, reason=reason),
            status_code=status.HTTP_302_FOUND,
        )
    return RedirectResponse(
        with_query(target, connection="ok", provider=provider, connection_id=str(account.id)),
        status_code=status.HTTP_302_FOUND,
    )
