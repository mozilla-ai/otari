"""A person's own access tokens, which read the management API as them from an agent or a script."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status

from gateway.api.deps import (
    CurrentIdentity,
    PersonalAccessTokenServiceDep,
    is_personal_access_token_request,
    verify_master_key,
)
from gateway.core.oauth_apps import APP_KEY_LIFETIMES
from gateway.schemas.identity.access_tokens import (
    AccessTokenCreate,
    AccessTokenCreated,
    AccessTokenPublic,
    AccessTokensPublic,
)
from gateway.services.dashboard_session_service import SESSION_COOKIE_NAME

router = APIRouter(prefix="/auth/access-tokens", tags=["auth"], dependencies=[Depends(verify_master_key)])


@router.get("")
async def list_access_tokens(identity: CurrentIdentity, service: PersonalAccessTokenServiceDep) -> AccessTokensPublic:
    """The caller's own access tokens, newest first. Never anybody else's, and never a token itself."""
    return AccessTokensPublic(data=[AccessTokenPublic.of(token) for token in await service.list(identity)])


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_access_token(
    body: AccessTokenCreate, request: Request, identity: CurrentIdentity, service: PersonalAccessTokenServiceDep
) -> AccessTokenCreated:
    """Mint an access token that reads what the caller can see on the dashboard, as them.

    It reads the organization the caller is acting in now, even after the dashboard switches
    to another. It authenticates `GET` requests to the management API and the MCP server at
    `/mcp`, is refused on every write, and lives a year at most. The token is in this response only.
    """
    if is_personal_access_token_request(request):
        # Unreachable while tokens are read-only, and kept so a token never mints its successor.
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access tokens are read-only")
    lifetime = APP_KEY_LIFETIMES[body.expiration]
    if lifetime is None:  # pragma: no cover - the schema offers no expiry without an end
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Choose when the token expires")
    row, token = await service.create(
        identity,
        name=body.name,
        lifetime=lifetime,
        # Re-read under the account lock, so a session that ended while this request ran cannot
        # outlive the cleanup that ended it by minting a token.
        session_token=request.cookies.get(SESSION_COOKIE_NAME),
    )
    return AccessTokenCreated(**AccessTokenPublic.of(row).model_dump(), token=token)


@router.delete("/{token_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_access_token(
    token_id: uuid.UUID, identity: CurrentIdentity, service: PersonalAccessTokenServiceDep
) -> None:
    """Revoke one of the caller's access tokens at once."""
    await service.revoke(identity, token_id)
