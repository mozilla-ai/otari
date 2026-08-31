"""The caller's own API keys, for a tenant who does not operate the deployment.

``/v1/keys`` is deployment-wide and operator-only (otari-ai#1880), which left a
hosted organization member with no way to mint a key at all: they could not use
the product without an operator handing them one out of band
(mozilla-ai/otari-ai#1941). The answer is not a looser gate on that router,
which scopes to the caller's whole active organization and takes a client-named
``user_id``; it is the same second-surface shape ``organization_usage.py``
established for usage reads (otari#837), applied to a write surface:

* **Ownership is derived, never accepted.** A key minted here is billed to the
  caller's own attribution row (``users.user_id`` = the identity's UUID as a
  string, the row ``get_or_create_attribution_user`` mints and the activation
  guide already keys on), and every load below carries that owner predicate.
  There is no ``user_id`` parameter, so there is nothing for an escalation to
  travel on, and somebody else's key answers the 404 a nonexistent one does.
* **The workspace must be one the caller may see.** A named ``workspace_id``
  goes through ``resolve_workspace_in_organization``, so a workspace in another
  organization, or one in this organization the caller is not a member of,
  answers 404 exactly as one that does not exist. Omitting it targets the
  organization's default workspace, put through the same check.
* **No budget bypass.** The operator's create takes ``exclude_from_budget``;
  this one does not, and always mints an enforced key: a member exempting their
  own key from budget would be minting their own unlimited spend.

The operator's deployment-wide view keeps the gate it has, unchanged. A key
minted here shows up there like any other row in the organization, and the
narrow-only ``allowed_models`` rule binds against the caller's own user default
exactly as it does on the operator surface.
"""

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.api.deps import CurrentIdentity, get_config, get_db, verify_master_key
from gateway.api.routes.keys import (
    _KEY_EXCEEDS_USER_DETAIL,
    CreateKeyResponse,
    KeyInfo,
    _load_key_in_organization,
)
from gateway.auth.models import generate_api_key, hash_key, key_prefix
from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey, User
from gateway.models.tenancy import User as TenancyUser
from gateway.models.tenancy import Workspace
from gateway.repositories.users_repository import get_or_create_attribution_user
from gateway.services.model_access import is_allowlist_subset, validate_allowed_models
from gateway.services.tenancy import OrganizationService
from gateway.services.tenancy.authorization import resolve_workspace_in_organization
from gateway.services.tenancy.errors import WorkspaceNotFoundError
from gateway.services.workspace_scope import organization_default_workspace_id

router = APIRouter(
    prefix="/v1/organizations/me/keys",
    tags=["organization-keys"],
    # Authentication only, like the rest of the ``/v1/organizations/me`` surface.
    # What the caller may touch is decided per request by the owner predicate and
    # the workspace resolver below, which is why the deployment operator gate
    # does not belong here.
    dependencies=[Depends(verify_master_key)],
)


class CreateOwnKeyRequest(BaseModel):
    """Create-key body for the member surface.

    Deliberately not ``CreateKeyRequest``: it carries no ``user_id`` (the owner
    is always the caller) and no ``exclude_from_budget`` (a member must not
    exempt their own spend from enforcement).
    """

    key_name: str | None = Field(default=None, description="Optional name for the key")
    expires_at: datetime | None = Field(default=None, description="Optional expiration timestamp")
    allowed_models: list[str] | None = Field(
        default=None,
        description="Model allow-list: null = any model your user default allows, [] = deny all, "
        "or canonical instance:model entries. A key can only narrow your own model access, "
        "never broaden it.",
    )
    reject_user_mismatch: bool | None = Field(
        default=None,
        description="Per-key override of the deployment-wide reject_user_mismatch setting: "
        "null (default) inherits it, true always rejects a request naming a different 'user', "
        "false always accepts it. Spend binds to your own user either way.",
    )
    capture_agent_telemetry: bool | None = Field(
        default=None,
        description="Per-key override of the deployment-wide capture_agent_telemetry setting: "
        "null (default) inherits it, true always stores this key's coding-agent telemetry, "
        "false always discards it.",
    )
    workspace_id: uuid.UUID | None = Field(
        default=None,
        description="Workspace this key belongs to, which must be one you may see in your "
        "active organization. Omitted means that organization's default workspace, refused "
        "when you are not a member of it.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class UpdateOwnKeyRequest(BaseModel):
    """Update-key body for the member surface.

    The operator's ``UpdateKeyRequest`` minus ``exclude_from_budget``, for the
    reason the create body gives. The tri-state fields follow the same
    ``model_fields_set`` idiom: absent = unchanged, null = clear to inherit,
    a value = pin it.
    """

    key_name: str | None = None
    is_active: bool | None = None
    expires_at: datetime | None = None
    reject_user_mismatch: bool | None = None
    capture_agent_telemetry: bool | None = None
    # Tri-state via model_fields_set: absent = unchanged, null = clear to
    # unrestricted (within the user default), [] = deny all, list = restrict.
    allowed_models: list[str] | None = None
    metadata: dict[str, Any] | None = None


async def _caller_context(db: AsyncSession, identity: TenancyUser) -> tuple[uuid.UUID, str]:
    """The organization this request acts in, and the owner id it acts as.

    The organization is the caller's own ``active_organization_id``, resolved
    through ``get_active_organization_for_user`` so a pointer with no live
    membership behind it refuses rather than resolving; moving between
    organizations is ``POST /v1/organizations/me/switch``. The owner id is the
    identity's UUID rendered as a string, the attribution convention
    ``get_or_create_attribution_user`` documents.
    """
    organization = await OrganizationService(db).get_active_organization_for_user(identity)
    return organization.id, str(identity.id)


@router.post("")
async def create_own_key(
    request: CreateOwnKeyRequest,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> CreateKeyResponse:
    """Create an API key owned by the caller, in a workspace they may see.

    The member-scoped counterpart of ``POST /v1/keys``: the owner is always the
    caller's own attribution user, the key is always budget-enforced, and the
    workspace must be visible to the caller (a member of it, or an organization
    owner/admin/superuser, who see every workspace). The secret is returned once.
    """
    organizations = OrganizationService(db)
    organization = await organizations.get_active_organization_for_user(identity)

    if request.workspace_id is not None:
        workspace = await resolve_workspace_in_organization(
            db,
            user=identity,
            workspace_id=request.workspace_id,
            organization=organization,
            organizations=organizations,
        )
        workspace_id = workspace.id
    else:
        resolved = await organization_default_workspace_id(db, organization.id)
        if resolved is None:
            # Not reachable through any path that creates an organization here;
            # refused rather than provisioned, as on the operator surface.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This organization has no workspace to hold a key; create one first.",
            )
        try:
            workspace = await resolve_workspace_in_organization(
                db,
                user=identity,
                workspace_id=resolved,
                organization=organization,
                organizations=organizations,
            )
        except WorkspaceNotFoundError:
            # The caller named nothing, so "not found" would point them at a
            # parameter they did not send; say what to do instead.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="You are not a member of this organization's default workspace; "
                "pass workspace_id naming a workspace you belong to.",
            ) from None
        workspace_id = workspace.id

    try:
        allowed_models = validate_allowed_models(config, request.allowed_models)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    owner = await get_or_create_attribution_user(
        db,
        user_id=str(identity.id),
        alias=identity.full_name or identity.email,
    )
    # A key must not grant more than its user's default, the same narrow-only
    # rule the operator surface enforces; here the user is always the caller.
    if not is_allowlist_subset(allowed_models, owner.allowed_models):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_KEY_EXCEEDS_USER_DETAIL)

    api_key = generate_api_key()
    db_key = APIKey(
        id=str(uuid.uuid4()),
        workspace_id=workspace_id,
        key_hash=hash_key(api_key),
        key_prefix=key_prefix(api_key),
        key_name=request.key_name,
        user_id=owner.user_id,
        expires_at=request.expires_at,
        allowed_models=allowed_models,
        exclude_from_budget=False,
        reject_user_mismatch=request.reject_user_mismatch,
        capture_agent_telemetry=request.capture_agent_telemetry,
        metadata_=request.metadata,
    )

    db.add(db_key)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(db_key)

    key_info = KeyInfo.from_model(db_key)
    return CreateKeyResponse(
        **key_info.model_dump(exclude={"last_used_at"}),
        key=api_key,
    )


@router.get("")
async def list_own_keys(
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    workspace_id: Annotated[uuid.UUID | None, Query(description="Only keys in this workspace.")] = None,
) -> list[KeyInfo]:
    """List the caller's own API keys in their active organization, newest first.

    Only keys billed to the caller's own user: an operator-minted key assigned
    to them is theirs to see here, and nobody else's key ever is. Naming a
    workspace outside their organization lists nothing rather than refusing, so
    the filter reports no more than the unfiltered read does.
    """
    organization_id, owner_user_id = await _caller_context(db, identity)
    statement = (
        select(APIKey)
        .join(Workspace, col(Workspace.id) == col(APIKey.workspace_id))
        .where(
            col(Workspace.organization_id) == organization_id,
            col(APIKey.user_id) == owner_user_id,
        )
    )
    if workspace_id is not None:
        statement = statement.where(col(APIKey.workspace_id) == workspace_id)
    # Ordered so paging through more than one page is stable, with the id as a
    # tiebreak for keys minted in the same instant.
    statement = statement.order_by(col(APIKey.created_at).desc(), col(APIKey.id))
    result = await db.execute(statement.offset(skip).limit(limit))
    keys = result.scalars().all()

    return [KeyInfo.from_model(key) for key in keys]


@router.patch("/{key_id}")
async def update_own_key(
    key_id: str,
    request: UpdateOwnKeyRequest,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
) -> KeyInfo:
    """Update one of the caller's own API keys."""
    organization_id, owner_user_id = await _caller_context(db, identity)
    key = await _load_key_in_organization(db, key_id, organization_id, owner_user_id=owner_user_id)

    if request.key_name is not None:
        key.key_name = request.key_name
    if request.is_active is not None:
        key.is_active = request.is_active
    if request.expires_at is not None:
        key.expires_at = request.expires_at
    if "reject_user_mismatch" in request.model_fields_set:
        key.reject_user_mismatch = request.reject_user_mismatch
    if "capture_agent_telemetry" in request.model_fields_set:
        key.capture_agent_telemetry = request.capture_agent_telemetry
    if "allowed_models" in request.model_fields_set:
        try:
            new_allowed = validate_allowed_models(config, request.allowed_models)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        # Narrow-only against the caller's own user default (see create_own_key).
        user_default = (
            await db.execute(select(User.allowed_models).where(User.user_id == owner_user_id))
        ).scalar_one_or_none()
        if not is_allowlist_subset(new_allowed, user_default):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_KEY_EXCEEDS_USER_DETAIL)
        key.allowed_models = new_allowed
    if request.metadata is not None:
        key.metadata_ = request.metadata

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(key)

    return KeyInfo.from_model(key)


@router.post("/{key_id}/rotate")
async def rotate_own_key(
    key_id: str,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CreateKeyResponse:
    """Rotate the secret of one of the caller's own API keys, in place.

    Same contract as the operator's rotate: the row keeps its identity, the new
    raw key is returned once, and the previous secret stops authenticating
    immediately with no grace window.
    """
    organization_id, owner_user_id = await _caller_context(db, identity)
    key = await _load_key_in_organization(db, key_id, organization_id, owner_user_id=owner_user_id)

    new_api_key = generate_api_key()
    key.key_hash = hash_key(new_api_key)
    key.key_prefix = key_prefix(new_api_key)
    key.last_used_at = None

    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
    await db.refresh(key)

    key_info = KeyInfo.from_model(key)
    return CreateKeyResponse(
        **key_info.model_dump(exclude={"last_used_at"}),
        key=new_api_key,
    )


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_own_key(
    key_id: str,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Delete (revoke) one of the caller's own API keys."""
    organization_id, owner_user_id = await _caller_context(db, identity)
    key = await _load_key_in_organization(db, key_id, organization_id, owner_user_id=owner_user_id)

    await db.delete(key)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
