import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.api.deps import CurrentIdentity, get_config, get_db, require_deployment_operator
from gateway.auth.models import generate_api_key, hash_key, key_prefix
from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey, User
from gateway.models.tenancy import Workspace
from gateway.repositories.users_repository import get_or_create_default_user
from gateway.services.model_access import is_allowlist_subset, validate_allowed_models
from gateway.services.tenancy import OrganizationService
from gateway.services.workspace_scope import organization_default_workspace_id

# A key inherits its user's default allow-list and may narrow it, never broaden
# it, so a key list that is not a subset of the user's is rejected on write.
_KEY_EXCEEDS_USER_DETAIL = (
    "This key's allowed_models exceeds its user's default; a key can only narrow "
    "the user's model access, not broaden it."
)

router = APIRouter(prefix="/v1/keys", tags=["keys"])


async def _caller_organization_id(
    db: Annotated[AsyncSession, Depends(get_db)],
    identity: CurrentIdentity,
) -> uuid.UUID:
    """The organization this request acts in.

    A key is minted, listed and revoked inside one organization, so every route
    here resolves the caller's before it touches a row. A dashboard session names
    the identity behind it and resolves that identity's active organization,
    which is what ``POST /v1/organizations/me/switch`` moves; a header master key
    names nobody, resolves the bootstrap operator, and therefore acts in the
    default organization. That is the same rule ``services/workspace_scope``
    already documents for a deployment-wide write, so an operator running several
    organizations behind one gateway works in the one they are currently in
    rather than across all of them (otari#817).
    """
    return (await OrganizationService(db).get_active_organization_for_user(identity)).id


CallerOrganization = Annotated[uuid.UUID, Depends(_caller_organization_id)]


async def _load_key_in_organization(
    db: AsyncSession,
    key_id: str,
    organization_id: uuid.UUID,
    *,
    owner_user_id: str | None = None,
) -> APIKey:
    """Load a key by id within one organization, or raise 404.

    Joined to the workspace rather than loaded by id alone: a key belongs to
    exactly one workspace and a workspace to exactly one organization, so this is
    what keeps a read, a rotation or a revoke inside the caller's tenant. A key in
    another organization answers the same 404 as one that does not exist, the
    ``resolve_workspace_in_organization`` rule in
    `services/tenancy/authorization.py`: a 403 would confirm the id names a real
    key somewhere.

    ``owner_user_id`` narrows the load to keys billed to that owner, for the
    member-scoped routes in ``organization_keys.py``: somebody else's key in the
    caller's own organization answers the same 404, for the same reason.
    """
    statement = (
        select(APIKey)
        .join(Workspace, col(Workspace.id) == col(APIKey.workspace_id))
        .where(col(APIKey.id) == key_id, col(Workspace.organization_id) == organization_id)
    )
    if owner_user_id is not None:
        statement = statement.where(col(APIKey.user_id) == owner_user_id)
    result = await db.execute(statement)
    key = result.scalar_one_or_none()

    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key with id '{key_id}' not found",
        )

    return key


class CreateKeyRequest(BaseModel):
    """Request model for creating a new API key."""

    key_name: str | None = Field(default=None, description="Optional name for the key")
    user_id: str | None = Field(default=None, description="Optional user ID to associate with this key")
    expires_at: datetime | None = Field(default=None, description="Optional expiration timestamp")
    allowed_models: list[str] | None = Field(
        default=None,
        description="Model allow-list: null = any model, [] = deny all, or canonical "
        "instance:model entries (with instance:* / instance:prefix* wildcards).",
    )
    exclude_from_budget: bool = Field(
        default=False,
        description="When true, requests on this key are logged with cost but never reserved, "
        "reconciled into the user's spend, or gated by budget.",
    )
    reject_user_mismatch: bool | None = Field(
        default=None,
        description="Per-key override of the deployment-wide reject_user_mismatch setting: "
        "null (default) inherits it, true always rejects a request naming a different 'user', "
        "false always accepts it. Spend binds to this key's own user either way.",
    )
    capture_agent_telemetry: bool | None = Field(
        default=None,
        description="Per-key override of the deployment-wide capture_agent_telemetry setting: "
        "null (default) inherits it, true always stores this key's coding-agent telemetry, false "
        "always discards it. Covers both behavioral events (tool_result, tool_decision, "
        "user_prompt, api_error) from POST /v1/logs and outcome-metric data points (lines of code, "
        "commits, pull requests, active time) from POST /v1/metrics. Usage capture and billing are "
        "unaffected either way.",
    )
    workspace_id: uuid.UUID | None = Field(
        default=None,
        description="Workspace this key belongs to, which must be one in the caller's "
        "organization. Omitted means that organization's default workspace. A key belongs "
        "to exactly one workspace: requests on it are scoped and billed there, so the "
        "workspace is read off the key rather than off a request header.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class CreateKeyResponse(BaseModel):
    """Response model for creating a new API key."""

    id: str
    key: str
    # Leading characters of the key, echoed so the client can key its show-once
    # reveal to the same fingerprint the list will display afterward.
    key_prefix: str | None
    key_name: str | None
    user_id: str | None
    created_at: str
    expires_at: str | None
    is_active: bool
    allowed_models: list[str] | None
    exclude_from_budget: bool
    reject_user_mismatch: bool | None
    capture_agent_telemetry: bool | None
    metadata: dict[str, Any]


class KeyInfo(BaseModel):
    """Response model for key information."""

    id: str
    # Display-only fingerprint (leading characters of the plaintext key). Null for
    # keys minted before the prefix was recorded; the full key is never returned.
    key_prefix: str | None
    key_name: str | None
    user_id: str | None
    created_at: str
    last_used_at: str | None
    expires_at: str | None
    is_active: bool
    allowed_models: list[str] | None
    exclude_from_budget: bool
    reject_user_mismatch: bool | None
    capture_agent_telemetry: bool | None
    workspace_id: uuid.UUID
    metadata: dict[str, Any]

    @classmethod
    def from_model(cls, key: APIKey) -> "KeyInfo":
        return cls(
            id=str(key.id),
            workspace_id=key.workspace_id,
            key_prefix=str(key.key_prefix) if key.key_prefix else None,
            key_name=str(key.key_name) if key.key_name else None,
            user_id=str(key.user_id) if key.user_id else None,
            created_at=key.created_at.isoformat(),
            last_used_at=key.last_used_at.isoformat() if key.last_used_at else None,
            expires_at=key.expires_at.isoformat() if key.expires_at else None,
            is_active=bool(key.is_active),
            allowed_models=list(key.allowed_models) if key.allowed_models is not None else None,
            exclude_from_budget=bool(key.exclude_from_budget),
            reject_user_mismatch=None if key.reject_user_mismatch is None else bool(key.reject_user_mismatch),
            capture_agent_telemetry=(
                None if key.capture_agent_telemetry is None else bool(key.capture_agent_telemetry)
            ),
            metadata=dict(key.metadata_) if key.metadata_ else {},
        )


class UpdateKeyRequest(BaseModel):
    """Request model for updating a key."""

    key_name: str | None = None
    is_active: bool | None = None
    expires_at: datetime | None = None
    exclude_from_budget: bool | None = None
    # Tri-state via model_fields_set, like allowed_models below: absent =
    # unchanged, null = clear to inheriting the deployment setting, true/false =
    # pin this key strict/lenient.
    reject_user_mismatch: bool | None = None
    # Tri-state via model_fields_set, same idiom: absent = unchanged, null =
    # clear to inheriting the deployment setting, true/false = pin this key's
    # behavioral-capture on/off.
    capture_agent_telemetry: bool | None = None
    # Tri-state via model_fields_set: absent = unchanged, null = clear to
    # unrestricted, [] = deny all, list = restrict. A plain default cannot tell
    # "absent" from "explicit null", so the handler checks model_fields_set.
    allowed_models: list[str] | None = None
    metadata: dict[str, Any] | None = None


@router.post("", dependencies=[Depends(require_deployment_operator)])
async def create_key(
    request: CreateKeyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    organization_id: CallerOrganization,
) -> CreateKeyResponse:
    """Create a new API key in the caller's organization.

    Requires master key authentication.

    If user_id is provided, the key will be associated with that user (creates user if it doesn't exist).
    If user_id is not provided, the key is associated with the shared "default" user, which is created
    on first use. Keys without an explicit owner therefore share one identity, and so share budget,
    usage, and files.

    ``workspace_id`` names a workspace in the caller's organization, and omitting
    it mints into that organization's default workspace. A key resolves that
    organization's provider credentials and bills there, so minting into another
    organization's workspace would spend its budget on its credentials.
    """
    try:
        allowed_models = validate_allowed_models(config, request.allowed_models)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    api_key = generate_api_key()
    key_hash = hash_key(api_key)
    key_id = uuid.uuid4()

    if request.user_id:
        result = await db.execute(select(User).where(User.user_id == request.user_id))
        user = result.scalar_one_or_none()
        if not user:
            user = User(
                user_id=request.user_id,
                alias=f"User {request.user_id}",
            )
            db.add(user)
        elif user.deleted_at is not None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User '{request.user_id}' has been deleted. Recreate via POST /v1/users first.",
            )
        user_id = request.user_id
    else:
        # No owner given: attach to the shared "default" user rather than minting a
        # throwaway per-key user, so the key still has a real, visible, budgetable
        # owner and nothing is untracked.
        user = await get_or_create_default_user(db)
        user_id = user.user_id

    # A key must not grant more than its user's default (a freshly created user
    # has no default, so this only bites when attaching to a restricted user).
    if not is_allowlist_subset(allowed_models, user.allowed_models):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_KEY_EXCEEDS_USER_DETAIL)

    # Checked rather than left to the foreign key: an id naming no workspace is
    # a bad request, and letting it reach the constraint answered 500 "Database
    # error" for a value the caller supplied and can fix. Checked for ownership
    # and not only existence, and answering the same 404 either way, so a caller
    # cannot use this route to discover another organization's workspace ids.
    if request.workspace_id is not None:
        named = await db.execute(
            select(col(Workspace.id)).where(
                col(Workspace.id) == request.workspace_id,
                col(Workspace.organization_id) == organization_id,
            )
        )
        if named.scalar_one_or_none() is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workspace '{request.workspace_id}' not found",
            )
        workspace_id = request.workspace_id
    else:
        resolved = await organization_default_workspace_id(db, organization_id)
        if resolved is None:
            # Not reachable through any path that creates an organization here,
            # each of which provisions a workspace; refused rather than
            # provisioned so a mint never creates a workspace inside a tenant.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This organization has no workspace to hold a key; create one first.",
            )
        workspace_id = resolved

    db_key = APIKey(
        id=str(key_id),
        workspace_id=workspace_id,
        key_hash=key_hash,
        key_prefix=key_prefix(api_key),
        key_name=request.key_name,
        user_id=user_id,
        expires_at=request.expires_at,
        allowed_models=allowed_models,
        exclude_from_budget=request.exclude_from_budget,
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


@router.get("", dependencies=[Depends(require_deployment_operator)])
async def list_keys(
    db: Annotated[AsyncSession, Depends(get_db)],
    organization_id: CallerOrganization,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    workspace_id: Annotated[uuid.UUID | None, Query(description="Only keys in this workspace.")] = None,
) -> list[KeyInfo]:
    """List the API keys in the caller's organization.

    Requires master key authentication. An unset ``workspace_id`` lists every key
    in that organization; naming a workspace in another one lists nothing rather
    than refusing, so the filter reports no more than the unfiltered read does.
    """
    statement = (
        select(APIKey)
        .join(Workspace, col(Workspace.id) == col(APIKey.workspace_id))
        .where(col(Workspace.organization_id) == organization_id)
    )
    if workspace_id is not None:
        statement = statement.where(col(APIKey.workspace_id) == workspace_id)
    result = await db.execute(statement.offset(skip).limit(limit))
    keys = result.scalars().all()

    return [KeyInfo.from_model(key) for key in keys]


@router.get("/{key_id}", dependencies=[Depends(require_deployment_operator)])
async def get_key(
    key_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    organization_id: CallerOrganization,
) -> KeyInfo:
    """Get details of a specific API key in the caller's organization.

    Requires master key authentication.
    """
    key = await _load_key_in_organization(db, key_id, organization_id)

    return KeyInfo.from_model(key)


@router.patch("/{key_id}", dependencies=[Depends(require_deployment_operator)])
async def update_key(
    key_id: str,
    request: UpdateKeyRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    organization_id: CallerOrganization,
) -> KeyInfo:
    """Update an API key in the caller's organization.

    Requires master key authentication.
    """
    key = await _load_key_in_organization(db, key_id, organization_id)

    if request.key_name is not None:
        key.key_name = request.key_name
    if request.is_active is not None:
        key.is_active = request.is_active
    if request.expires_at is not None:
        key.expires_at = request.expires_at
    if request.exclude_from_budget is not None:
        key.exclude_from_budget = request.exclude_from_budget
    if "reject_user_mismatch" in request.model_fields_set:
        key.reject_user_mismatch = request.reject_user_mismatch
    if "capture_agent_telemetry" in request.model_fields_set:
        key.capture_agent_telemetry = request.capture_agent_telemetry
    # Tri-state: only touch the allow-list when the field was supplied. A supplied
    # null clears to unrestricted; [] denies all; a list restricts.
    if "allowed_models" in request.model_fields_set:
        try:
            new_allowed = validate_allowed_models(config, request.allowed_models)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        # Enforce narrow-only against the key's user default (see create_key).
        if key.user_id is not None:
            user_default = (
                await db.execute(select(User.allowed_models).where(User.user_id == key.user_id))
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


@router.post("/{key_id}/rotate", dependencies=[Depends(require_deployment_operator)])
async def rotate_key(
    key_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    organization_id: CallerOrganization,
) -> CreateKeyResponse:
    """Rotate an API key's secret in place, within the caller's organization.

    Requires master key authentication.

    Generates a new secret for the same key row (id, user, name, expiry, and
    metadata are preserved) and returns the new raw key once, using the same
    response shape as key creation. The previous secret stops authenticating
    immediately; there is no grace window.
    """
    key = await _load_key_in_organization(db, key_id, organization_id)

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


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_deployment_operator)])
async def delete_key(
    key_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    organization_id: CallerOrganization,
) -> None:
    """Delete (revoke) an API key in the caller's organization.

    Requires master key authentication.
    """
    key = await _load_key_in_organization(db, key_id, organization_id)

    await db.delete(key)
    try:
        await db.commit()
    except SQLAlchemyError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error",
        ) from None
