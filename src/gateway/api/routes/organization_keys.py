"""The caller's own API keys, for a tenant who does not operate the deployment.

``/api/v1/keys`` is deployment-wide and operator-only (otari-ai#1880), which left a
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
  travel on, and somebody else's key answers the 404 a nonexistent one does. An
  owner an operator has revoked through ``DELETE /api/v1/users`` stays revoked: this
  surface refuses rather than reviving the row, which is what would restore the
  spend that route deactivated.
* **The workspace must be one the caller may see.** A named ``workspace_id``
  goes through ``resolve_workspace_in_organization``, so a workspace in another
  organization, or one in this organization the caller is not a member of,
  answers 404 exactly as one that does not exist. Omitting it targets the
  organization's default workspace, put through the same check.
* **A viewer is read-only.** Creating, changing, rotating or revoking a key
  takes a role above ``viewer`` in both the organization and the key's
  workspace (or organization owner/admin), through
  ``OrganizationService.require_workspace_contributor_access``.
  A viewer still lists the keys billed to them, so an operator can issue them one.
* **No budget bypass.** The operator's create takes ``exclude_from_budget``;
  this one does not, and always mints an enforced key: a member exempting their
  own key from budget would be minting their own unlimited spend.
* **A service key belongs to the organization.** Only an owner or admin creates
  one, on this organization's own budgets, and every owner or admin manages it.
  It gets an owner of its own rather than the caller's attribution row, which
  serves every organization the caller belongs to: end users belong to their
  owner, so sharing it would let another organization's key reach them. That
  owner carries no member ceiling, which is why only someone who sets those
  ceilings may create one.

The operator's deployment-wide view keeps the gate it has, unchanged. A key
minted here shows up there like any other row in the organization, and the
narrow-only ``allowed_models`` rule binds against the caller's own user default
exactly as it does on the operator surface.
"""

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.api.deps import (
    ApiKeyFormatPortDep,
    ApiKeyServiceDep,
    BudgetServiceDep,
    CurrentIdentity,
    DeploymentUserServiceDep,
    GrowthSignalPortDep,
    OrganizationServiceDep,
    get_config,
    get_db,
    verify_master_key,
)
from gateway.api.routes._helpers import database_error
from gateway.api.routes.keys import (
    CreateKeyResponse,
    KeyInfo,
    _load_key_in_organization,
    _organization_service_key,
)
from gateway.auth.models import hash_key, key_suffix
from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS
from gateway.exceptions.organizations_exceptions import WorkspaceNotFoundError
from gateway.models.api_keys import MAX_END_USER_BUDGETS, APIKey
from gateway.models.tenancy import User as TenancyUser
from gateway.models.tenancy import Workspace
from gateway.models.users import User
from gateway.ports.growth_signal_port import GrowthActivationEvent
from gateway.repositories.users_repository import add_service_key_owner, get_or_create_attribution_user
from gateway.schemas.api_keys import FederationRuleCreate, FederationRulePublic
from gateway.services.api_keys import (
    DEFAULT_ORGANIZATION_MAX_END_USERS,
    END_USER_SETTINGS,
    admit_organization_key,
    organization_key_changes,
    refuse_service_key_flag_change,
)
from gateway.services.model_access import (
    KEY_MODELS_EXCEED_USER_DETAIL,
    KEY_TOOLS_EXCEED_USER_DETAIL,
    is_allowlist_subset,
    tools_within_user,
    validate_allowed_models,
    validate_allowed_tools,
)
from gateway.services.tenancy import DeploymentUserService, OrganizationService
from gateway.services.tenancy.authorization import resolve_workspace_in_organization
from gateway.services.workspace_scope import organization_default_workspace_id

router = APIRouter(
    prefix="/organizations/me/keys",
    tags=["organization-keys"],
    # Authentication only, like the rest of the ``/api/v1/organizations/me`` surface.
    # What the caller may touch is decided per request by the owner predicate and
    # the workspace resolver below, which is why the deployment operator gate
    # does not belong here.
    dependencies=[Depends(verify_master_key)],
)


class CreateOwnKeyRequest(BaseModel):
    """Create-key body for the member surface.

    Deliberately not ``CreateKeyRequest``: it carries no ``user_id`` (the owner
    is the caller, or one of the key's own for a service key) and no
    ``exclude_from_budget`` (a member must not exempt their own spend from
    enforcement). A field it does not take is refused rather than dropped.
    """

    model_config = ConfigDict(extra="forbid")

    key_name: str | None = Field(default=None, description="Optional name for the key")
    expires_at: datetime | None = Field(default=None, description="Optional expiration timestamp")
    allowed_models: list[str] | None = Field(
        default=None,
        description="Model allow-list: null = any model your user default allows, [] = deny all, "
        "or canonical instance:model entries. A key can only narrow your own model access, "
        "never broaden it.",
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Search tool allow-list for POST /api/v1/search: null = inherit your own, [] = no tool, "
        "or search tool names ('*' for every search tool). A key can only narrow your own tool access.",
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
    is_service_key: bool = Field(
        default=False,
        description="When true, a request may name an end user in its 'user' field. Each end user is created on "
        "first use and billed to its own budget, while this key's own ceiling caps all of them together. Only an "
        "organization owner or admin may create one. It belongs to the organization rather than to you: every "
        "owner or admin manages it, and its end users are reachable from no other key. Fixed once the key exists.",
    )
    end_user_budget_id: str | None = Field(
        default=None,
        description="Budget of this organization each end user this key creates is capped at, unless the request "
        "names another with Otari-End-User-Budget. Null leaves end users capped only by this key's own ceiling.",
    )
    end_user_budget_ids: list[str] | None = Field(
        default=None,
        max_length=MAX_END_USER_BUDGETS,
        description="Budgets of this organization a request may start a new end user on by naming one in "
        "Otari-End-User-Budget. Null allows end_user_budget_id alone. When both are set, end_user_budget_id must "
        "be on the list.",
    )
    max_end_users: int | None = Field(
        default=None,
        ge=0,
        description="Most live end users this service key may have. A request that would create one more is "
        f"refused with 403 and end_user_limit_reached; deleting end users frees room. Null is "
        f"{DEFAULT_ORGANIZATION_MAX_END_USERS}: an organization's service key always has a cap.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class UpdateOwnKeyRequest(BaseModel):
    """Update-key body for the member surface.

    The operator's ``UpdateKeyRequest`` minus ``exclude_from_budget``, for the
    reason the create body gives. The tri-state fields follow the same
    ``model_fields_set`` idiom: absent = unchanged, null = clear to inherit,
    a value = pin it. ``is_service_key`` is accepted only unchanged, and
    ``max_end_users`` takes no null: an organization's service key always has a
    cap, so null is refused with 400.
    """

    model_config = ConfigDict(extra="forbid")

    key_name: str | None = None
    is_active: bool | None = None
    expires_at: datetime | None = None
    reject_user_mismatch: bool | None = None
    capture_agent_telemetry: bool | None = None
    # Tri-state via model_fields_set: absent = unchanged, null = clear to
    # unrestricted (within the user default), [] = deny all, list = restrict.
    allowed_models: list[str] | None = None
    # Tri-state the same way: null = inherit your own search tool access.
    allowed_tools: list[str] | None = None
    is_service_key: bool | None = None
    # A service key's alone. Tri-state via model_fields_set: absent = unchanged,
    # null = end users this key creates from now on are uncapped.
    end_user_budget_id: str | None = None
    # Tri-state via model_fields_set: absent = unchanged, null = the default alone.
    end_user_budget_ids: list[str] | None = Field(default=None, max_length=MAX_END_USER_BUDGETS)
    # Absent = unchanged; an organization's service key always has a cap.
    max_end_users: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] | None = None


async def _load_own_writable_key(
    db: AsyncSession,
    organizations: OrganizationService,
    identity: TenancyUser,
    key_id: str,
    *,
    revoking: bool = False,
) -> APIKey:
    """One of the caller's own keys, or 404, refused with 403 to a viewer of its workspace.

    An owner or admin also reaches the organization's service keys. A viewer
    still lists their keys; changing, rotating or revoking one is a write, and an
    operator who issued a key to a read-only member is the one who changes it.
    A caller removed from the key's workspace may still ``revoking`` it,
    switching it off or deleting it, and nothing else.
    """
    organization = await organizations.get_active_organization_for_user(identity)
    key = await _load_key_in_organization(
        db,
        key_id,
        organization.id,
        owner_user_id=str(identity.id),
        or_organization_service_keys=await organizations.manages_organization(user=identity, organization=organization),
    )
    await organizations.hold_workspace_write_access(
        user=identity, organization_id=organization.id, workspace_id=key.workspace_id, revoking=revoking
    )
    return key


@router.post("")
async def create_own_key(
    request: CreateOwnKeyRequest,
    identity: CurrentIdentity,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    growth: GrowthSignalPortDep,
    key_format: ApiKeyFormatPortDep,
    organizations: OrganizationServiceDep,
    budgets: BudgetServiceDep,
) -> CreateKeyResponse:
    """Create an API key owned by the caller, in a workspace they may see.

    The member-scoped counterpart of ``POST /api/v1/keys``: the owner is always the
    caller's own attribution user, the key is always budget-enforced, and the
    workspace must be visible to the caller (a member of it, or an organization
    owner/admin, who see every workspace). A viewer of the workspace is refused
    with 403. The secret is returned once.

    A service key is the organization's rather than the caller's: only an
    organization owner or admin may create one, its end-user budgets must be
    this organization's, and it is owned by a user minted for it alone.
    """
    organization = await organizations.get_active_organization_for_user(identity)
    end_users = await admit_organization_key(
        organizations,
        budgets,
        user=identity,
        organization=organization,
        is_service_key=request.is_service_key,
        requested=request.model_dump(include=set(END_USER_SETTINGS)),
    )

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
            # A viewer is refused whichever workspace they name, so they hear
            # that rather than advice to name one.
            await organizations.require_workspace_contributor_access(
                user=identity, organization_id=organization.id, workspace_id=resolved, membership_required=False
            )
            # The caller named nothing, so "not found" would point them at a
            # parameter they did not send; say what to do instead.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="You are not a member of this organization's default workspace; "
                "pass workspace_id naming a workspace you belong to.",
            ) from None
        workspace_id = workspace.id
    await organizations.hold_workspace_write_access(
        user=identity, organization_id=organization.id, workspace_id=workspace_id, revoking=False
    )

    try:
        allowed_models = validate_allowed_models(config, request.allowed_models)
        allowed_tools = validate_allowed_tools(config, request.allowed_tools)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # ``get_or_create_attribution_user`` revives a soft-deleted row. That is what
    # the membership paths calling it want (re-adding a member must find their
    # existing owner, not mint a second one) and the wrong answer here.
    # ``DELETE /api/v1/users`` is the operator's revocation of a spend identity: it
    # soft-deletes the row and deactivates every key it holds, and the data plane
    # then refuses a request whose owner is deleted. Reviving it is therefore
    # restoring spend, which is not a member's to do for themselves, so this
    # refuses where ``POST /api/v1/keys`` refuses the same owner.
    revoked = (await db.execute(select(User.deleted_at).where(User.user_id == str(identity.id)))).scalar_one_or_none()
    if revoked is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Your spend identity has been deactivated on this deployment; ask an operator to restore it.",
        )

    owner = await get_or_create_attribution_user(
        db,
        user_id=str(identity.id),
        alias=identity.full_name or identity.email,
    )
    # A key must not grant more than its user's default, the same narrow-only
    # rule the operator surface enforces; here the user is always the caller.
    if not is_allowlist_subset(allowed_models, owner.allowed_models, tool_keys=config.legacy_search_tool_keys()):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=KEY_MODELS_EXCEED_USER_DETAIL)
    if not tools_within_user(config, allowed_tools, user_tools=owner.allowed_tools, user_models=owner.allowed_models):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=KEY_TOOLS_EXCEED_USER_DETAIL)

    # ``record_activation`` puts first-occurrence detection on the caller, and
    # this schema has nowhere to record "already notified", so the milestone is
    # derived from the owner's own keys: an index seek on ``api_keys.user_id``,
    # read before the insert below because afterwards there is always one.
    first_key = (
        await db.execute(select(APIKey.id).where(APIKey.user_id == owner.user_id).limit(1))
    ).scalar_one_or_none() is None

    if request.is_service_key:
        owner = add_service_key_owner(db, alias=request.key_name or "Service key", creator=owner)

    api_key = key_format.mint()
    db_key = APIKey(
        id=str(uuid.uuid4()),
        workspace_id=workspace_id,
        key_hash=hash_key(api_key),
        key_prefix=key_format.fingerprint(api_key),
        key_suffix=key_suffix(api_key),
        key_name=request.key_name,
        user_id=owner.user_id,
        expires_at=request.expires_at,
        allowed_models=allowed_models,
        allowed_tools=allowed_tools,
        exclude_from_budget=False,
        self_service=True,
        reject_user_mismatch=request.reject_user_mismatch,
        capture_agent_telemetry=request.capture_agent_telemetry,
        is_service_key=request.is_service_key,
        **end_users,
        metadata_=request.metadata,
    )

    db.add(db_key)
    try:
        await db.commit()
    except DATABASE_ERRORS as exc:
        await db.rollback()
        raise database_error("create own key", exc) from None
    await db.refresh(db_key)

    if first_key and identity.email is not None:
        # Skipped for an identity with no address, which is the bootstrap
        # operator's normal state (`services/tenancy/provisioning_service`): the
        # port's ``email`` is what a vendor keys a person on, and there is
        # nothing to send in its place.
        await growth.record_activation(
            background_tasks=background_tasks,
            event=GrowthActivationEvent.API_KEY_CREATED,
            user_id=identity.id,
            email=identity.email,
        )

    key_info = KeyInfo.from_model(db_key)
    return CreateKeyResponse(
        **key_info.model_dump(exclude={"last_used_at"}),
        key=api_key,
    )


@router.get("")
async def list_own_keys(
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    organizations: OrganizationServiceDep,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    workspace_id: Annotated[uuid.UUID | None, Query(description="Only keys in this workspace.")] = None,
) -> list[KeyInfo]:
    """List the caller's own API keys in their active organization, newest first.

    Only keys billed to the caller's own user: an operator-minted key assigned
    to them is theirs to see here, and nobody else's key ever is. Naming a
    workspace outside their organization lists nothing rather than refusing, so
    the filter reports no more than the unfiltered read does. An owner or admin
    also sees the organization's service keys, which they manage.
    """
    organization = await organizations.get_active_organization_for_user(identity)
    owned = col(APIKey.user_id) == str(identity.id)
    if await organizations.manages_organization(user=identity, organization=organization):
        owned = or_(owned, _organization_service_key())
    statement = (
        select(APIKey)
        .join(Workspace, col(Workspace.id) == col(APIKey.workspace_id))
        .where(col(Workspace.organization_id) == organization.id, owned)
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
    api_keys: ApiKeyServiceDep,
    organizations: OrganizationServiceDep,
    budgets: BudgetServiceDep,
) -> KeyInfo:
    """Update one of the caller's own API keys, or one of the organization's service keys for an owner or admin."""
    fields = request.model_fields_set
    switching_off = fields == {"is_active"} and request.is_active is False
    key = await _load_own_writable_key(db, organizations, identity, key_id, revoking=switching_off)
    refuse_service_key_flag_change(key, request.is_service_key)

    # Tri-state via model_fields_set: the nullable columns take null to clear
    # them, so "absent" and "null" have to be told apart. ``is_active`` and
    # ``metadata`` are not nullable, so ``is not None`` is the whole distinction there.
    changes = request.model_dump(include=fields - {"is_active", "metadata", "is_service_key"} - END_USER_SETTINGS)
    if request.is_active is not None:
        changes["is_active"] = request.is_active
    if request.metadata is not None:
        changes["metadata_"] = request.metadata
    changes |= await organization_key_changes(
        organizations,
        budgets,
        user=identity,
        key=key,
        requested=request.model_dump(include=fields & END_USER_SETTINGS),
        revoking=switching_off,
    )

    try:
        await api_keys.update_owned(config, key, changes, owner_id=str(key.user_id))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except DATABASE_ERRORS as exc:
        raise database_error("update own key", exc) from None

    return KeyInfo.from_model(key)


@router.post("/{key_id}/rotate")
async def rotate_own_key(
    key_id: str,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    key_format: ApiKeyFormatPortDep,
    organizations: OrganizationServiceDep,
) -> CreateKeyResponse:
    """Rotate the secret of one of the caller's own API keys, in place.

    Same contract as the operator's rotate: the row keeps its identity, the new
    raw key is returned once, and the previous secret stops authenticating
    immediately with no grace window.
    """
    key = await _load_own_writable_key(db, organizations, identity, key_id)

    new_api_key = key_format.mint()
    key.key_hash = hash_key(new_api_key)
    key.key_prefix = key_format.fingerprint(new_api_key)
    key.key_suffix = key_suffix(new_api_key)
    key.last_used_at = None

    try:
        await db.commit()
    except DATABASE_ERRORS as exc:
        await db.rollback()
        raise database_error("rotate own key", exc) from None
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
    organizations: OrganizationServiceDep,
) -> None:
    """Delete (revoke) one of the caller's own API keys."""
    key = await _load_own_writable_key(db, organizations, identity, key_id, revoking=True)

    await db.delete(key)
    try:
        await db.commit()
    except DATABASE_ERRORS as exc:
        await db.rollback()
        raise database_error("delete own key", exc) from None


async def _federation_rule_key(
    db: AsyncSession,
    organizations: OrganizationService,
    deployment_users: DeploymentUserService,
    identity: TenancyUser,
    key_id: str,
    *,
    adding: bool,
) -> APIKey:
    """The key a federation rule route acts on, or 404.

    The deployment operator may act on any key in their active organization, as on ``/api/v1/keys``; anyone else
    only on their own, and an owner or admin also on the organization's service keys. Adding a rule also takes an
    owner or admin of the organization (or the operator), since a rule decides which workloads may act as the key;
    listing and deleting do not, so a caller whose role was lowered can still see and take back what they set up.
    """
    operates = await deployment_users.has_administration_access(identity)
    organization = await organizations.get_active_organization_for_user(identity)
    if operates:
        return await _load_key_in_organization(db, key_id, organization.id)
    if adding:
        await organizations.require_active_organization_management_access(user=identity, organization=organization)
    return await _load_key_in_organization(
        db,
        key_id,
        organization.id,
        owner_user_id=str(identity.id),
        or_organization_service_keys=await organizations.manages_organization(user=identity, organization=organization),
    )


@router.post("/{key_id}/federation-rules", status_code=status.HTTP_201_CREATED)
async def create_own_federation_rule(
    key_id: str,
    request: FederationRuleCreate,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    config: Annotated[GatewayConfig, Depends(get_config)],
    api_keys: ApiKeyServiceDep,
    organizations: OrganizationServiceDep,
    deployment_users: DeploymentUserServiceDep,
) -> FederationRulePublic:
    """Let an OpenID Connect identity token stand in for a key, through the token exchange.

    Needs ``federation_member_rules``, else 403, and an organization owner or admin acting on their own key, or the
    deployment operator on any key in their active organization, else 403. The issuer may be any https URL; its
    signing keys are fetched only from a public host. The rule matches only in ``POST /api/v1/oauth/token`` naming
    its id, never as a bare identity token, so it lets a workload act as this key and no other: a rule naming another
    tenant's workload gives that workload nothing it did not ask for. A rule this key already holds answers 409.
    """
    if not config.federation_member_rules:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This deployment does not let organizations manage federation rules.",
        )
    key = await _federation_rule_key(db, organizations, deployment_users, identity, key_id, adding=True)
    return await api_keys.add_federation_rule(key.id, request, allowed_issuers=(), member_managed=True)


@router.get("/{key_id}/federation-rules")
async def list_own_federation_rules(
    key_id: str,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    api_keys: ApiKeyServiceDep,
    organizations: OrganizationServiceDep,
    deployment_users: DeploymentUserServiceDep,
) -> list[FederationRulePublic]:
    """List the identities that may authenticate as a key, oldest first.

    The caller's own key, or for the deployment operator any key in their active organization. Includes any rule
    the operator gave the key on ``/api/v1/keys``, which ``member_managed`` tells apart.
    """
    key = await _federation_rule_key(db, organizations, deployment_users, identity, key_id, adding=False)
    return await api_keys.list_federation_rules(key.id)


@router.delete("/{key_id}/federation-rules/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_own_federation_rule(
    key_id: str,
    rule_id: str,
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    api_keys: ApiKeyServiceDep,
    organizations: OrganizationServiceDep,
    deployment_users: DeploymentUserServiceDep,
) -> Response:
    """Remove a federation rule from a key. Tokens it matched stop with the next request.

    The caller's own key whatever their role, or for the deployment operator any key in their active organization.
    Works whether or not ``federation_member_rules`` is on, so a rule can always be taken back.
    """
    key = await _federation_rule_key(db, organizations, deployment_users, identity, key_id, adding=False)
    await api_keys.delete_federation_rule(key.id, rule_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
