"""First boot: the default organization, its workspace, and the operator identity.

Standalone Otari authenticates an operator with the master key, which proves
possession of a deployment-wide credential and names no user. Every tenancy
surface, on the other hand, reads off an identity with an active organization and
a role. This module is the bridge: the first master-key-authenticated request
provisions the default organization, its default workspace, and one owner
identity, and every later request resolves that same identity.

That is otari-ai#1716 option A: the master key is the bootstrap credential, and
it stops being the steady-state dashboard login once real sessions arrive. Two
things follow from being a bootstrap step rather than a login, and both are
deliberate:

- The operator identity has no email. It is an operator label, not a sign-in
  address, exactly like the gateway users M4 re-parents onto this table, and the
  claim flow that gives such an identity an address is a separate track.
- It is a superuser. The master key already carries deployment-wide authority,
  so scoping the identity it maps to more narrowly would be theater.

The identity is anchored by a ``runtime_settings`` row rather than by the default
organization's slug: an operator may rename, leave, or delete that organization,
and the anchor has to survive all three.
"""

import uuid

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.log_config import logger
from gateway.models.entities import RuntimeSetting
from gateway.models.tenancy import User
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.services.tenancy.errors import TenancyError

# Stored in runtime_settings, and deliberately not a SETTABLE_KEY, so
# runtime_settings_service ignores it exactly as it ignores the master-key hash
# and the dashboard-session marker.
BOOTSTRAP_IDENTITY_KEY = "tenancy_bootstrap_user_id"

DEFAULT_ORGANIZATION_NAME = "Default organization"
DEFAULT_ORGANIZATION_SLUG = "default"
DEFAULT_WORKSPACE_NAME = "Default workspace"
OPERATOR_FULL_NAME = "Operator"


class BootstrapIdentityUnavailableError(TenancyError):
    """First-boot provisioning lost a race and could not resolve the identity."""

    def __init__(self) -> None:
        super().__init__("Could not resolve the operator identity; retry the request")


async def ensure_bootstrap_identity(db: AsyncSession) -> User:
    """Return the operator identity, provisioning the tenancy root on first call.

    Idempotent: the marker row makes the common path a single indexed lookup and
    the provisioning path run once per deployment. Commits, because it is the
    unit of work that has to be durable before any request reads it.
    """
    existing = await _load_marked_identity(db)
    if existing is not None:
        return existing

    try:
        return await _provision(db)
    except IntegrityError:
        # Two first requests raced. Whichever lost re-reads the winner's work:
        # the slug and the marker are both unique, so exactly one provisioned.
        await db.rollback()
        logger.info("Concurrent first-boot tenancy provisioning; using the identity that won")
        resolved = await _load_marked_identity(db)
        if resolved is None:
            raise BootstrapIdentityUnavailableError from None
        return resolved


async def _load_marked_identity(db: AsyncSession) -> User | None:
    """Resolve the identity the marker names, or None if there is not one yet."""
    marker = await db.get(RuntimeSetting, BOOTSTRAP_IDENTITY_KEY)
    if marker is None:
        return None
    try:
        user_id = uuid.UUID(marker.value)
    except ValueError:
        logger.warning("Ignoring an unreadable %s marker; re-provisioning", BOOTSTRAP_IDENTITY_KEY)
        return None
    return await db.get(User, user_id)


async def _provision(db: AsyncSession) -> User:
    """Create the default organization, workspace, operator identity, and memberships.

    Ordered by the foreign keys: the organization exists before the identity that
    points at it, and the identity exists before it can own anything.
    """
    organizations = OrganizationRepository(db)
    organization = await organizations.get_by_slug(DEFAULT_ORGANIZATION_SLUG)
    if organization is None:
        organization = await organizations.create_organization(
            name=DEFAULT_ORGANIZATION_NAME,
            slug=DEFAULT_ORGANIZATION_SLUG,
            created_by_user_id=None,
        )

    operator = await UserRepository(db).create_local_identity(
        full_name=OPERATOR_FULL_NAME,
        active_organization_id=organization.id,
        is_superuser=True,
    )
    await organizations.update_organization(organization, {"created_by_user_id": operator.id})
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id,
        user_id=operator.id,
        role="owner",
    )

    workspaces = WorkspaceRepository(db)
    workspace = await workspaces.get_by_organization_and_name(organization.id, DEFAULT_WORKSPACE_NAME)
    if workspace is None:
        workspace = await workspaces.create_workspace(
            name=DEFAULT_WORKSPACE_NAME,
            organization_id=organization.id,
            created_by_user_id=operator.id,
        )
    await WorkspaceMemberRepository(db).create(
        workspace_id=workspace.id,
        user_id=operator.id,
        role="owner",
    )

    db.add(RuntimeSetting(key=BOOTSTRAP_IDENTITY_KEY, value=str(operator.id)))
    await db.commit()
    await db.refresh(operator)
    logger.info("Provisioned the default organization and operator identity on first boot")
    return operator


__all__ = [
    "BOOTSTRAP_IDENTITY_KEY",
    "DEFAULT_ORGANIZATION_NAME",
    "DEFAULT_ORGANIZATION_SLUG",
    "DEFAULT_WORKSPACE_NAME",
    "BootstrapIdentityUnavailableError",
    "ensure_bootstrap_identity",
]
