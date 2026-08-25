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
  address, like the gateway's own spend identities, and the claim flow that gives
  such an identity an address is a separate track.
- It is a superuser. The master key already carries deployment-wide authority,
  so scoping the identity it maps to more narrowly would be theater.

The identity is anchored by a ``runtime_settings`` row rather than by the default
organization's slug: an operator may rename, leave, or delete that organization,
and the anchor has to survive all three.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.log_config import logger
from gateway.models.entities import RuntimeSetting
from gateway.models.tenancy import Organization, User
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.repositories.users_repository import get_or_create_attribution_user
from gateway.services.tenancy.errors import (
    ForeignTenancyError,
    TenancyError,
    WorkspaceBudgetDefaultBudgetNotFoundError,
)

# Stored in runtime_settings, and deliberately not a SETTABLE_KEY, so
# runtime_settings_service ignores it exactly as it ignores the master-key hash
# and the dashboard-session marker.
BOOTSTRAP_IDENTITY_KEY = "tenancy_bootstrap_user_id"

DEFAULT_ORGANIZATION_NAME = "Default organization"
DEFAULT_ORGANIZATION_SLUG = "default"
DEFAULT_WORKSPACE_NAME = "Default workspace"
OPERATOR_FULL_NAME = "Operator"

# How many foreign organizations the refusal names. Enough to recognize the
# database, without reading an unbounded table into a log line.
_FOREIGN_ORGANIZATIONS_NAMED = 5


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
    existing = await load_bootstrap_identity(db)
    if existing is not None:
        return existing

    await _refuse_to_shadow_existing_tenancy(db)

    try:
        return await _provision(db)
    except IntegrityError:
        # Two first requests raced. Whichever lost re-reads the winner's work:
        # the slug and the marker are both unique, so exactly one provisioned.
        await db.rollback()
        logger.info("Concurrent first-boot tenancy provisioning; using the identity that won")
        resolved = await load_bootstrap_identity(db)
        if resolved is None:
            raise BootstrapIdentityUnavailableError from None
        return resolved


async def _refuse_to_shadow_existing_tenancy(db: AsyncSession) -> None:
    """Refuse to provision beside organizations this deployment did not create.

    Provisioning adopts an organization whose slug is ``default``, which is the
    one it would have made itself. Anything else it would quietly ignore: it
    would create its own organization, point the marker at that, and every route
    is scoped to the marked identity's organization, so the rows already in the
    database become unreachable through the API. ``POST /v1/organizations/me/switch``
    is no way back either: it refuses an organization the caller holds no active
    membership in, and a freshly provisioned operator holds none in an imported
    one. The marker is deliberately not a settable key, so the only way back is
    editing ``runtime_settings`` by hand.

    That is the state a restored or imported tenancy arrives in, because the
    platform's slugs are ``{name}-{prefix}`` and never the literal ``default``.
    Failing here turns silent data loss into an error naming the fix.

    It also fires for a deployment that created organizations of its own
    (``POST /v1/organizations``) and then lost its marker, since those slugs are
    not ``default`` either. The refusal is still the right answer there: what it
    reports is that the deployment can no longer say which organization it
    serves, and the fix below is the same one.

    It catches one of the two orderings. The check runs only while the marker is
    unresolved, so it covers importing into a database this gateway has never
    served a tenancy request against. Importing *after* it has provisioned its
    own tenancy is not caught: the marker already resolves, so this never runs
    again, and the imported organizations are silently unreachable exactly as
    described above. The fix in both cases is two rows, not one: the marker names
    the identity, and that identity's ``active_organization_id`` names the
    organization served. An imported identity belongs to several organizations
    and its pointer holds whichever it was last active in, so moving the marker
    alone adopts whatever that happens to be. See "Adopting an existing tenancy"
    in ``docs/access-control.md``.

    Not a startup error either, despite reading like one: the only caller is
    ``ensure_bootstrap_identity``, which runs from the request dependency, so
    this surfaces as a failed tenancy request rather than a refusal to boot.
    """
    # Filtered and bounded in the query rather than in Python: this runs on every
    # request until the marker resolves, only the non-default slugs decide the
    # answer, and the message needs a handful of them rather than all of them.
    unadoptable = list(
        (
            await db.execute(
                select(Organization)
                .where(col(Organization.slug) != DEFAULT_ORGANIZATION_SLUG)
                .order_by(col(Organization.created_at), col(Organization.id))
                .limit(_FOREIGN_ORGANIZATIONS_NAMED + 1)
            )
        )
        .scalars()
        .all()
    )
    if not unadoptable:
        return

    named = sorted(f"{one.name!r} ({one.slug})" for one in unadoptable[:_FOREIGN_ORGANIZATIONS_NAMED])
    if len(unadoptable) > _FOREIGN_ORGANIZATIONS_NAMED:
        named.append("and others")
    names = ", ".join(named)
    raise ForeignTenancyError(
        f"This database already holds organizations this gateway did not provision: {names}. "
        f"Provisioning beside them would make them unreachable, because every route is scoped to "
        f"the organization the operator identity is pointed at. Point the {BOOTSTRAP_IDENTITY_KEY} "
        f"marker at an identity in the organization you mean to serve, and point that identity's "
        f"active_organization_id at it (both are needed; see docs/access-control.md), or start "
        f"from an empty database."
    )


async def load_bootstrap_identity(db: AsyncSession) -> User | None:
    """Resolve the identity the marker names, or None if there is not one yet.

    The read half of ``ensure_bootstrap_identity``, and public because the
    sign-in policy needs it too: ``user_service.operator_has_password`` asks
    what *this* identity holds rather than what any identity holds, and it must
    be able to ask that without provisioning anything.

    A marker that does not resolve (an unreadable value, or one naming a row
    that is gone) is reported as "no identity yet", which is what makes
    ``ensure_bootstrap_identity`` provision one and what makes the deployment
    read as unclaimed.
    """
    marker = await db.get(RuntimeSetting, BOOTSTRAP_IDENTITY_KEY)
    if marker is None:
        return None
    try:
        user_id = uuid.UUID(marker.value)
    except ValueError:
        logger.warning(
            "Ignoring an unreadable %s marker; treating this deployment as unprovisioned",
            BOOTSTRAP_IDENTITY_KEY,
        )
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
    # The operator's request-plane owner, so the first-boot identity can hold a
    # key like any other member. Aliased by name rather than address because this
    # identity deliberately has none.
    await get_or_create_attribution_user(db, user_id=str(operator.id), alias=OPERATOR_FULL_NAME)

    workspaces = WorkspaceRepository(db)
    workspace = await workspaces.get_by_organization_and_name(organization.id, DEFAULT_WORKSPACE_NAME)
    if workspace is None:
        workspace = await workspaces.create_workspace(
            name=DEFAULT_WORKSPACE_NAME,
            organization_id=organization.id,
            created_by_user_id=operator.id,
        )
    # Serialized against a concurrent ``create_default`` on this workspace, via
    # the same lock ``WorkspaceService.add_member`` takes and for the reason
    # ``WorkspaceRepository.lock`` gives: this path reads the workspace's
    # defaults before materializing, that one reads its members, and without a
    # shared lock both can read before either commits, leaving the operator
    # with no ceiling. ``create_workspace`` is the documented exception because
    # a workspace it just made cannot have a default yet; this path is not, since
    # it adopts an existing one. Reachable despite the marker being unresolved:
    # ``get_current_identity`` returns a dashboard session's identity without
    # consulting the marker at all, so a signed-in operator can be creating a
    # default while a request with no session is provisioning here.
    await workspaces.lock(workspace.id)
    member = await WorkspaceMemberRepository(db).create(
        workspace_id=workspace.id,
        user_id=operator.id,
        role="owner",
    )
    # The fourth path that creates a ``WorkspaceMember``, and it materializes
    # the workspace's budget defaults like the other three, so
    # ``WorkspaceService.create_workspace``'s claim that every one of them does
    # is true. A no-op on a genuine first boot, where a default cannot exist yet:
    # creating one needs an identity, and there is none until this returns. It
    # binds when the marker is unresolved on a database that has already run,
    # which is the identity it names having been deleted (``_load_marked_identity``
    # reports a marker whose user is gone as no marker) or the row cleared by
    # hand. The workspace is adopted in that case, and the operator would join a
    # workspace whose other members are all capped as the one that is not.
    #
    # Imported inside the function, matching
    # ``OrganizationService._apply_workspace_assignments``. There the deferral is
    # load-bearing, since a module-level import genuinely closes a cycle; here it
    # is not, and it stays deferred only to keep this module free of an
    # import-time dependency on the half of the package that imports it back.
    # ``tests/unit/test_service_module_imports.py`` pins the graph either way.
    #
    # Flush-only, so it lands in the commit below and a lost race rolls it back
    # with everything else.
    from gateway.services.tenancy.workspace_budget_default_service import (
        WorkspaceBudgetDefaultService,
    )

    try:
        await WorkspaceBudgetDefaultService(db).materialize_for_member(member)
    except WorkspaceBudgetDefaultBudgetNotFoundError as exc:
        # A stored default naming a budget that is gone, which is only reachable
        # on a database whose ``RESTRICT`` foreign key was not enforced. Logged
        # and skipped rather than raised, because raising here is unrecoverable:
        # the marker below would never be written, every later request would
        # re-enter this function and fail identically, and deleting the offending
        # default needs an authorized identity that no longer exists. An operator
        # who joins uncapped is what happened before this call existed, and the
        # dashboard can still fix it.
        logger.warning("Skipping budget-default materialization for the operator identity: %s", exc)

    # Upsert rather than insert: a marker whose value no longer resolves (an
    # unreadable id, or one naming a row that is gone) is what
    # ``load_bootstrap_identity`` reports as "no identity yet", and it says it will
    # re-provision. Adding a second row with the same primary key would instead
    # raise, be swallowed as a lost race, and leave every later request answering
    # 500 with nothing able to clear it.
    marker = await db.get(RuntimeSetting, BOOTSTRAP_IDENTITY_KEY)
    if marker is None:
        db.add(RuntimeSetting(key=BOOTSTRAP_IDENTITY_KEY, value=str(operator.id)))
    else:
        marker.value = str(operator.id)
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
    "load_bootstrap_identity",
]
