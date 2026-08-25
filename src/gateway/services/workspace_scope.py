"""Which workspace a request-plane row belongs to.

Every row in ``api_keys``, ``usage_logs``, ``model_aliases``,
``routing_policies``, ``routing_memory``, ``router_preferences`` and
``file_objects`` carries a workspace, so every writer needs one. There are
only two sources:

- **An API key request.** The workspace comes off the key that authenticated it,
  never off a header. A caller controls its headers and not which key it holds,
  so reading a header here would let anyone bill another workspace.
- **A master-key request.** There is no key row, so it lands in the deployment's
  default workspace: the operator acting deployment-wide, which is what the
  master key means.

This id is not only a billing label. Since otari#643, it also decides *whose*
organization-scoped provider keys satisfy a bare ``provider:model`` selector
(one naming no ``config.providers`` instance) via
``services/provider_kwargs.get_provider_kwargs``: a keyed request resolves
through its own workspace's organization, and a **master-key** request
therefore resolves through the default workspace's organization, not through
every organization the deployment holds. An operator running several
organizations behind one gateway who calls a bare selector with the master
key gets the default workspace's key or none, the same as any other
deployment-wide write this module resolves.

The default is resolved per call and deliberately not memoized. ``RESTRICT``
looks like it makes a cached id safe, since a workspace holding request-plane
rows cannot be deleted, but it does not cover a default holding *nothing*, which
is deletable as soon as a second workspace exists. A process cache would then
hand every deployment-wide write an id naming no row, and one process clearing
its own cache does nothing for the other workers and replicas that also hold it.
The lookup it saves is two indexed reads on the master-key path, which is
operator traffic; keyed requests read the workspace straight off the key.
"""

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.models.entities import APIKey
from gateway.models.tenancy import Organization, Workspace
from gateway.services.tenancy.provisioning_service import (
    DEFAULT_ORGANIZATION_NAME,
    DEFAULT_ORGANIZATION_SLUG,
    DEFAULT_WORKSPACE_NAME,
)

# api_key id -> its workspace. A key belongs to exactly one workspace and never
# moves, so this cannot go stale; it exists because a usage row is written on the
# hot path and must not pay a lookup for something immutable. A deleted key
# leaves its entry behind, which is inert: nothing can authenticate on it again.
_key_workspace: dict[str, uuid.UUID] = {}

# workspace id -> the organization that owns it, for the same reason and with the
# same safety argument as the cache above: the column is written once at creation
# and nothing moves a workspace between organizations. An endpoint that did would
# have to clear this, and pricing would serve the old organization's rates until
# it did.
_workspace_organization: dict[uuid.UUID, uuid.UUID] = {}


def reset_key_workspace_cache() -> None:
    """Drop the cached key workspaces and workspace organizations.

    Called between tests, which empty the database they share. Both caches are
    keyed on ids a test can delete, so leaving either behind would hand the next
    one a workspace or organization that no longer exists.
    """
    _key_workspace.clear()
    _workspace_organization.clear()


async def lookup_default_workspace_id(db: AsyncSession) -> uuid.UUID | None:
    """The default workspace, or ``None`` when the deployment has none yet.

    The read-only half of :func:`default_workspace_id`, for a caller that must
    not write: the alias and policy cache refreshers run on a timer in every
    worker, and creating a workspace as a side effect of reloading a cache would
    turn a read path into a writer racing three replicas.
    """
    resolved = (
        await db.execute(
            select(col(Workspace.id))
            .join(Organization, col(Organization.id) == col(Workspace.organization_id))
            .where(col(Organization.slug) == DEFAULT_ORGANIZATION_SLUG, col(Workspace.name) == DEFAULT_WORKSPACE_NAME)
        )
    ).scalar_one_or_none()
    if resolved is not None:
        return resolved
    # Fall back to the oldest workspace before reporting none: an operator may
    # have renamed the default, and a caller that then minted a second beside it
    # would be worse than using the one that exists. The id breaks a
    # ``created_at`` tie, which one transaction creating two workspaces produces,
    # so every process picks the same one rather than each picking its own.
    return (
        await db.execute(
            select(col(Workspace.id)).order_by(col(Workspace.created_at), col(Workspace.id)).limit(1)
        )
    ).scalar_one_or_none()


async def default_workspace_id(db: AsyncSession) -> uuid.UUID:
    """The workspace a deployment-wide write lands in.

    Seeded by the migration that added the columns, and adopted by first-boot
    provisioning rather than duplicated, so it exists on every migrated database.
    """
    resolved = await lookup_default_workspace_id(db)
    if resolved is None:
        return await _create_default_workspace(db)

    return resolved


async def _create_default_workspace(db: AsyncSession) -> uuid.UUID:
    """Create the default organization and workspace, and return the workspace id.

    The migration seeds these, so a migrated database never reaches here. What
    does is a schema built by ``create_all`` (tests, and anyone bootstrapping
    without Alembic), where a NOT NULL workspace would otherwise make the first
    key unwritable.

    Same slug and name the migration and ``provisioning_service`` use, so
    whichever runs first wins and the others adopt it: three creators, one
    default, which is what keeps an upgraded database from ending up with two.
    No identity is created here; ``created_by_user_id`` is nullable and
    provisioning fills it in when it runs.
    """
    organization = (
        (await db.execute(select(Organization).where(col(Organization.slug) == DEFAULT_ORGANIZATION_SLUG)))
        .scalars()
        .first()
    )
    if organization is None:
        organization = Organization(name=DEFAULT_ORGANIZATION_NAME, slug=DEFAULT_ORGANIZATION_SLUG)
        db.add(organization)
        await db.flush()

    workspace = Workspace(name=DEFAULT_WORKSPACE_NAME, organization_id=organization.id)
    db.add(workspace)
    await db.flush()
    return workspace.id


async def workspace_for_key_id(db: AsyncSession, api_key_id: str | None) -> uuid.UUID:
    """The workspace a usage row belongs to, given the key that authenticated it.

    Usage is written from several paths that carry the key's id rather than the
    key row, so this resolves the one from the other and memoizes it. A master-key
    request has no id and lands in the default workspace; an id that resolves to
    nothing (a key deleted mid-flight) does too, because refusing to record usage
    would lose the spend entirely.
    """
    if api_key_id is None:
        return await default_workspace_id(db)

    cached = _key_workspace.get(api_key_id)
    if cached is not None:
        return cached

    resolved = (await db.execute(select(APIKey.workspace_id).where(APIKey.id == api_key_id))).scalar_one_or_none()
    if resolved is None:
        return await default_workspace_id(db)

    _key_workspace[api_key_id] = resolved
    return resolved


async def resolve_workspace_id(db: AsyncSession, api_key: APIKey | None) -> uuid.UUID:
    """The workspace to stamp on a row written for this request."""
    if api_key is not None:
        return api_key.workspace_id
    return await default_workspace_id(db)


async def organization_for_workspace_id(db: AsyncSession, workspace_id: uuid.UUID) -> uuid.UUID | None:
    """The organization owning a workspace, memoized, or ``None`` if it has none.

    Read by pricing resolution, which needs the organization to look for a rate
    override before falling back to the deployment price list.

    Memoized on the same argument the key cache is: a workspace's organization is
    immutable here. ``Workspace.organization_id`` is set at creation and every
    other reference to it filters or compares; there is no endpoint that moves a
    workspace between organizations, and adding one would have to drop this cache
    (the docstring on the cache below says so, so it is not a silent trap).

    ``None`` rather than an exception for a workspace that resolves to nothing,
    because the caller's next step is to skip the override and price against the
    deployment row. A missing workspace must not fail a request that would
    otherwise be priced and served.
    """
    cached = _workspace_organization.get(workspace_id)
    if cached is not None:
        return cached

    resolved = (
        await db.execute(select(col(Workspace.organization_id)).where(col(Workspace.id) == workspace_id))
    ).scalar_one_or_none()
    if resolved is None:
        return None

    _workspace_organization[workspace_id] = resolved
    return resolved


async def organization_for_key_id(db: AsyncSession, api_key_id: str | None) -> uuid.UUID | None:
    """The organization whose pricing overrides apply to this request.

    The key names the workspace and the workspace names the organization, so this
    is the two existing lookups composed. A master-key request has no key row and
    lands in the default workspace, which is the operator acting deployment-wide,
    so it resolves to that workspace's organization and is priced under the same
    overrides as the rest of the deployment.

    Deliberately never read from a header. The organization decides what a request
    costs, so taking it from something the caller controls would let anyone bill
    at another organization's negotiated rate; it comes off the key, exactly as
    the workspace does.
    """
    workspace_id = await workspace_for_key_id(db, api_key_id)
    return await organization_for_workspace_id(db, workspace_id)


__all__ = [
    "default_workspace_id",
    "lookup_default_workspace_id",
    "organization_for_key_id",
    "organization_for_workspace_id",
    "reset_key_workspace_cache",
    "resolve_workspace_id",
    "workspace_for_key_id",
]
