"""Which workspace a request-plane row belongs to.

Every row in ``api_keys``, ``usage_logs``, ``model_aliases`` and
``routing_policies`` carries a workspace, so every writer needs one. There are
only two sources:

- **An API key request.** The workspace comes off the key that authenticated it,
  never off a header. A caller controls its headers and not which key it holds,
  so reading a header here would let anyone bill another workspace.
- **A master-key request.** There is no key row, so it lands in the deployment's
  default workspace: the operator acting deployment-wide, which is what the
  master key means.

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


def reset_key_workspace_cache() -> None:
    """Drop the cached key workspaces (between tests that swap databases)."""
    _key_workspace.clear()


async def default_workspace_id(db: AsyncSession) -> uuid.UUID:
    """The workspace a deployment-wide write lands in.

    Seeded by the migration that added the columns, and adopted by first-boot
    provisioning rather than duplicated, so it exists on every migrated database.
    """
    resolved = (
        await db.execute(
            select(col(Workspace.id))
            .join(Organization, col(Organization.id) == col(Workspace.organization_id))
            .where(col(Organization.slug) == DEFAULT_ORGANIZATION_SLUG, col(Workspace.name) == DEFAULT_WORKSPACE_NAME)
        )
    ).scalar_one_or_none()
    if resolved is None:
        # Fall back to the oldest workspace before creating one: an operator may
        # have renamed the default, and minting a second beside it would be worse
        # than billing to the one that exists. The id breaks a ``created_at``
        # tie, which one transaction creating two workspaces produces, so every
        # process picks the same one rather than each picking its own.
        resolved = (
            await db.execute(
                select(col(Workspace.id)).order_by(col(Workspace.created_at), col(Workspace.id)).limit(1)
            )
        ).scalar_one_or_none()
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


__all__ = [
    "default_workspace_id",
    "reset_key_workspace_cache",
    "resolve_workspace_id",
    "workspace_for_key_id",
]
