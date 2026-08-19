"""The tenancy repositories, against a real PostgreSQL schema.

Two things are worth pinning here rather than in a unit test with a fake
session: the transaction contract (repositories stage, services commit), which
only means anything against a real transaction, and the constraints the
reconciled schema relies on to keep tenants apart, which only exist in the
database.
"""

import uuid
from datetime import datetime

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.models.tenancy import (
    Organization,
    OrganizationCreate,
    OrganizationUpdate,
    User,
    WorkspaceMemberUpdate,
)
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)

pytestmark = pytest.mark.asyncio


async def _organization(db: AsyncSession, *, name: str = "Acme", slug: str | None = None) -> Organization:
    return await OrganizationRepository(db).create_organization(
        name=name,
        slug=slug or f"acme-{uuid.uuid4().hex[:8]}",
        created_by_user_id=None,
    )


async def _identity(
    db: AsyncSession,
    organization: Organization,
    *,
    full_name: str | None = None,
    email: str | None = None,
) -> User:
    repository = UserRepository(db)
    if email is None:
        return await repository.create_local_identity(
            full_name=full_name,
            active_organization_id=organization.id,
        )
    user = User(email=email, full_name=full_name, active_organization_id=organization.id)
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user


# =============================================================================
# The generic base: what every rehomed repository inherits
# =============================================================================


async def test_create_returns_generated_values(async_db: AsyncSession) -> None:
    repository = OrganizationRepository(async_db)

    organization = await repository.create(OrganizationCreate(name="Acme", slug="acme"))

    assert isinstance(organization.id, uuid.UUID)
    assert isinstance(organization.created_at, datetime)
    # Timezone-aware on the way out, which is the point of converting the
    # platform's naive timestamp columns on arrival.
    assert organization.created_at.tzinfo is not None


async def test_writes_are_staged_and_not_committed(async_db: AsyncSession) -> None:
    """The contract the whole slice depends on: repositories flush, services commit."""
    repository = OrganizationRepository(async_db)

    await repository.create(OrganizationCreate(name="Acme", slug="acme"))
    assert await repository.count() == 1

    await async_db.rollback()

    assert await repository.count() == 0


async def test_update_only_applies_the_fields_the_caller_set(async_db: AsyncSession) -> None:
    """A partial update must not reset the columns it does not mention."""
    repository = OrganizationRepository(async_db)
    organization = await repository.create(OrganizationCreate(name="Acme", slug="acme"))

    updated = await repository.update(organization, OrganizationUpdate(name="Acme Inc"))

    assert updated.name == "Acme Inc"
    assert updated.slug == "acme"


async def test_get_and_delete_round_trip(async_db: AsyncSession) -> None:
    repository = OrganizationRepository(async_db)
    organization = await repository.create(OrganizationCreate(name="Acme", slug="acme"))

    assert await repository.get(organization.id) is not None

    await repository.delete(organization)

    assert await repository.get(organization.id) is None
    assert await repository.count() == 0


async def test_get_all_pages(async_db: AsyncSession) -> None:
    repository = OrganizationRepository(async_db)
    for index in range(3):
        await repository.create(OrganizationCreate(name=f"Org {index}", slug=f"org-{index}"))

    assert len(await repository.get_all(limit=2)) == 2
    assert len(await repository.get_all(skip=2)) == 1


async def test_get_all_pages_in_a_defined_order(async_db: AsyncSession) -> None:
    """Paging needs an ORDER BY, and every tenancy repository inherits this one.

    ``OFFSET``/``LIMIT`` over an unordered query is undefined: two pages of an
    unchanged table may repeat a row and skip another, and the counts stay right
    either way. Asserting the partition alone would not catch a missing
    ``ORDER BY``, because undefined is not the same as wrong and a freshly
    filled table usually reads back in insertion order regardless. So the order
    itself is what is asserted. Primary keys here are ``uuid4``, which sorts
    nothing like insertion order, so this is red the moment the clause goes.
    """
    repository = OrganizationRepository(async_db)
    for index in range(7):
        await repository.create(OrganizationCreate(name=f"Org {index}", slug=f"org-{index}"))

    listed = await repository.get_all(limit=1000)

    assert [org.id for org in listed] == sorted(org.id for org in listed)

    # And the partition that ordering buys: paged in twos, every row once.
    paged = [
        org.slug for offset in range(0, len(listed), 2) for org in await repository.get_all(skip=offset, limit=2)
    ]
    assert len(paged) == len(listed)
    assert len(set(paged)) == len(listed), "a page repeated a row, so another was skipped"


# =============================================================================
# Organizations
# =============================================================================


async def test_slug_is_unique(async_db: AsyncSession) -> None:
    """Slugs address an organization, so a duplicate has to be refused."""
    await _organization(async_db, slug="acme")

    with pytest.raises(IntegrityError):
        await _organization(async_db, slug="acme")


async def test_a_local_identity_anchors_its_default_organization(async_db: AsyncSession) -> None:
    """The column is written by this edition and read by the other one.

    Nothing here reads ``default_organization_id``, so a NULL would look
    harmless: the hosted edition resolves an identity's offered-credit owner
    through it and treats "no anchor" as "nobody", which is a silent forfeit
    rather than an error. It is stamped with the active organization, matching
    what the platform writes at every creation site.
    """
    organization = await _organization(async_db)

    identity = await _identity(async_db, organization)

    assert identity.default_organization_id == organization.id
    assert identity.active_organization_id == organization.id


async def test_get_by_slug(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme")

    found = await OrganizationRepository(async_db).get_by_slug("acme")

    assert found is not None
    assert found.id == organization.id
    assert await OrganizationRepository(async_db).get_by_slug("nope") is None


# =============================================================================
# Organization membership
# =============================================================================


async def test_membership_is_unique_per_user_and_organization(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    user = await _identity(async_db, organization)
    repository = OrganizationMemberRepository(async_db)
    await repository.create_membership(organization_id=organization.id, user_id=user.id, role="owner")

    with pytest.raises(IntegrityError):
        await repository.create_membership(organization_id=organization.id, user_id=user.id, role="member")


async def test_active_membership_lookup_ignores_suspended_rows(async_db: AsyncSession) -> None:
    """Removal suspends a membership, and a suspended member is not a member."""
    organization = await _organization(async_db)
    user = await _identity(async_db, organization)
    repository = OrganizationMemberRepository(async_db)
    membership = await repository.create_membership(
        organization_id=organization.id,
        user_id=user.id,
        role="member",
        status="suspended",
    )

    assert await repository.get_by_organization_and_user(organization.id, user.id) is not None
    assert await repository.get_active_by_organization_and_user(organization.id, user.id) is None

    await repository.update_membership(membership, {"status": "active"})

    assert await repository.get_active_by_organization_and_user(organization.id, user.id) is not None


async def test_member_lookup_by_id_is_scoped_to_its_organization(async_db: AsyncSession) -> None:
    """The cross-tenant boundary: another organization's member id resolves to nothing."""
    organization = await _organization(async_db, slug="acme")
    other_organization = await _organization(async_db, slug="globex")
    user = await _identity(async_db, organization)
    repository = OrganizationMemberRepository(async_db)
    membership = await repository.create_membership(
        organization_id=organization.id,
        user_id=user.id,
        role="owner",
    )

    assert await repository.get_by_id_and_organization(membership.id, organization.id) is not None
    assert await repository.get_by_id_and_organization(membership.id, other_organization.id) is None


async def test_active_owner_is_the_earliest_one(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    first = await _identity(async_db, organization, full_name="First")
    second = await _identity(async_db, organization, full_name="Second")
    repository = OrganizationMemberRepository(async_db)
    first_membership = await repository.create_membership(
        organization_id=organization.id,
        user_id=first.id,
        role="owner",
    )
    await repository.create_membership(organization_id=organization.id, user_id=second.id, role="owner")
    await repository.create_membership(
        organization_id=organization.id,
        user_id=(await _identity(async_db, organization, full_name="Third")).id,
        role="member",
    )

    owner = await repository.get_active_owner(organization.id)

    assert owner is not None
    assert owner.id == first_membership.id
    assert await repository.count_active_owners(organization.id) == 2


async def test_roster_joins_identities_and_hides_suspended_members(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    named = await _identity(async_db, organization, full_name="Zoe Zhang", email="zoe@example.com")
    unnamed = await _identity(async_db, organization, email="adam@example.com")
    local = await _identity(async_db, organization)
    removed = await _identity(async_db, organization, full_name="Removed", email="removed@example.com")
    repository = OrganizationMemberRepository(async_db)
    await repository.create_membership(organization_id=organization.id, user_id=named.id, role="owner")
    await repository.create_membership(organization_id=organization.id, user_id=unnamed.id, role="member")
    await repository.create_membership(organization_id=organization.id, user_id=local.id, role="member")
    await repository.create_membership(
        organization_id=organization.id,
        user_id=removed.id,
        role="member",
        status="suspended",
    )

    rows, count = await repository.get_by_organization_with_users(organization.id)

    assert count == 3
    # Ordered by display name, which falls back to the email address when an
    # identity has no full name: "adam@example.com" sorts before "Zoe Zhang".
    # A local identity has neither, so ``nulls_last`` puts it at the end on
    # both engines rather than at the head on SQLite.
    assert [user.id for _, user in rows] == [unnamed.id, named.id, local.id]


async def test_memberships_join_their_organizations(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, name="Acme", slug="acme")
    invited_to = await _organization(async_db, name="Globex", slug="globex")
    user = await _identity(async_db, organization)
    repository = OrganizationMemberRepository(async_db)
    await repository.create_membership(organization_id=organization.id, user_id=user.id, role="owner")
    await repository.create_membership(
        organization_id=invited_to.id,
        user_id=user.id,
        role="member",
        status="invited",
    )

    all_rows = await repository.get_by_user_with_organizations(user.id)
    invited_rows = await repository.get_by_user_with_organizations(user.id, status="invited")

    assert {organization_row.name for _, organization_row in all_rows} == {"Acme", "Globex"}
    assert [organization_row.name for _, organization_row in invited_rows] == ["Globex"]


# =============================================================================
# Workspaces
# =============================================================================


async def test_workspace_name_is_unique_within_its_organization(async_db: AsyncSession) -> None:
    """The same workspace name in two organizations is two different workspaces."""
    organization = await _organization(async_db, slug="acme")
    other_organization = await _organization(async_db, slug="globex")
    user = await _identity(async_db, organization)
    repository = WorkspaceRepository(async_db)
    await repository.create_workspace(
        name="Default workspace",
        organization_id=organization.id,
        created_by_user_id=user.id,
    )
    await repository.create_workspace(
        name="Default workspace",
        organization_id=other_organization.id,
        created_by_user_id=user.id,
    )

    with pytest.raises(IntegrityError):
        await repository.create_workspace(
            name="Default workspace",
            organization_id=organization.id,
            created_by_user_id=user.id,
        )


async def test_workspaces_list_newest_first_with_a_total(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    repository = WorkspaceRepository(async_db)
    for name in ("first", "second", "third"):
        await repository.create_workspace(name=name, organization_id=organization.id, created_by_user_id=None)

    page, count = await repository.get_by_organization(organization.id, limit=2)

    assert count == 3
    assert len(page) == 2


async def test_deleting_a_workspace_cascades_to_its_members(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    user = await _identity(async_db, organization)
    workspaces = WorkspaceRepository(async_db)
    members = WorkspaceMemberRepository(async_db)
    workspace = await workspaces.create_workspace(
        name="Default workspace",
        organization_id=organization.id,
        created_by_user_id=user.id,
    )
    await members.create(workspace_id=workspace.id, user_id=user.id, role="owner")

    await workspaces.delete_workspace(workspace)

    assert await members.get_by_workspace_and_user(workspace.id, user.id) is None


async def test_workspace_member_role_updates_and_removal(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    user = await _identity(async_db, organization)
    workspace = await WorkspaceRepository(async_db).create_workspace(
        name="Default workspace",
        organization_id=organization.id,
        created_by_user_id=user.id,
    )
    repository = WorkspaceMemberRepository(async_db)
    member = await repository.create(workspace_id=workspace.id, user_id=user.id, role="member")

    updated = await repository.update(member, WorkspaceMemberUpdate(role="admin"))

    assert updated.role == "admin"
    assert updated.status == "active"

    await repository.delete(updated)

    assert await repository.get_by_workspace_and_user(workspace.id, user.id) is None


async def test_a_users_workspaces_are_scoped_to_one_organization(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme")
    other_organization = await _organization(async_db, slug="globex")
    user = await _identity(async_db, organization)
    workspaces = WorkspaceRepository(async_db)
    members = WorkspaceMemberRepository(async_db)
    mine = await workspaces.create_workspace(
        name="Default workspace",
        organization_id=organization.id,
        created_by_user_id=user.id,
    )
    theirs = await workspaces.create_workspace(
        name="Default workspace",
        organization_id=other_organization.id,
        created_by_user_id=user.id,
    )
    await members.create(workspace_id=mine.id, user_id=user.id, role="owner")
    await members.create(workspace_id=theirs.id, user_id=user.id, role="member")

    rows, count = await members.get_workspaces_for_user(
        user_id=user.id,
        organization_id=organization.id,
    )

    assert count == 1
    assert [row.workspace_id for row in rows] == [mine.id]


async def test_workspace_members_are_listed_in_directory_order(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    workspace = await WorkspaceRepository(async_db).create_workspace(
        name="Default workspace",
        organization_id=organization.id,
        created_by_user_id=None,
    )
    zoe = await _identity(async_db, organization, full_name="Zoe Zhang", email="zoe@example.com")
    adam = await _identity(async_db, organization, full_name="Adam Ant", email="adam@example.com")
    repository = WorkspaceMemberRepository(async_db)
    await repository.create(workspace_id=workspace.id, user_id=zoe.id, role="owner")
    await repository.create(workspace_id=workspace.id, user_id=adam.id, role="member")

    rows, count = await repository.get_by_workspace(workspace.id)

    assert count == 2
    assert [row.user_id for row in rows] == [adam.id, zoe.id]


# =============================================================================
# Identities
# =============================================================================


async def test_local_identities_have_no_email_and_still_coexist(async_db: AsyncSession) -> None:
    """A unique index tolerates repeated NULLs, which is what makes the M4 backfill possible."""
    organization = await _organization(async_db)

    first = await _identity(async_db, organization, full_name="Operator")
    second = await _identity(async_db, organization, full_name="Another operator")

    assert first.email is None
    assert second.email is None
    assert first.id != second.id


async def test_email_addresses_are_unique(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    await _identity(async_db, organization, email="operator@example.com")

    with pytest.raises(IntegrityError):
        await _identity(async_db, organization, email="operator@example.com")


async def test_active_organization_can_be_repointed(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme")
    other_organization = await _organization(async_db, slug="globex")
    repository = UserRepository(async_db)
    user = await _identity(async_db, organization)

    updated = await repository.set_active_organization(user, other_organization.id)

    assert updated.active_organization_id == other_organization.id


async def test_identity_lookup_by_email(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    user = await _identity(async_db, organization, email="operator@example.com")
    repository = UserRepository(async_db)

    assert (await repository.get_by_email("operator@example.com")) is not None
    assert (await repository.get_by_email("operator@example.com")).id == user.id  # type: ignore[union-attr]
    assert await repository.get_by_email("nobody@example.com") is None
