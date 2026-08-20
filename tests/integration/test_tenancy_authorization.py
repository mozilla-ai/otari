"""The tenancy authorization rules, exercised at the service layer.

The API tests can only ever act as the one operator identity a standalone
deployment has, and that identity is an owner and a superuser, so the rules that
matter most (what a *non*-owner may do) are unreachable through the routes until
per-identity sign-in lands. They are reachable here, by calling the services with
identities built at whatever role the case needs, which is also the layer that
owns the rules.
"""

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.models.tenancy import (
    ActiveOrganizationMemberCreateRequest,
    ActiveOrganizationMemberUpdateRequest,
    InviteOrganizationMemberRequest,
    Organization,
    OrganizationMember,
    User,
    Workspace,
    WorkspaceCreate,
    WorkspaceMemberUpdate,
    WorkspaceUpdate,
)
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.services.tenancy import OrganizationService, WorkspaceService
from gateway.services.tenancy.errors import (
    InvitationAlreadyPendingError,
    MembershipUpdateError,
    NotAuthorizedError,
    OrganizationNameRequiredError,
    OrganizationNotFoundError,
    WorkspaceNotFoundError,
)

_TEST_CONFIG = GatewayConfig()

pytestmark = pytest.mark.asyncio


async def _organization(db: AsyncSession, *, slug: str = "acme") -> Organization:
    return await OrganizationRepository(db).create_organization(
        name=slug.title(),
        slug=slug,
        created_by_user_id=None,
    )


async def _member(
    db: AsyncSession,
    organization: Organization,
    *,
    role: str,
    full_name: str,
    is_superuser: bool = False,
) -> User:
    user = await UserRepository(db).create_local_identity(
        full_name=full_name,
        active_organization_id=organization.id,
        is_superuser=is_superuser,
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id,
        user_id=user.id,
        role=role,
    )
    return user


async def _workspace(
    db: AsyncSession,
    organization: Organization,
    *,
    name: str,
    owner: User | None = None,
) -> Workspace:
    workspace = await WorkspaceRepository(db).create_workspace(
        name=name,
        organization_id=organization.id,
        created_by_user_id=owner.id if owner else None,
    )
    if owner is not None:
        await WorkspaceMemberRepository(db).create(
            workspace_id=workspace.id,
            user_id=owner.id,
            role="owner",
        )
    return workspace


# =============================================================================
# Which organization a caller is served
# =============================================================================


async def test_the_served_organization_follows_the_identity_not_its_memberships(
    async_db: AsyncSession,
) -> None:
    """Two rows decide the adopted tenant, and the docs have to say both.

    `docs/access-control.md` tells an operator to adopt an imported tenancy by
    pointing the bootstrap marker at an owner of the organization they want. The
    marker only names the *identity*; the organization served is that identity's
    ``active_organization_id``. An imported platform identity belongs to several
    organizations and that pointer holds whichever it was last active in, so a
    recipe that moved the marker alone adopted the wrong tenant and reported the
    operator's role there rather than their ownership in the one they meant.
    """
    adopted = await _organization(async_db, slug="acme-1a2b3c4d")
    elsewhere = await _organization(async_db, slug="other-9z8y7x6w")

    # Owner of the one to adopt, but currently pointed at the other, which is the
    # shape an import produces.
    operator = await UserRepository(async_db).create_local_identity(
        full_name="Ada",
        active_organization_id=elsewhere.id,
    )
    await OrganizationMemberRepository(async_db).create_membership(
        organization_id=adopted.id, user_id=operator.id, role="owner"
    )
    await OrganizationMemberRepository(async_db).create_membership(
        organization_id=elsewhere.id, user_id=operator.id, role="member"
    )
    service = OrganizationService(async_db)

    served = await service.get_active_organization_for_user(operator)
    assert served.slug == elsewhere.slug, "ownership elsewhere does not move the pointer"

    # Step 2 of the documented recipe, and the one that was missing.
    await UserRepository(async_db).set_active_organization(operator, adopted.id)

    served = await service.get_active_organization_for_user(operator)
    assert served.slug == adopted.slug
    membership = await service.members.get_active_by_organization_and_user(adopted.id, operator.id)
    assert membership is not None
    assert membership.role == "owner", "and the operator manages it, rather than only reading it"


# =============================================================================
# Organization-level rules
# =============================================================================


async def test_a_plain_member_cannot_rename_the_organization(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    member = await _member(async_db, organization, role="member", full_name="Member")

    with pytest.raises(NotAuthorizedError):
        await OrganizationService(async_db).update_active_organization_for_user(
            user=member,
            organization_name="Renamed",
        )


async def test_a_viewer_cannot_remove_members(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    viewer = await _member(async_db, organization, role="viewer", full_name="Viewer")
    service = OrganizationService(async_db)
    owner_membership = await service.members.get_active_by_organization_and_user(organization.id, owner.id)
    assert owner_membership is not None

    with pytest.raises(NotAuthorizedError):
        await service.remove_active_organization_member_for_user(
            user=viewer,
            organization_member_id=owner_membership.id,
        )


async def test_an_admin_cannot_modify_an_owner(async_db: AsyncSession) -> None:
    """Only an owner outranks an owner, whatever the admin's other powers."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    second_owner = await _member(async_db, organization, role="owner", full_name="Second owner")
    admin = await _member(async_db, organization, role="admin", full_name="Admin")
    service = OrganizationService(async_db)
    target = await service.members.get_active_by_organization_and_user(organization.id, owner.id)
    assert target is not None
    # A second owner exists, so the last-owner guard is not what refuses this.
    assert await service.members.count_active_owners(organization.id) == 2
    assert second_owner.id != owner.id

    with pytest.raises(MembershipUpdateError, match="owners"):
        await service.update_active_organization_member_for_user(
            user=admin,
            organization_member_id=target.id,
            request=ActiveOrganizationMemberUpdateRequest(role="member"),
        )


async def test_an_admin_cannot_promote_themselves_to_owner(async_db: AsyncSession) -> None:
    """The escalation the rank rule reads past.

    "Only an owner outranks an owner" is checked against the target's *current*
    role, so an admin acting on their own non-owner membership clears it and can
    write `role="owner"`. That buys them a rank the same rule then protects from
    every other admin. Deliberately narrower than the platform, which permits it.
    """
    organization = await _organization(async_db)
    await _member(async_db, organization, role="owner", full_name="Owner")
    admin = await _member(async_db, organization, role="admin", full_name="Admin")
    service = OrganizationService(async_db)
    own = await service.members.get_active_by_organization_and_user(organization.id, admin.id)
    assert own is not None

    with pytest.raises(MembershipUpdateError, match="grant the owner role"):
        await service.update_active_organization_member_for_user(
            user=admin,
            organization_member_id=own.id,
            request=ActiveOrganizationMemberUpdateRequest(role="owner"),
        )


async def test_an_admin_cannot_promote_another_member_to_owner(async_db: AsyncSession) -> None:
    """The same rule, reached through someone else's membership rather than their own."""
    organization = await _organization(async_db)
    await _member(async_db, organization, role="owner", full_name="Owner")
    admin = await _member(async_db, organization, role="admin", full_name="Admin")
    member = await _member(async_db, organization, role="member", full_name="Member")
    service = OrganizationService(async_db)
    target = await service.members.get_active_by_organization_and_user(organization.id, member.id)
    assert target is not None

    with pytest.raises(MembershipUpdateError, match="grant the owner role"):
        await service.update_active_organization_member_for_user(
            user=admin,
            organization_member_id=target.id,
            request=ActiveOrganizationMemberUpdateRequest(role="owner"),
        )


async def test_an_admin_cannot_add_a_new_owner(async_db: AsyncSession) -> None:
    """Adding is the third way to write a role, and it takes the same rule.

    Without it the update guard is decorative: an admin who cannot promote
    anyone can still invite an owner, or their own second address as one.
    """
    organization = await _organization(async_db)
    await _member(async_db, organization, role="owner", full_name="Owner")
    admin = await _member(async_db, organization, role="admin", full_name="Admin")

    with pytest.raises(MembershipUpdateError, match="grant the owner role"):
        await OrganizationService(async_db).create_active_organization_member_for_user(
            user=admin,
            request=ActiveOrganizationMemberCreateRequest(email="ada@example.com", role="owner"),
        )


async def test_an_owner_can_promote_someone_to_owner(async_db: AsyncSession) -> None:
    """The narrowing is on the actor's rank, not on the role itself."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    member = await _member(async_db, organization, role="member", full_name="Member")
    service = OrganizationService(async_db)
    target = await service.members.get_active_by_organization_and_user(organization.id, member.id)
    assert target is not None

    updated = await service.update_active_organization_member_for_user(
        user=owner,
        organization_member_id=target.id,
        request=ActiveOrganizationMemberUpdateRequest(role="owner"),
    )

    assert updated.role == "owner"


async def test_a_plain_member_cannot_add_members(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    member = await _member(async_db, organization, role="member", full_name="Member")

    with pytest.raises(NotAuthorizedError):
        await OrganizationService(async_db).create_active_organization_member_for_user(
            user=member,
            request=ActiveOrganizationMemberCreateRequest(email="ada@example.com"),
        )


async def test_an_admin_can_add_members(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    admin = await _member(async_db, organization, role="admin", full_name="Admin")

    result = await OrganizationService(async_db).create_active_organization_member_for_user(
        user=admin,
        request=ActiveOrganizationMemberCreateRequest(email="ada@example.com", role="member"),
    )

    assert result.status == "active"
    assert result.user_id is not None


async def test_an_added_member_lands_in_the_adding_organization_only(async_db: AsyncSession) -> None:
    """A new identity's active organization is the one it was added to."""
    organization = await _organization(async_db, slug="acme")
    elsewhere = await _organization(async_db, slug="globex")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    service = OrganizationService(async_db)

    result = await service.create_active_organization_member_for_user(
        user=owner,
        request=ActiveOrganizationMemberCreateRequest(email="ada@example.com"),
    )

    assert result.user_id is not None
    added = await UserRepository(async_db).get(result.user_id)
    assert added is not None
    assert added.active_organization_id == organization.id
    assert not await service.user_has_active_membership(organization_id=elsewhere.id, user_id=added.id)


async def test_an_owner_can_demote_another_owner(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    other_owner = await _member(async_db, organization, role="owner", full_name="Other owner")
    service = OrganizationService(async_db)
    target = await service.members.get_active_by_organization_and_user(organization.id, other_owner.id)
    assert target is not None

    updated = await service.update_active_organization_member_for_user(
        user=owner,
        organization_member_id=target.id,
        request=ActiveOrganizationMemberUpdateRequest(role="admin"),
    )

    assert updated.role == "admin"


async def test_a_suspended_member_has_no_access_at_all(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    member = await _member(async_db, organization, role="admin", full_name="Suspended")
    members = OrganizationMemberRepository(async_db)
    membership = await members.get_active_by_organization_and_user(organization.id, member.id)
    assert membership is not None
    await members.update_membership(membership, {"status": "suspended"})

    with pytest.raises(NotAuthorizedError):
        await OrganizationService(async_db).get_active_organization_for_user(member)


async def test_membership_in_another_organization_grants_nothing(async_db: AsyncSession) -> None:
    """An owner of one organization is a stranger to the next."""
    organization = await _organization(async_db, slug="acme")
    other_organization = await _organization(async_db, slug="globex")
    outsider = await _member(async_db, other_organization, role="owner", full_name="Outsider")
    await UserRepository(async_db).set_active_organization(outsider, organization.id)

    with pytest.raises(NotAuthorizedError):
        await OrganizationService(async_db).get_active_organization_for_user(outsider)


async def test_a_suspended_membership_falls_back_to_a_live_one(async_db: AsyncSession) -> None:
    """Losing access to the organization you were pointed at must not 404 every page.

    This is the reachable stale-pointer case. A pointer at a *deleted*
    organization is not: ``active_organization_id`` is a real foreign key, which
    is why the delete path repoints every affected identity first.
    """
    suspended_in = await _organization(async_db, slug="suspended-in")
    live = await _organization(async_db, slug="live")
    user = await _member(async_db, suspended_in, role="owner", full_name="Owner")
    members = OrganizationMemberRepository(async_db)
    await members.create_membership(organization_id=live.id, user_id=user.id, role="member")
    membership = await members.get_active_by_organization_and_user(suspended_in.id, user.id)
    assert membership is not None
    await members.update_membership(membership, {"status": "suspended"})

    context = await OrganizationService(async_db).get_active_membership_context_for_user(user)

    assert context.organization.id == live.id
    assert user.active_organization_id == live.id


async def test_an_identity_with_no_live_membership_has_no_context(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    user = await _member(async_db, organization, role="member", full_name="Member")
    members = OrganizationMemberRepository(async_db)
    membership = await members.get_active_by_organization_and_user(organization.id, user.id)
    assert membership is not None
    await members.update_membership(membership, {"status": "suspended"})

    with pytest.raises(OrganizationNotFoundError):
        await OrganizationService(async_db).get_active_membership_context_for_user(user)


# =============================================================================
# Workspace-level rules
# =============================================================================


async def test_a_plain_member_cannot_create_a_workspace(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    member = await _member(async_db, organization, role="member", full_name="Member")

    with pytest.raises(NotAuthorizedError):
        await WorkspaceService(async_db).create_workspace(
            user=member,
            workspace_create=WorkspaceCreate(name="Research"),
        )


async def test_a_member_only_sees_the_workspaces_they_belong_to(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    member = await _member(async_db, organization, role="member", full_name="Member")
    theirs = await _workspace(async_db, organization, name="Theirs", owner=member)
    await _workspace(async_db, organization, name="Not theirs", owner=owner)
    service = WorkspaceService(async_db)

    as_member = await service.list_workspaces(user=member)
    as_owner = await service.list_workspaces(user=owner)

    assert [workspace.id for workspace in as_member.data] == [theirs.id]
    assert as_member.count == 1
    assert as_owner.count == 2


async def test_a_workspace_the_member_is_not_in_is_not_found(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    member = await _member(async_db, organization, role="member", full_name="Member")
    private = await _workspace(async_db, organization, name="Private", owner=owner)

    with pytest.raises(WorkspaceNotFoundError):
        await WorkspaceService(async_db).get_workspace(user=member, workspace_id=private.id)


async def test_a_workspace_owner_can_rename_their_own_workspace(async_db: AsyncSession) -> None:
    """The second half of the two-level model: workspace roles carry management too."""
    organization = await _organization(async_db)
    member = await _member(async_db, organization, role="member", full_name="Member")
    workspace = await _workspace(async_db, organization, name="Theirs", owner=member)

    updated = await WorkspaceService(async_db).update_workspace(
        user=member,
        workspace_id=workspace.id,
        workspace_update=WorkspaceUpdate(name="Renamed"),
    )

    assert updated.name == "Renamed"


async def test_a_plain_workspace_member_cannot_rename_it(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    member = await _member(async_db, organization, role="member", full_name="Member")
    workspace = await _workspace(async_db, organization, name="Shared", owner=owner)
    await WorkspaceMemberRepository(async_db).create(
        workspace_id=workspace.id,
        user_id=member.id,
        role="member",
    )

    with pytest.raises(NotAuthorizedError):
        await WorkspaceService(async_db).update_workspace(
            user=member,
            workspace_id=workspace.id,
            workspace_update=WorkspaceUpdate(name="Renamed"),
        )


async def test_a_workspace_owner_cannot_delete_their_workspace(async_db: AsyncSession) -> None:
    """Deleting is an organization-level action, unlike renaming."""
    organization = await _organization(async_db)
    member = await _member(async_db, organization, role="member", full_name="Member")
    workspace = await _workspace(async_db, organization, name="Theirs", owner=member)

    with pytest.raises(NotAuthorizedError):
        await WorkspaceService(async_db).delete_workspace(user=member, workspace_id=workspace.id)


async def test_a_superuser_sees_every_workspace_without_membership(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    operator = await _member(async_db, organization, role="member", full_name="Operator", is_superuser=True)
    await _workspace(async_db, organization, name="Theirs", owner=owner)
    service = WorkspaceService(async_db)

    listed = await service.list_workspaces(user=operator)

    assert listed.count == 1
    assert await service.get_workspace(user=operator, workspace_id=listed.data[0].id) is not None


async def test_a_workspace_id_from_another_organization_is_not_found(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme")
    other_organization = await _organization(async_db, slug="globex")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    elsewhere = await _workspace(async_db, other_organization, name="Elsewhere")

    with pytest.raises(WorkspaceNotFoundError):
        await WorkspaceService(async_db).get_workspace(user=owner, workspace_id=elsewhere.id)


async def test_an_unknown_workspace_id_is_not_found(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")

    with pytest.raises(WorkspaceNotFoundError):
        await WorkspaceService(async_db).get_workspace(user=owner, workspace_id=uuid.uuid4())


async def test_an_organization_admin_manages_workspaces_they_are_not_in(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    admin = await _member(async_db, organization, role="admin", full_name="Admin")
    workspace = await _workspace(async_db, organization, name="Someone else's", owner=owner)

    updated = await WorkspaceService(async_db).update_workspace(
        user=admin,
        workspace_id=workspace.id,
        workspace_update=WorkspaceUpdate(description="Reviewed"),
    )

    assert updated.description == "Reviewed"
    assert not [
        member
        for member in (await WorkspaceMemberRepository(async_db).get_by_workspace(workspace.id))[0]
        if member.user_id == admin.id
    ]


async def test_the_operator_identity_and_organization_member_rows_are_distinct_concepts(
    async_db: AsyncSession,
) -> None:
    """A workspace member row never implies an organization member row, or the reverse."""
    organization = await _organization(async_db)
    member = await _member(async_db, organization, role="member", full_name="Member")
    workspace = await _workspace(async_db, organization, name="Theirs", owner=member)

    organization_rows = await OrganizationMemberRepository(async_db).get_by_user(member.id)
    workspace_rows = await WorkspaceMemberRepository(async_db).get_by_workspace(workspace.id)

    assert [row.role for row in organization_rows] == ["member"]
    assert [row.role for row in workspace_rows[0]] == ["owner"]
    assert isinstance(organization_rows[0], OrganizationMember)


async def test_a_suspended_workspace_membership_grants_no_visibility(async_db: AsyncSession) -> None:
    """Suspended is not a quieter kind of member: it grants nothing.

    The management check has always read the status; the two visibility paths did
    not, so a member removed from a workspace kept reading it and its roster
    while being refused every action in it.
    """
    organization = await _organization(async_db, slug="acme")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    plain = await _member(async_db, organization, role="member", full_name="Plain")
    workspace = await _workspace(async_db, organization, name="Research", owner=owner)

    members = WorkspaceMemberRepository(async_db)
    membership = await members.create(workspace_id=workspace.id, user_id=plain.id, role="member")
    await members.update(membership, WorkspaceMemberUpdate(status="suspended"))
    await async_db.commit()

    service = WorkspaceService(async_db)
    with pytest.raises(WorkspaceNotFoundError):
        await service.get_workspace(user=plain, workspace_id=workspace.id)
    with pytest.raises(WorkspaceNotFoundError):
        await service.list_members(user=plain, workspace_id=workspace.id)

    listed = await service.list_workspaces(user=plain)
    assert listed.count == 0
    assert listed.data == []


async def test_a_blank_organization_name_is_refused_rather_than_substituted(async_db: AsyncSession) -> None:
    """``min_length=1`` admits a single space, which must not rename the organization.

    The fallback this replaced stored the literal "Organization", so a caller who
    submitted whitespace got a rename they never asked for.
    """
    organization = await _organization(async_db, slug="acme")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    await async_db.commit()

    service = OrganizationService(async_db)
    with pytest.raises(OrganizationNameRequiredError):
        await service.update_active_organization_for_user(user=owner, organization_name="   ")

    refreshed = await OrganizationRepository(async_db).get(organization.id)
    assert refreshed is not None
    assert refreshed.name == "Acme"


async def test_a_superuser_manages_a_workspace_they_are_not_a_member_of(async_db: AsyncSession) -> None:
    """The superuser arm of workspace management, matching the read and org-level checks.

    A superuser already saw every workspace and could delete one through the
    organization-level guard, so refusing them a rename was the odd one out.
    """
    organization = await _organization(async_db, slug="acme")
    outsider = await _member(async_db, organization, role="member", full_name="Root", is_superuser=True)
    workspace = await _workspace(async_db, organization, name="Research")
    await async_db.commit()

    renamed = await WorkspaceService(async_db).update_workspace(
        user=outsider,
        workspace_id=workspace.id,
        workspace_update=WorkspaceUpdate(name="Renamed"),
    )
    assert renamed.name == "Renamed"


# =============================================================================
# Invitations
# =============================================================================


async def test_a_plain_member_cannot_invite(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    member = await _member(async_db, organization, role="member", full_name="Member")

    with pytest.raises(NotAuthorizedError):
        await OrganizationService(async_db).invite_active_organization_member_for_user(
            user=member,
            request=InviteOrganizationMemberRequest(email="ada@example.com"),
            config=_TEST_CONFIG,
        )


async def test_an_admin_can_invite(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    admin = await _member(async_db, organization, role="admin", full_name="Admin")

    result = await OrganizationService(async_db).invite_active_organization_member_for_user(
        user=admin,
        request=InviteOrganizationMemberRequest(email="ada@example.com", role="member"),
        config=_TEST_CONFIG,
    )

    assert result.status == "invited"
    assert result.mail_sent is False  # _TEST_CONFIG has no SMTP configured


async def test_an_admin_cannot_invite_an_owner(async_db: AsyncSession) -> None:
    """The same rule ``create_active_organization_member_for_user`` enforces, on its twin."""
    organization = await _organization(async_db)
    await _member(async_db, organization, role="owner", full_name="Owner")
    admin = await _member(async_db, organization, role="admin", full_name="Admin")

    with pytest.raises(MembershipUpdateError, match="grant the owner role"):
        await OrganizationService(async_db).invite_active_organization_member_for_user(
            user=admin,
            request=InviteOrganizationMemberRequest(email="ada@example.com", role="owner"),
            config=_TEST_CONFIG,
        )


async def test_a_viewer_cannot_revoke_an_invitation(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    admin = await _member(async_db, organization, role="admin", full_name="Admin")
    viewer = await _member(async_db, organization, role="viewer", full_name="Viewer")
    service = OrganizationService(async_db)
    invitation = await service.invite_active_organization_member_for_user(
        user=admin,
        request=InviteOrganizationMemberRequest(email="ada@example.com"),
        config=_TEST_CONFIG,
    )

    with pytest.raises(NotAuthorizedError):
        await service.revoke_organization_member_invitation_for_user(
            user=viewer,
            invitation_id=invitation.invitation_id,
        )


async def test_an_admin_cannot_revoke_a_pending_owner_invitation(async_db: AsyncSession) -> None:
    """Revoke goes through the same "only an owner outranks an owner" guard as every other write."""
    organization = await _organization(async_db)
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    admin = await _member(async_db, organization, role="admin", full_name="Admin")
    service = OrganizationService(async_db)
    invitation = await service.invite_active_organization_member_for_user(
        user=owner,
        request=InviteOrganizationMemberRequest(email="ada@example.com", role="owner"),
        config=_TEST_CONFIG,
    )

    with pytest.raises(MembershipUpdateError, match="owners"):
        await service.revoke_organization_member_invitation_for_user(
            user=admin,
            invitation_id=invitation.invitation_id,
        )


async def test_inviting_an_address_with_a_pending_invitation_conflicts(async_db: AsyncSession) -> None:
    organization = await _organization(async_db)
    admin = await _member(async_db, organization, role="admin", full_name="Admin")
    service = OrganizationService(async_db)
    await service.invite_active_organization_member_for_user(
        user=admin,
        request=InviteOrganizationMemberRequest(email="ada@example.com"),
        config=_TEST_CONFIG,
    )

    with pytest.raises(InvitationAlreadyPendingError):
        await service.invite_active_organization_member_for_user(
            user=admin,
            request=InviteOrganizationMemberRequest(email="ada@example.com"),
            config=_TEST_CONFIG,
        )
