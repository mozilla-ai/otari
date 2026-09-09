"""The invitee's side of the invitation flow: list, accept, decline (otari-ai#1999).

Exercised at the service layer for the reason `test_tenancy_authorization.py`
gives: the API test client can only ever act as the one operator identity a
standalone deployment has, and every case here needs a *second* identity, the
one an invitation is addressed to. `test_invitations_api.py` covers the route
wiring over HTTP; the rules are here.

What the inbox is for is the case the emailed link serves worst: an address
that already holds an identity, invited to a second organization. Losing that
one email used to leave the invitation reachable by nothing.
"""

import hashlib
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.core.config import GatewayConfig
from gateway.models.tenancy import (
    Invitation,
    InviteOrganizationMemberRequest,
    InviteOrganizationMemberResultPublic,
    Organization,
    OrganizationMemberRole,
    User,
    WorkspaceAssignmentRequest,
)
from gateway.repositories.tenancy import (
    InvitationRepository,
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.services.tenancy import OrganizationService
from gateway.services.tenancy.errors import (
    InvitationAlreadyUsedError,
    InvitationExpiredError,
    InvitationNotFoundError,
)

_TEST_CONFIG = GatewayConfig()

pytestmark = pytest.mark.asyncio


async def _organization(db: AsyncSession, *, slug: str) -> Organization:
    return await OrganizationRepository(db).create_organization(
        name=slug.title(),
        slug=slug,
        created_by_user_id=None,
    )


async def _owner(db: AsyncSession, organization: Organization, *, full_name: str) -> User:
    """An active owner of ``organization``, i.e. somebody who may invite."""
    user = await UserRepository(db).create_local_identity(
        full_name=full_name,
        active_organization_id=organization.id,
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=organization.id,
        user_id=user.id,
        role="owner",
    )
    return user


async def _identity_with_a_home(db: AsyncSession, *, email: str) -> tuple[User, Organization]:
    """An identity that already belongs somewhere, which is who the inbox is for.

    An invitation to an address with no identity mints a password-less one that
    cannot sign in until ``POST /v1/auth/signup`` claims it, so it could not
    reach an inbox at all. Every case here starts from an identity that can.
    """
    home = await _organization(db, slug=f"home-{uuid.uuid4().hex[:8]}")
    user = await UserRepository(db).create_local_identity(
        full_name="Invitee",
        active_organization_id=home.id,
        email=email,
    )
    await OrganizationMemberRepository(db).create_membership(
        organization_id=home.id,
        user_id=user.id,
        role="owner",
    )
    return user, home


async def _invite(
    service: OrganizationService,
    inviter: User,
    *,
    email: str,
    role: OrganizationMemberRole = "member",
    workspace_assignments: list[WorkspaceAssignmentRequest] | None = None,
) -> InviteOrganizationMemberResultPublic:
    return await service.invite_active_organization_member_for_user(
        user=inviter,
        request=InviteOrganizationMemberRequest(
            email=email,
            role=role,
            workspace_assignments=workspace_assignments,
        ),
        config=_TEST_CONFIG,
    )


async def test_the_inbox_lists_an_invitation_the_switcher_deliberately_hides(
    async_db: AsyncSession,
) -> None:
    """The gap this closes: `invited` is absent from the memberships list by design.

    `list_organization_memberships_for_user` filters to `active`, because an
    `invited` membership is not somewhere the caller may act yet. That left the
    invitation reachable only through the emailed link.
    """
    invitee, home = await _identity_with_a_home(async_db, email="invitee@example.com")
    inviting = await _organization(async_db, slug="inviting")
    admin = await _owner(async_db, inviting, full_name="Admin")
    service = OrganizationService(async_db)

    issued = await _invite(service, admin, email="invitee@example.com", role="admin")

    memberships = await service.list_organization_memberships_for_user(user=invitee)
    assert {row.organization.slug for row in memberships.data} == {home.slug}

    inbox = await service.list_pending_organization_invitations_for_user(user=invitee)
    assert inbox.count == 1
    (waiting,) = inbox.data
    assert waiting.organization_id == inviting.id
    assert waiting.organization_name == "Inviting"
    assert waiting.role == "admin"
    assert waiting.email == "invitee@example.com"
    assert waiting.organization_member_id == issued.organization_member_id
    assert waiting.invitation_id == issued.invitation_id


async def test_the_inbox_is_scoped_to_the_caller_and_not_to_the_address(
    async_db: AsyncSession,
) -> None:
    """One identity's inbox never carries another's invitation."""
    first, _ = await _identity_with_a_home(async_db, email="first@example.com")
    second, _ = await _identity_with_a_home(async_db, email="second@example.com")
    inviting = await _organization(async_db, slug="inviting")
    admin = await _owner(async_db, inviting, full_name="Admin")
    service = OrganizationService(async_db)

    await _invite(service, admin, email="first@example.com")

    assert (await service.list_pending_organization_invitations_for_user(user=first)).count == 1
    assert (await service.list_pending_organization_invitations_for_user(user=second)).count == 0


async def test_accepting_from_the_inbox_needs_no_token_and_applies_the_parked_assignments(
    async_db: AsyncSession,
) -> None:
    """The whole of what the emailed link does, minus the token.

    The workspace assignments parked at invite time are the part most easily
    lost by a second accept path, so they are what this asserts: an invitee who
    accepts here lands in the same workspaces as one who followed the link.
    """
    invitee, home = await _identity_with_a_home(async_db, email="invitee@example.com")
    inviting = await _organization(async_db, slug="inviting")
    admin = await _owner(async_db, inviting, full_name="Admin")
    workspace = await WorkspaceRepository(async_db).create_workspace(
        name="Shared",
        organization_id=inviting.id,
        created_by_user_id=admin.id,
    )
    service = OrganizationService(async_db)

    issued = await _invite(
        service,
        admin,
        email="invitee@example.com",
        role="member",
        workspace_assignments=[WorkspaceAssignmentRequest(workspace_id=workspace.id, role="admin")],
    )

    result = await service.accept_pending_membership_for_user(
        user=invitee,
        organization_member_id=issued.organization_member_id,
    )
    assert result.organization_name == "Inviting"
    assert result.role == "member"

    memberships = await service.list_organization_memberships_for_user(user=invitee)
    assert {row.organization.slug for row in memberships.data} == {home.slug, "inviting"}

    assignment = await WorkspaceMemberRepository(async_db).get_by_workspace_and_user(workspace.id, invitee.id)
    assert assignment is not None
    assert assignment.role == "admin"

    # Consumed, so the emailed link for the same invitation is spent too.
    assert (await service.list_pending_organization_invitations_for_user(user=invitee)).count == 0
    invitation = await InvitationRepository(async_db).get(issued.invitation_id)
    assert invitation is not None
    assert invitation.status == "accepted"


async def test_declining_cancels_the_invitation_and_kills_the_emailed_link(
    async_db: AsyncSession,
) -> None:
    """Why decline suspends rather than only cancelling.

    The emailed link is a separate credential from the decline. Leaving the
    membership `invited` would leave that link working, so accepting it later
    would flip the membership to `active` and silently undo the decline, which
    is the hazard `_cancel_pending_invitation_for_membership` records for every
    other path that suspends.
    """
    invitee, _ = await _identity_with_a_home(async_db, email="invitee@example.com")
    inviting = await _organization(async_db, slug="inviting")
    admin = await _owner(async_db, inviting, full_name="Admin")
    service = OrganizationService(async_db)

    issued = await _invite(service, admin, email="invitee@example.com")
    token = issued.accept_link.split("token=")[1]

    await service.decline_pending_membership_for_user(
        user=invitee,
        organization_member_id=issued.organization_member_id,
    )

    assert (await service.list_pending_organization_invitations_for_user(user=invitee)).count == 0
    invitation = await InvitationRepository(async_db).get(issued.invitation_id)
    assert invitation is not None
    assert invitation.status == "cancelled"

    membership = await OrganizationMemberRepository(async_db).get(
        issued.organization_member_id,
    )
    assert membership is not None
    assert membership.status == "suspended"

    # The link the invitee was mailed is dead, rather than a way to undo the
    # decline. Refused as already-used rather than not-found: the row is still
    # there and still theirs, it is its `cancelled` status that ends it.
    with pytest.raises(InvitationAlreadyUsedError):
        await service.accept_invitation(token)
    assert inviting.id not in {
        row.organization.id for row in (await service.list_organization_memberships_for_user(user=invitee)).data
    }


async def test_a_declined_address_can_be_invited_again(
    async_db: AsyncSession,
) -> None:
    """Suspend-not-delete is what makes a decline reversible rather than final."""
    invitee, _ = await _identity_with_a_home(async_db, email="invitee@example.com")
    inviting = await _organization(async_db, slug="inviting")
    admin = await _owner(async_db, inviting, full_name="Admin")
    service = OrganizationService(async_db)

    first = await _invite(service, admin, email="invitee@example.com")
    await service.decline_pending_membership_for_user(
        user=invitee,
        organization_member_id=first.organization_member_id,
    )

    second = await _invite(service, admin, email="invitee@example.com", role="viewer")
    # The same membership revived, not a second one.
    assert second.organization_member_id == first.organization_member_id

    inbox = await service.list_pending_organization_invitations_for_user(user=invitee)
    assert inbox.count == 1
    assert inbox.data[0].role == "viewer"
    assert inbox.data[0].invitation_id == second.invitation_id


async def test_declining_an_owner_invitation_is_the_invitees_own_to_do(
    async_db: AsyncSession,
) -> None:
    """No rank guard on this path, unlike revoke.

    `revoke_organization_member_invitation_for_user` runs
    `_validate_membership_update` so an admin cannot suspend a pending *owner*.
    That question is about an actor outranking somebody else; here the actor is
    the target.
    """
    invitee, _ = await _identity_with_a_home(async_db, email="invitee@example.com")
    inviting = await _organization(async_db, slug="inviting")
    admin = await _owner(async_db, inviting, full_name="Admin")
    service = OrganizationService(async_db)

    issued = await _invite(service, admin, email="invitee@example.com", role="owner")
    await service.decline_pending_membership_for_user(
        user=invitee,
        organization_member_id=issued.organization_member_id,
    )

    invitation = await InvitationRepository(async_db).get(issued.invitation_id)
    assert invitation is not None
    assert invitation.status == "cancelled"


@pytest.mark.parametrize("action", ["accept", "decline"])
async def test_another_identitys_invitation_is_not_found_rather_than_forbidden(
    async_db: AsyncSession,
    action: str,
) -> None:
    """Same collapse every cross-tenant lookup makes: a 403 here is an existence oracle."""
    _, _ = await _identity_with_a_home(async_db, email="invitee@example.com")
    outsider, _ = await _identity_with_a_home(async_db, email="outsider@example.com")
    inviting = await _organization(async_db, slug="inviting")
    admin = await _owner(async_db, inviting, full_name="Admin")
    service = OrganizationService(async_db)

    issued = await _invite(service, admin, email="invitee@example.com")

    with pytest.raises(InvitationNotFoundError):
        if action == "accept":
            await service.accept_pending_membership_for_user(
                user=outsider,
                organization_member_id=issued.organization_member_id,
            )
        else:
            await service.decline_pending_membership_for_user(
                user=outsider,
                organization_member_id=issued.organization_member_id,
            )


@pytest.mark.parametrize("action", ["accept", "decline"])
async def test_an_unknown_membership_id_is_not_found(
    async_db: AsyncSession,
    action: str,
) -> None:
    invitee, _ = await _identity_with_a_home(async_db, email="invitee@example.com")
    service = OrganizationService(async_db)

    with pytest.raises(InvitationNotFoundError):
        if action == "accept":
            await service.accept_pending_membership_for_user(user=invitee, organization_member_id=uuid.uuid4())
        else:
            await service.decline_pending_membership_for_user(user=invitee, organization_member_id=uuid.uuid4())


async def test_accepting_twice_answers_the_same_success_rather_than_a_404(
    async_db: AsyncSession,
) -> None:
    """Two clicks before the list refreshes is the ordinary way to reach this.

    Reporting "not found" for an action that succeeded would be worse than
    saying so twice, and it tells the caller nothing they could not already
    see: only their own memberships reach the check.
    """
    invitee, _ = await _identity_with_a_home(async_db, email="invitee@example.com")
    inviting = await _organization(async_db, slug="inviting")
    admin = await _owner(async_db, inviting, full_name="Admin")
    service = OrganizationService(async_db)

    issued = await _invite(service, admin, email="invitee@example.com", role="admin")
    first = await service.accept_pending_membership_for_user(
        user=invitee,
        organization_member_id=issued.organization_member_id,
    )
    second = await service.accept_pending_membership_for_user(
        user=invitee,
        organization_member_id=issued.organization_member_id,
    )

    assert second == first
    assert second.organization_name == "Inviting"
    assert second.role == "admin"
    # The second call is a read: it must not re-run the accept's writes.
    invitation = await InvitationRepository(async_db).get(issued.invitation_id)
    assert invitation is not None
    assert invitation.status == "accepted"


async def test_the_idempotent_branch_is_any_active_membership_and_decline_gets_none_of_it(
    async_db: AsyncSession,
) -> None:
    """Where the two calls deliberately part company.

    Accept is a read once the membership is `active`, so it answers for one the
    caller never had an invitation to (their own ordinary membership) rather
    than paying a lookup to tell that apart. Decline writes, so it stays
    narrow: only a membership still `invited` has anything to decline, and an
    active one collapses into the same 404 as somebody else's.
    """
    invitee, home = await _identity_with_a_home(async_db, email="invitee@example.com")
    service = OrganizationService(async_db)
    own = await OrganizationMemberRepository(async_db).get_by_organization_and_user(home.id, invitee.id)
    assert own is not None

    accepted = await service.accept_pending_membership_for_user(
        user=invitee,
        organization_member_id=own.id,
    )
    assert accepted.organization_name == home.name
    assert accepted.role == "owner"

    with pytest.raises(InvitationNotFoundError):
        await service.decline_pending_membership_for_user(
            user=invitee,
            organization_member_id=own.id,
        )

    # Untouched: the accept above resolved to a read, so it wrote nothing.
    refreshed = await OrganizationMemberRepository(async_db).get(own.id)
    assert refreshed is not None
    assert refreshed.status == "active"


async def test_a_lapsed_invitation_is_omitted_from_the_inbox_and_refused_on_accept(
    async_db: AsyncSession,
) -> None:
    """Expiry is lazy, so the deadline is what the inbox filters on, not the status.

    `_resolve_pending_invitation` only flips a row to `expired` when someone
    presents its token, so an unopened link sits `pending` with its deadline
    long past. Filtering on the stored status alone would list it as if it were
    still actionable.
    """
    invitee, _ = await _identity_with_a_home(async_db, email="invitee@example.com")
    inviting = await _organization(async_db, slug="inviting")
    admin = await _owner(async_db, inviting, full_name="Admin")
    service = OrganizationService(async_db)

    issued = await _invite(service, admin, email="invitee@example.com")
    invitation = await InvitationRepository(async_db).get(issued.invitation_id)
    assert invitation is not None
    assert invitation.status == "pending"
    invitation.expires_at = datetime.now(UTC) - timedelta(hours=1)
    async_db.add(invitation)
    await async_db.commit()

    assert (await service.list_pending_organization_invitations_for_user(user=invitee)).count == 0

    with pytest.raises(InvitationExpiredError):
        await service.accept_pending_membership_for_user(
            user=invitee,
            organization_member_id=issued.organization_member_id,
        )

    # Recorded now that a caller has asked, so the roster stops offering a
    # revoke for a link that is already dead.
    refreshed = await InvitationRepository(async_db).get(issued.invitation_id)
    assert refreshed is not None
    assert refreshed.status == "expired"


async def test_a_revoked_invitation_leaves_nothing_in_the_inbox(
    async_db: AsyncSession,
) -> None:
    """Both halves of the pair are checked, not just the invitation's status."""
    invitee, _ = await _identity_with_a_home(async_db, email="invitee@example.com")
    inviting = await _organization(async_db, slug="inviting")
    admin = await _owner(async_db, inviting, full_name="Admin")
    service = OrganizationService(async_db)

    issued = await _invite(service, admin, email="invitee@example.com")
    await service.revoke_organization_member_invitation_for_user(
        user=admin,
        invitation_id=issued.invitation_id,
    )

    assert (await service.list_pending_organization_invitations_for_user(user=invitee)).count == 0
    with pytest.raises(InvitationNotFoundError):
        await service.accept_pending_membership_for_user(
            user=invitee,
            organization_member_id=issued.organization_member_id,
        )


async def test_the_inbox_pages_over_several_waiting_organizations(
    async_db: AsyncSession,
) -> None:
    """`count` is the total and the page is exact, which needs the filter in the query.

    Dropping lapsed rows after paging would make both lie, and the dashboard's
    `fetchAllPaged` walk stops on a short page, so a filtered-out row would end
    the walk early.
    """
    invitee, _ = await _identity_with_a_home(async_db, email="invitee@example.com")
    service = OrganizationService(async_db)
    for index in range(3):
        organization = await _organization(async_db, slug=f"inviting-{index}")
        admin = await _owner(async_db, organization, full_name=f"Admin {index}")
        await _invite(service, admin, email="invitee@example.com")

    first_page = await service.list_pending_organization_invitations_for_user(user=invitee, limit=2)
    assert first_page.count == 3
    assert len(first_page.data) == 2

    second_page = await service.list_pending_organization_invitations_for_user(user=invitee, skip=2, limit=2)
    assert second_page.count == 3
    assert len(second_page.data) == 1

    seen = [row.organization_name for row in (*first_page.data, *second_page.data)]
    assert sorted(seen) == ["Inviting-0", "Inviting-1", "Inviting-2"]


async def test_declining_cancels_every_live_link_to_the_membership(
    async_db: AsyncSession,
) -> None:
    """A second live invitation would otherwise be a way to undo the decline.

    One pending row per membership is the invite path's invariant and not a
    database constraint, so a decline that cancelled only the row it resolved
    would leave the other one accepting into a membership it had just
    suspended, flipping it back to `active`.
    """
    invitee, _ = await _identity_with_a_home(async_db, email="invitee@example.com")
    inviting = await _organization(async_db, slug="inviting")
    admin = await _owner(async_db, inviting, full_name="Admin")
    service = OrganizationService(async_db)

    issued = await _invite(service, admin, email="invitee@example.com")
    second_token = uuid.uuid4().hex
    second = Invitation(
        organization_id=inviting.id,
        organization_member_id=issued.organization_member_id,
        email="invitee@example.com",
        invited_by_user_id=admin.id,
        token_hash=hashlib.sha256(second_token.encode()).hexdigest(),
        workspace_assignments=[],
        expires_at=datetime.now(UTC) + timedelta(days=7),
    )
    async_db.add(second)
    await async_db.commit()

    await service.decline_pending_membership_for_user(
        user=invitee,
        organization_member_id=issued.organization_member_id,
    )

    for invitation_id in (issued.invitation_id, second.id):
        cancelled = await InvitationRepository(async_db).get(invitation_id)
        assert cancelled is not None
        assert cancelled.status == "cancelled"

    with pytest.raises(InvitationAlreadyUsedError):
        await service.accept_invitation(second_token)

    membership = await OrganizationMemberRepository(async_db).get(issued.organization_member_id)
    assert membership is not None
    assert membership.status == "suspended"


async def test_a_stale_pending_row_alongside_a_live_one_resolves_to_the_live_invitation(
    async_db: AsyncSession,
) -> None:
    """At most one pending row per membership is a service invariant, not a constraint.

    So the resolve picks the live one rather than trusting that there is only
    ever one, the same way the invite path re-reads timestamps instead of
    statuses.
    """
    invitee, _ = await _identity_with_a_home(async_db, email="invitee@example.com")
    inviting = await _organization(async_db, slug="inviting")
    admin = await _owner(async_db, inviting, full_name="Admin")
    service = OrganizationService(async_db)

    issued = await _invite(service, admin, email="invitee@example.com")
    stale = Invitation(
        organization_id=inviting.id,
        organization_member_id=issued.organization_member_id,
        email="invitee@example.com",
        invited_by_user_id=admin.id,
        token_hash=uuid.uuid4().hex,
        workspace_assignments=[],
        expires_at=datetime.now(UTC) - timedelta(hours=1),
    )
    async_db.add(stale)
    await async_db.commit()

    result = await service.accept_pending_membership_for_user(
        user=invitee,
        organization_member_id=issued.organization_member_id,
    )
    assert result.organization_name == "Inviting"

    live = await InvitationRepository(async_db).get(issued.invitation_id)
    assert live is not None
    assert live.status == "accepted"
