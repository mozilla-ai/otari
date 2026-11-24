"""The first-request setup guide: derived status, key issuance, and dismissal.

Exercised at the service layer, matching `test_workspace_member_budget_policies.py`
and `test_tenancy_authorization.py`: the API can only ever act as the one
superuser operator identity a standalone deployment has, so the rules that
matter most (a member who may see a workspace without managing it, a foreign
workspace) are only reachable by calling the service with identities built at
whatever role a case needs.
"""

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.adapters.api_key_format_adapter import DefaultApiKeyFormatAdapter
from gateway.auth.models import hash_key
from gateway.core.config import GatewayConfig
from gateway.exceptions import TenancyConflictError
from gateway.exceptions.organizations_exceptions import (
    NotAuthorizedError,
    WorkspaceActivationUnavailableError,
    WorkspaceAlreadyActivatedError,
    WorkspaceNotFoundError,
)
from gateway.models.api_keys import APIKey, DisabledReason
from gateway.models.tenancy import (
    Organization,
    User,
    Workspace,
    WorkspaceActivationState,
    WorkspaceMemberActivationState,
)
from gateway.models.usage import UsageLog
from gateway.ports.api_key_format_port import KeyRoute, Local
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.repositories.users_repository import get_or_create_attribution_user
from gateway.services.api_keys import key_refusal
from gateway.services.tenancy.workspace_activation_service import (
    ACTIVATION_KEY_NAME,
    WorkspaceActivationService,
)

pytestmark = pytest.mark.asyncio

# The open-source format, which is what a service built outside a request would get.
KEY_FORMAT = DefaultApiKeyFormatAdapter(None)


def _config(*, activation_guide: bool = True) -> GatewayConfig:
    return GatewayConfig(activation_guide=activation_guide, master_key="test-master-key", auto_migrate=False)


async def _organization(db: AsyncSession, *, slug: str) -> Organization:
    return await OrganizationRepository(db).create_organization(name=slug.title(), slug=slug, created_by_user_id=None)


async def _member(db: AsyncSession, organization: Organization, *, role: str, full_name: str) -> User:
    user = await UserRepository(db).create_local_identity(
        full_name=full_name,
        active_organization_id=organization.id,
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
    owner: User,
    classification: str = "eligible",
) -> Workspace:
    workspace = await WorkspaceRepository(db).create_workspace(
        name=name,
        organization_id=organization.id,
        created_by_user_id=owner.id,
    )
    await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=owner.id, role="owner")
    if classification != "eligible":
        workspace.activation_classification = classification
        await db.commit()
    return workspace


async def _usage(
    db: AsyncSession,
    workspace_id: uuid.UUID,
    *,
    status: str = "success",
    source: str = "gateway",
    status_code: int | None = None,
    seconds_ago: int = 0,
    model: str = "openai:gpt-4o-mini",
    provider: str = "openai",
    cost: Decimal | None = Decimal("0.000123"),
    latency_ms: int | None = 412,
    billed_to: User | None = None,
) -> UsageLog:
    user_id = None
    if billed_to is not None:
        # A request on a member's own key bills to their attribution row.
        user_id = (await get_or_create_attribution_user(db, user_id=str(billed_to.id), alias=None)).user_id
    row = UsageLog(
        workspace_id=workspace_id,
        user_id=user_id,
        model=model,
        provider=provider,
        endpoint="/v1/chat/completions",
        status=status,
        source=source,
        status_code=status_code,
        cost=cost,
        latency_ms=latency_ms,
        timestamp=datetime.now(UTC) - timedelta(seconds=seconds_ago),
    )
    db.add(row)
    await db.commit()
    return row


async def _keys_in(db: AsyncSession, workspace_id: uuid.UUID) -> list[APIKey]:
    rows = await db.execute(select(APIKey).where(APIKey.workspace_id == workspace_id))
    return list(rows.scalars().all())


async def _setup(db: AsyncSession, *, slug: str, classification: str = "eligible") -> tuple[User, Workspace]:
    organization = await _organization(db, slug=slug)
    owner = await _member(db, organization, role="owner", full_name="Owner")
    workspace = await _workspace(db, organization, name="Engineering", owner=owner, classification=classification)
    return owner, workspace


# ----------------------------------------------------------------------
# Derived status
# ----------------------------------------------------------------------


async def test_a_workspace_with_no_traffic_is_waiting_and_is_offered_the_guide(async_db: AsyncSession) -> None:
    owner, workspace = await _setup(async_db, slug="acme-waiting")

    status = await WorkspaceActivationService(async_db, _config(), KEY_FORMAT).get_status(
        user=owner, workspace_id=workspace.id
    )

    assert status.status == "waiting"
    assert status.activation_attempt is None
    assert status.latest_attempt is None
    assert status.experience_eligible is True
    assert status.dismissed is False


async def test_a_failed_request_is_reported_with_its_category_while_the_guide_keeps_waiting(
    async_db: AsyncSession,
) -> None:
    owner, workspace = await _setup(async_db, slug="acme-failed")
    row = await _usage(async_db, workspace.id, status="error", status_code=403, cost=None, latency_ms=None)

    status = await WorkspaceActivationService(async_db, _config(), KEY_FORMAT).get_status(
        user=owner, workspace_id=workspace.id
    )

    assert status.status == "failed"
    assert status.activation_attempt is None
    assert status.latest_attempt is not None
    assert status.latest_attempt.request_id == row.id
    assert status.latest_attempt.status == "failed"
    # A budget, a model allow-list or a rate limit: one screen, one category.
    assert status.latest_attempt.error_category == "policy"
    assert status.latest_attempt.occurred_at.endswith("+00:00")
    # Still on offer: a failed attempt is news, not the end of the flow.
    assert status.experience_eligible is True


async def test_the_first_success_activates_the_workspace_and_retires_the_offer(async_db: AsyncSession) -> None:
    owner, workspace = await _setup(async_db, slug="acme-activated")
    first = await _usage(async_db, workspace.id, seconds_ago=120)
    await _usage(async_db, workspace.id, model="openai:gpt-4o", seconds_ago=1)

    status = await WorkspaceActivationService(async_db, _config(), KEY_FORMAT).get_status(
        user=owner, workspace_id=workspace.id
    )

    assert status.status == "activated"
    assert status.activation_attempt is not None
    # The receipt is the request that activated the workspace, not the newest one.
    assert status.activation_attempt.request_id == first.id
    assert status.activation_attempt.model == "openai:gpt-4o-mini"
    assert status.activation_attempt.status == "success"
    assert status.activation_attempt.error_category is None
    assert status.activation_attempt.cost_usd == pytest.approx(0.000123)
    assert status.activation_attempt.latency_ms == 412
    assert status.experience_eligible is False


async def test_imported_usage_does_not_activate_a_workspace(async_db: AsyncSession) -> None:
    """Somebody else's traffic recorded here for cost reporting is not a call to this gateway."""
    owner, workspace = await _setup(async_db, slug="acme-imported")
    await _usage(async_db, workspace.id, source="claude_code")

    status = await WorkspaceActivationService(async_db, _config(), KEY_FORMAT).get_status(
        user=owner, workspace_id=workspace.id
    )

    assert status.status == "waiting"
    assert status.latest_attempt is None
    assert status.experience_eligible is True


async def test_an_absorbed_attempt_is_not_reported_as_a_failure(async_db: AsyncSession) -> None:
    """A failed attempt a routing policy recovered from is not the request's outcome."""
    owner, workspace = await _setup(async_db, slug="acme-absorbed")
    await _usage(async_db, workspace.id, status="absorbed", status_code=502)

    status = await WorkspaceActivationService(async_db, _config(), KEY_FORMAT).get_status(
        user=owner, workspace_id=workspace.id
    )

    assert status.status == "waiting"
    assert status.latest_attempt is None


async def test_a_workspace_in_another_organization_is_not_found(async_db: AsyncSession) -> None:
    _, workspace = await _setup(async_db, slug="acme-foreign-owner")
    other_organization = await _organization(async_db, slug="acme-foreign-other")
    outsider = await _member(async_db, other_organization, role="owner", full_name="Outsider")

    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)
    with pytest.raises(WorkspaceNotFoundError):
        await service.get_status(user=outsider, workspace_id=workspace.id)


# ----------------------------------------------------------------------
# Eligibility
# ----------------------------------------------------------------------


async def test_a_disabled_deployment_withdraws_the_offer_and_refuses_a_key(async_db: AsyncSession) -> None:
    owner, workspace = await _setup(async_db, slug="acme-disabled")
    service = WorkspaceActivationService(async_db, _config(activation_guide=False), KEY_FORMAT)

    status = await service.get_status(user=owner, workspace_id=workspace.id)
    assert status.status == "waiting"
    assert status.experience_eligible is False

    with pytest.raises(WorkspaceActivationUnavailableError):
        await service.issue_api_key(user=owner, workspace_id=workspace.id)
    assert await _keys_in(async_db, workspace.id) == []


async def test_a_workspace_classified_out_of_the_guide_is_not_offered_it(async_db: AsyncSession) -> None:
    owner, workspace = await _setup(async_db, slug="acme-internal", classification="internal")
    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)

    status = await service.get_status(user=owner, workspace_id=workspace.id)
    assert status.experience_eligible is False

    with pytest.raises(WorkspaceActivationUnavailableError):
        await service.issue_api_key(user=owner, workspace_id=workspace.id)


async def test_a_viewer_is_not_offered_the_guide(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme-viewer")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    viewer = await _member(async_db, organization, role="member", full_name="Viewer")
    workspace = await _workspace(async_db, organization, name="Engineering", owner=owner)
    await WorkspaceMemberRepository(async_db).create(workspace_id=workspace.id, user_id=viewer.id, role="viewer")

    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)
    status = await service.get_status(user=viewer, workspace_id=workspace.id)
    # Visible, and honestly reported, rather than refused: the guide simply is
    # not theirs to act on.
    assert status.status == "waiting"
    assert status.experience_eligible is False

    with pytest.raises(NotAuthorizedError):
        await service.issue_api_key(user=viewer, workspace_id=workspace.id)
    with pytest.raises(NotAuthorizedError):
        await service.dismiss(user=viewer, workspace_id=workspace.id)


async def test_an_organization_viewer_holding_member_in_the_workspace_is_not_offered_the_guide(
    async_db: AsyncSession,
) -> None:
    """The more restrictive role wins, as on the member key surface."""
    organization = await _organization(async_db, slug="acme-org-viewer")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    viewer = await _member(async_db, organization, role="viewer", full_name="Viewer")
    workspace = await _workspace(async_db, organization, name="Engineering", owner=owner)
    await WorkspaceMemberRepository(async_db).create(workspace_id=workspace.id, user_id=viewer.id, role="member")

    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)
    assert (await service.get_status(user=viewer, workspace_id=workspace.id)).experience_eligible is False
    with pytest.raises(NotAuthorizedError):
        await service.issue_api_key(user=viewer, workspace_id=workspace.id)
    assert await _keys_in(async_db, workspace.id) == []


# ----------------------------------------------------------------------
# A member who does not manage the workspace
# ----------------------------------------------------------------------


async def _with_member(db: AsyncSession, *, slug: str) -> tuple[User, User, Workspace]:
    organization = await _organization(db, slug=slug)
    owner = await _member(db, organization, role="owner", full_name="Owner")
    member = await _member(db, organization, role="member", full_name="Member")
    workspace = await _workspace(db, organization, name="Engineering", owner=owner)
    await WorkspaceMemberRepository(db).create(workspace_id=workspace.id, user_id=member.id, role="member")
    return owner, member, workspace


async def test_a_member_is_offered_the_guide_for_their_own_first_request(async_db: AsyncSession) -> None:
    owner, member, workspace = await _with_member(async_db, slug="acme-member-offered")
    # The workspace already serves its managers; the member has sent nothing.
    await _usage(async_db, workspace.id, billed_to=owner)
    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)

    assert (await service.get_status(user=owner, workspace_id=workspace.id)).status == "activated"
    status = await service.get_status(user=member, workspace_id=workspace.id)
    assert status.status == "waiting"
    assert status.latest_attempt is None
    assert status.experience_eligible is True
    assert status.dismissed is False


async def test_a_member_activates_on_their_own_first_success(async_db: AsyncSession) -> None:
    _owner, member, workspace = await _with_member(async_db, slug="acme-member-activates")
    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)

    await _usage(async_db, workspace.id, billed_to=member, status="error", status_code=403, seconds_ago=60)
    status = await service.get_status(user=member, workspace_id=workspace.id)
    assert status.status == "failed"
    assert status.latest_attempt is not None and status.latest_attempt.error_category == "policy"
    assert status.experience_eligible is True

    first = await _usage(async_db, workspace.id, billed_to=member)
    status = await service.get_status(user=member, workspace_id=workspace.id)
    assert status.status == "activated"
    assert status.activation_attempt is not None and status.activation_attempt.request_id == first.id
    assert status.experience_eligible is False
    with pytest.raises(WorkspaceAlreadyActivatedError):
        await service.issue_api_key(user=member, workspace_id=workspace.id)


async def test_a_member_is_issued_a_key_of_their_own_and_the_managers_key_is_untouched(
    async_db: AsyncSession,
) -> None:
    owner, member, workspace = await _with_member(async_db, slug="acme-member-key")
    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)
    managers = await service.issue_api_key(user=owner, workspace_id=workspace.id)

    issued = await service.issue_api_key(user=member, workspace_id=workspace.id)

    key = await async_db.get(APIKey, issued.key_id)
    assert key is not None
    assert key.key_name == ACTIVATION_KEY_NAME
    assert key.user_id == str(member.id)
    assert key.self_service is True, "demoting the member to viewer must switch it off with their other keys"
    assert key.allowed_models is None and key.allowed_tools is None, "it inherits the member's access"
    managers_key = await async_db.get(APIKey, managers.key_id)
    assert managers_key is not None
    assert managers_key.key_hash == hash_key(managers.key), "the managers' plaintext still authenticates"
    assert managers_key.user_id == str(owner.id)

    state = await async_db.get(WorkspaceMemberActivationState, (workspace.id, member.id))
    assert state is not None and state.api_key_id == issued.key_id
    workspace_state = await async_db.get(WorkspaceActivationState, workspace.id)
    assert workspace_state is not None and workspace_state.api_key_id == managers.key_id

    status = await service.get_status(user=member, workspace_id=workspace.id)
    assert status.issued_key is not None and status.issued_key.key_id == issued.key_id
    assert (await service.get_status(user=owner, workspace_id=workspace.id)).issued_key is not None

    rotated = await service.issue_api_key(user=member, workspace_id=workspace.id)
    assert rotated.key_id == issued.key_id, "a second press rotates the member's row"
    assert len(await _keys_in(async_db, workspace.id)) == 2


@pytest.mark.parametrize("reason", [DisabledReason.OPERATOR, DisabledReason.REVOKED])
async def test_a_members_key_disabled_for_a_reason_they_cannot_clear_is_not_revived(
    async_db: AsyncSession, reason: DisabledReason
) -> None:
    _owner, member, workspace = await _with_member(async_db, slug=f"acme-member-disabled-{reason.value}")
    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)
    first = await service.issue_api_key(user=member, workspace_id=workspace.id)
    key = await async_db.get(APIKey, first.key_id)
    assert key is not None
    key.is_active = False
    key.disabled_reason = reason
    await async_db.commit()

    second = await service.issue_api_key(user=member, workspace_id=workspace.id)

    assert second.key_id != first.key_id
    await async_db.refresh(key)
    assert key.is_active is False and key.disabled_reason == reason
    assert key.key_hash == hash_key(first.key), "the disabled row keeps the hash it had"


async def test_a_members_key_moved_to_another_owner_is_not_rotated(async_db: AsyncSession) -> None:
    owner, member, workspace = await _with_member(async_db, slug="acme-member-moved")
    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)
    first = await service.issue_api_key(user=member, workspace_id=workspace.id)
    key = await async_db.get(APIKey, first.key_id)
    assert key is not None
    key.user_id = (await get_or_create_attribution_user(async_db, user_id=str(owner.id), alias=None)).user_id
    await async_db.commit()

    second = await service.issue_api_key(user=member, workspace_id=workspace.id)

    assert second.key_id != first.key_id
    await async_db.refresh(key)
    assert key.key_hash == hash_key(first.key)


async def test_a_members_skip_retires_the_guide_for_them_alone(async_db: AsyncSession) -> None:
    owner, member, workspace = await _with_member(async_db, slug="acme-member-skip")
    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)

    await service.dismiss(user=member, workspace_id=workspace.id)
    await service.dismiss(user=member, workspace_id=workspace.id)

    member_view = await service.get_status(user=member, workspace_id=workspace.id)
    assert member_view.dismissed is True
    assert member_view.experience_eligible is False
    with pytest.raises(WorkspaceActivationUnavailableError):
        await service.issue_api_key(user=member, workspace_id=workspace.id)
    owner_view = await service.get_status(user=owner, workspace_id=workspace.id)
    assert owner_view.dismissed is False
    assert owner_view.experience_eligible is True
    assert await async_db.get(WorkspaceActivationState, workspace.id) is None


async def test_a_managers_skip_does_not_retire_a_members_guide(async_db: AsyncSession) -> None:
    owner, member, workspace = await _with_member(async_db, slug="acme-manager-skip")
    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)

    await service.dismiss(user=owner, workspace_id=workspace.id)

    assert (await service.get_status(user=member, workspace_id=workspace.id)).experience_eligible is True


async def test_a_revoked_member_is_neither_offered_nor_given_a_setup_key(async_db: AsyncSession) -> None:
    _owner, member, workspace = await _with_member(async_db, slug="acme-member-revoked")
    attribution = await get_or_create_attribution_user(async_db, user_id=str(member.id), alias=None)
    attribution.deleted_at = datetime.now(UTC)
    await async_db.commit()
    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)

    assert (await service.get_status(user=member, workspace_id=workspace.id)).experience_eligible is False
    with pytest.raises(TenancyConflictError):
        await service.issue_api_key(user=member, workspace_id=workspace.id)
    await async_db.refresh(attribution)
    assert attribution.deleted_at is not None, "the refusal must not revive the identity"
    assert await _keys_in(async_db, workspace.id) == []


@pytest.mark.parametrize(("activation_guide", "classification"), [(False, "eligible"), (True, "internal")])
async def test_a_member_is_not_offered_the_guide_where_the_workspace_is_not(
    async_db: AsyncSession, activation_guide: bool, classification: str
) -> None:
    organization = await _organization(async_db, slug=f"acme-member-off-{classification}-{activation_guide}")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    member = await _member(async_db, organization, role="member", full_name="Member")
    workspace = await _workspace(async_db, organization, name="Engineering", owner=owner, classification=classification)
    await WorkspaceMemberRepository(async_db).create(workspace_id=workspace.id, user_id=member.id, role="member")
    service = WorkspaceActivationService(async_db, _config(activation_guide=activation_guide), KEY_FORMAT)

    assert (await service.get_status(user=member, workspace_id=workspace.id)).experience_eligible is False
    with pytest.raises(WorkspaceActivationUnavailableError):
        await service.issue_api_key(user=member, workspace_id=workspace.id)


# ----------------------------------------------------------------------
# Key issuance
# ----------------------------------------------------------------------


async def test_issuing_the_key_twice_rotates_one_row_rather_than_collecting_two(async_db: AsyncSession) -> None:
    owner, workspace = await _setup(async_db, slug="acme-rotate")
    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)

    first = await service.issue_api_key(user=owner, workspace_id=workspace.id)
    second = await service.issue_api_key(user=owner, workspace_id=workspace.id)

    assert first.key != second.key
    assert first.key_id == second.key_id
    assert second.key_name == ACTIVATION_KEY_NAME
    assert second.key_prefix is not None and second.key.startswith(second.key_prefix)
    # The rotation branch re-fingerprints both halves; a stale suffix would name the
    # plaintext the first call handed out, which no longer authenticates.
    assert second.key_suffix is not None and second.key.endswith(second.key_suffix)
    assert first.key_suffix is not None and first.key.endswith(first.key_suffix)

    keys = await _keys_in(async_db, workspace.id)
    assert [key.key_name for key in keys] == [ACTIVATION_KEY_NAME]
    # The previous plaintext no longer authenticates: the row carries the new hash.
    assert keys[0].is_active is True
    # Owned by the caller's own request-plane row, so the request the guide walks
    # them through bills through their per-member ceilings rather than sitting
    # outside every budget on the shared "default" owner.
    assert keys[0].user_id == str(owner.id)

    state = await async_db.get(WorkspaceActivationState, workspace.id)
    assert state is not None
    assert state.api_key_id == first.key_id
    assert state.first_presented_at is not None
    assert state.last_presented_at is not None
    assert state.first_presented_at <= state.last_presented_at


async def test_rotating_the_key_moves_its_owner_to_whoever_asked_last(async_db: AsyncSession) -> None:
    """The only live plaintext belongs to the last caller, so the row's owner does too.

    Otherwise a second manager's requests would bill against the first issuer's
    per-member ceilings, on a key the first issuer can no longer use.
    """
    organization = await _organization(async_db, slug="acme-rotate-owner")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    admin = await _member(async_db, organization, role="admin", full_name="Admin")
    workspace = await _workspace(async_db, organization, name="Engineering", owner=owner)

    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)
    first = await service.issue_api_key(user=owner, workspace_id=workspace.id)
    second = await service.issue_api_key(user=admin, workspace_id=workspace.id)

    assert first.key_id == second.key_id
    keys = await _keys_in(async_db, workspace.id)
    assert len(keys) == 1
    assert keys[0].user_id == str(admin.id)


async def test_issuing_the_key_is_refused_once_the_workspace_has_activated(async_db: AsyncSession) -> None:
    owner, workspace = await _setup(async_db, slug="acme-retired")
    await _usage(async_db, workspace.id)

    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)
    with pytest.raises(WorkspaceAlreadyActivatedError):
        await service.issue_api_key(user=owner, workspace_id=workspace.id)
    assert await _keys_in(async_db, workspace.id) == []


async def test_a_key_deleted_from_the_keys_page_is_replaced_rather_than_resurrected(async_db: AsyncSession) -> None:
    owner, workspace = await _setup(async_db, slug="acme-deleted-key")
    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)
    issued = await service.issue_api_key(user=owner, workspace_id=workspace.id)

    deleted = await async_db.get(APIKey, issued.key_id)
    assert deleted is not None
    await async_db.delete(deleted)
    await async_db.commit()

    reissued = await service.issue_api_key(user=owner, workspace_id=workspace.id)
    assert reissued.key_id != issued.key_id
    state = await async_db.get(WorkspaceActivationState, workspace.id)
    assert state is not None
    assert state.api_key_id == reissued.key_id


async def test_a_reopened_guide_is_told_about_the_key_it_already_issued(async_db: AsyncSession) -> None:
    owner, workspace = await _setup(async_db, slug="acme-issued-key")
    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)

    before = await service.get_status(user=owner, workspace_id=workspace.id)
    assert before.issued_key is None

    issued = await service.issue_api_key(user=owner, workspace_id=workspace.id)
    status = await service.get_status(user=owner, workspace_id=workspace.id)

    assert status.issued_key is not None
    assert status.issued_key.key_id == issued.key_id
    assert status.issued_key.key_prefix == issued.key_prefix
    assert status.issued_key.key_suffix == issued.key_suffix
    dumped = status.model_dump_json()
    assert issued.key not in dumped
    assert hash_key(issued.key) not in dumped


async def test_a_revoked_or_expired_setup_key_is_not_offered_back(async_db: AsyncSession) -> None:
    owner, workspace = await _setup(async_db, slug="acme-issued-revoked")
    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)
    issued = await service.issue_api_key(user=owner, workspace_id=workspace.id)
    key = await async_db.get(APIKey, issued.key_id)
    assert key is not None

    key.is_active = False
    await async_db.commit()
    assert (await service.get_status(user=owner, workspace_id=workspace.id)).issued_key is None

    key.is_active = True
    key.expires_at = datetime.now(UTC) - timedelta(minutes=1)
    await async_db.commit()
    assert (await service.get_status(user=owner, workspace_id=workspace.id)).issued_key is None

    await async_db.delete(key)
    await async_db.commit()
    assert (await service.get_status(user=owner, workspace_id=workspace.id)).issued_key is None


async def test_the_issued_key_is_only_described_to_whoever_is_offered_the_guide(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme-issued-viewer")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    viewer = await _member(async_db, organization, role="member", full_name="Viewer")
    workspace = await _workspace(async_db, organization, name="Engineering", owner=owner)
    await WorkspaceMemberRepository(async_db).create(workspace_id=workspace.id, user_id=viewer.id, role="viewer")
    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)
    await service.issue_api_key(user=owner, workspace_id=workspace.id)

    assert (await service.get_status(user=viewer, workspace_id=workspace.id)).issued_key is None

    await service.dismiss(user=owner, workspace_id=workspace.id)
    assert (await service.get_status(user=owner, workspace_id=workspace.id)).issued_key is None


async def test_the_issued_key_is_offered_back_only_to_the_manager_holding_it(async_db: AsyncSession) -> None:
    """Another manager never saw that plaintext, so they are offered a key of their own."""
    organization = await _organization(async_db, slug="acme-issued-holder")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    admin = await _member(async_db, organization, role="admin", full_name="Admin")
    workspace = await _workspace(async_db, organization, name="Engineering", owner=owner)
    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)

    issued = await service.issue_api_key(user=owner, workspace_id=workspace.id)
    owner_view = await service.get_status(user=owner, workspace_id=workspace.id)
    admin_view = await service.get_status(user=admin, workspace_id=workspace.id)
    assert owner_view.issued_key is not None and owner_view.issued_key.key_id == issued.key_id
    assert admin_view.experience_eligible is True
    assert admin_view.issued_key is None

    rotated = await service.issue_api_key(user=admin, workspace_id=workspace.id)
    assert (await service.get_status(user=owner, workspace_id=workspace.id)).issued_key is None
    admin_view = await service.get_status(user=admin, workspace_id=workspace.id)
    assert admin_view.issued_key is not None and admin_view.issued_key.key_suffix == rotated.key_suffix


async def test_replacing_an_expired_setup_key_issues_one_that_authenticates(async_db: AsyncSession) -> None:
    owner, workspace = await _setup(async_db, slug="acme-expired-rotate")
    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)
    issued = await service.issue_api_key(user=owner, workspace_id=workspace.id)
    record = await async_db.get(APIKey, issued.key_id)
    assert record is not None
    record.expires_at = datetime.now(UTC) - timedelta(minutes=1)
    await async_db.commit()

    reissued = await service.issue_api_key(user=owner, workspace_id=workspace.id)

    assert reissued.key_id == issued.key_id
    await async_db.refresh(record)
    assert record.expires_at is None
    assert key_refusal(record, datetime.now(UTC)) is None


# ----------------------------------------------------------------------
# Dismissal
# ----------------------------------------------------------------------


async def test_dismiss_retires_the_guide_idempotently(async_db: AsyncSession) -> None:
    owner, workspace = await _setup(async_db, slug="acme-dismiss")
    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)
    await service.issue_api_key(user=owner, workspace_id=workspace.id)

    await service.dismiss(user=owner, workspace_id=workspace.id)
    state = await async_db.get(WorkspaceActivationState, workspace.id)
    assert state is not None
    assert state.dismissed_at is not None
    dismissed_at = state.dismissed_at

    await service.dismiss(user=owner, workspace_id=workspace.id)
    await async_db.refresh(state)
    assert state.dismissed_at == dismissed_at, "a second Skip must not restamp the dismissal"

    status = await service.get_status(user=owner, workspace_id=workspace.id)
    assert status.dismissed is True
    assert status.experience_eligible is False
    with pytest.raises(WorkspaceActivationUnavailableError):
        await service.issue_api_key(user=owner, workspace_id=workspace.id)


async def test_dismiss_leaves_the_key_it_issued_working(async_db: AsyncSession) -> None:
    """Skip retires the card, not the credential.

    The likeliest sequence in the whole flow is copy the key, paste it into your
    app, press Skip to clear the card, then send the request. Deactivating an
    unused key would answer that with a 401, in the flow that exists to make it
    a 200. Revoking a key is the Keys page's job.
    """
    owner, workspace = await _setup(async_db, slug="acme-dismiss-key")
    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)
    issued = await service.issue_api_key(user=owner, workspace_id=workspace.id)

    await service.dismiss(user=owner, workspace_id=workspace.id)

    key = await async_db.get(APIKey, issued.key_id)
    assert key is not None
    assert key.is_active is True
    assert key.key_hash == hash_key(issued.key), "the plaintext the operator copied still authenticates"


async def test_dismiss_before_the_guide_ever_issued_a_key_still_records_it(async_db: AsyncSession) -> None:
    """Skip has to work on the presentation that never got as far as a key."""
    owner, workspace = await _setup(async_db, slug="acme-dismiss-early")
    service = WorkspaceActivationService(async_db, _config(), KEY_FORMAT)

    await service.dismiss(user=owner, workspace_id=workspace.id)

    state = await async_db.get(WorkspaceActivationState, workspace.id)
    assert state is not None
    assert state.dismissed_at is not None
    assert state.api_key_id is None


class _ProbeKeyFormat:
    """A format an overlay might bind, to show the service mints through the port."""

    def mint(self) -> str:
        return "probe-" + uuid.uuid4().hex + uuid.uuid4().hex

    def fingerprint(self, api_key: str) -> str:
        return api_key[:13]

    def route(self, presented: str) -> KeyRoute:
        return Local()


async def test_the_setup_key_is_minted_in_the_bound_format(async_db: AsyncSession) -> None:
    owner, workspace = await _setup(async_db, slug="acme-probe-format")
    service = WorkspaceActivationService(async_db, _config(), _ProbeKeyFormat())

    issued = await service.issue_api_key(user=owner, workspace_id=workspace.id)

    assert issued.key.startswith("probe-")
    assert issued.key_prefix == issued.key[:13]
    assert issued.key_suffix == issued.key[-4:]
