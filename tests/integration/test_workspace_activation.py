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

from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey, UsageLog, WorkspaceActivationState
from gateway.models.tenancy import Organization, User, Workspace
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    OrganizationRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.services.tenancy.errors import (
    NotAuthorizedError,
    WorkspaceActivationUnavailableError,
    WorkspaceAlreadyActivatedError,
    WorkspaceNotFoundError,
)
from gateway.services.tenancy.workspace_activation_service import (
    ACTIVATION_KEY_NAME,
    WorkspaceActivationService,
)

pytestmark = pytest.mark.asyncio


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
) -> UsageLog:
    row = UsageLog(
        workspace_id=workspace_id,
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

    status = await WorkspaceActivationService(async_db, _config()).get_status(user=owner, workspace_id=workspace.id)

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

    status = await WorkspaceActivationService(async_db, _config()).get_status(user=owner, workspace_id=workspace.id)

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

    status = await WorkspaceActivationService(async_db, _config()).get_status(user=owner, workspace_id=workspace.id)

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

    status = await WorkspaceActivationService(async_db, _config()).get_status(user=owner, workspace_id=workspace.id)

    assert status.status == "waiting"
    assert status.latest_attempt is None
    assert status.experience_eligible is True


async def test_an_absorbed_attempt_is_not_reported_as_a_failure(async_db: AsyncSession) -> None:
    """A failed attempt a routing policy recovered from is not the request's outcome."""
    owner, workspace = await _setup(async_db, slug="acme-absorbed")
    await _usage(async_db, workspace.id, status="absorbed", status_code=502)

    status = await WorkspaceActivationService(async_db, _config()).get_status(user=owner, workspace_id=workspace.id)

    assert status.status == "waiting"
    assert status.latest_attempt is None


async def test_a_workspace_in_another_organization_is_not_found(async_db: AsyncSession) -> None:
    _, workspace = await _setup(async_db, slug="acme-foreign-owner")
    other_organization = await _organization(async_db, slug="acme-foreign-other")
    outsider = await _member(async_db, other_organization, role="owner", full_name="Outsider")

    service = WorkspaceActivationService(async_db, _config())
    with pytest.raises(WorkspaceNotFoundError):
        await service.get_status(user=outsider, workspace_id=workspace.id)


# ----------------------------------------------------------------------
# Eligibility
# ----------------------------------------------------------------------


async def test_a_disabled_deployment_withdraws_the_offer_and_refuses_a_key(async_db: AsyncSession) -> None:
    owner, workspace = await _setup(async_db, slug="acme-disabled")
    service = WorkspaceActivationService(async_db, _config(activation_guide=False))

    status = await service.get_status(user=owner, workspace_id=workspace.id)
    assert status.status == "waiting"
    assert status.experience_eligible is False

    with pytest.raises(WorkspaceActivationUnavailableError):
        await service.issue_api_key(user=owner, workspace_id=workspace.id)
    assert await _keys_in(async_db, workspace.id) == []


async def test_a_workspace_classified_out_of_the_guide_is_not_offered_it(async_db: AsyncSession) -> None:
    owner, workspace = await _setup(async_db, slug="acme-internal", classification="internal")
    service = WorkspaceActivationService(async_db, _config())

    status = await service.get_status(user=owner, workspace_id=workspace.id)
    assert status.experience_eligible is False

    with pytest.raises(WorkspaceActivationUnavailableError):
        await service.issue_api_key(user=owner, workspace_id=workspace.id)


async def test_a_member_who_cannot_manage_the_workspace_is_not_offered_the_guide(async_db: AsyncSession) -> None:
    organization = await _organization(async_db, slug="acme-viewer")
    owner = await _member(async_db, organization, role="owner", full_name="Owner")
    viewer = await _member(async_db, organization, role="member", full_name="Viewer")
    workspace = await _workspace(async_db, organization, name="Engineering", owner=owner)
    await WorkspaceMemberRepository(async_db).create(workspace_id=workspace.id, user_id=viewer.id, role="member")

    service = WorkspaceActivationService(async_db, _config())
    status = await service.get_status(user=viewer, workspace_id=workspace.id)
    # Visible, and honestly reported, rather than refused: the guide simply is
    # not theirs to act on.
    assert status.status == "waiting"
    assert status.experience_eligible is False

    with pytest.raises(NotAuthorizedError):
        await service.issue_api_key(user=viewer, workspace_id=workspace.id)
    with pytest.raises(NotAuthorizedError):
        await service.dismiss(user=viewer, workspace_id=workspace.id)


# ----------------------------------------------------------------------
# Key issuance
# ----------------------------------------------------------------------


async def test_issuing_the_key_twice_rotates_one_row_rather_than_collecting_two(async_db: AsyncSession) -> None:
    owner, workspace = await _setup(async_db, slug="acme-rotate")
    service = WorkspaceActivationService(async_db, _config())

    first = await service.issue_api_key(user=owner, workspace_id=workspace.id)
    second = await service.issue_api_key(user=owner, workspace_id=workspace.id)

    assert first.key != second.key
    assert first.key_id == second.key_id
    assert second.key_name == ACTIVATION_KEY_NAME
    assert second.key_prefix is not None and second.key.startswith(second.key_prefix)

    keys = await _keys_in(async_db, workspace.id)
    assert [key.key_name for key in keys] == [ACTIVATION_KEY_NAME]
    # The previous plaintext no longer authenticates: the row carries the new hash.
    assert keys[0].is_active is True

    state = await async_db.get(WorkspaceActivationState, workspace.id)
    assert state is not None
    assert state.api_key_id == first.key_id
    assert state.first_presented_at is not None
    assert state.last_presented_at is not None
    assert state.first_presented_at <= state.last_presented_at


async def test_issuing_the_key_is_refused_once_the_workspace_has_activated(async_db: AsyncSession) -> None:
    owner, workspace = await _setup(async_db, slug="acme-retired")
    await _usage(async_db, workspace.id)

    service = WorkspaceActivationService(async_db, _config())
    with pytest.raises(WorkspaceAlreadyActivatedError):
        await service.issue_api_key(user=owner, workspace_id=workspace.id)
    assert await _keys_in(async_db, workspace.id) == []


async def test_a_key_deleted_from_the_keys_page_is_replaced_rather_than_resurrected(async_db: AsyncSession) -> None:
    owner, workspace = await _setup(async_db, slug="acme-deleted-key")
    service = WorkspaceActivationService(async_db, _config())
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


# ----------------------------------------------------------------------
# Dismissal
# ----------------------------------------------------------------------


async def test_dismiss_retires_the_guide_idempotently_and_deactivates_an_unused_key(async_db: AsyncSession) -> None:
    owner, workspace = await _setup(async_db, slug="acme-dismiss")
    service = WorkspaceActivationService(async_db, _config())
    issued = await service.issue_api_key(user=owner, workspace_id=workspace.id)

    await service.dismiss(user=owner, workspace_id=workspace.id)
    state = await async_db.get(WorkspaceActivationState, workspace.id)
    assert state is not None
    assert state.dismissed_at is not None
    dismissed_at = state.dismissed_at

    key = await async_db.get(APIKey, issued.key_id)
    assert key is not None
    assert key.is_active is False

    await service.dismiss(user=owner, workspace_id=workspace.id)
    await async_db.refresh(state)
    assert state.dismissed_at == dismissed_at, "a second Skip must not restamp the dismissal"

    status = await service.get_status(user=owner, workspace_id=workspace.id)
    assert status.dismissed is True
    assert status.experience_eligible is False
    with pytest.raises(WorkspaceActivationUnavailableError):
        await service.issue_api_key(user=owner, workspace_id=workspace.id)


async def test_dismiss_leaves_a_key_that_has_already_served_a_request_alone(async_db: AsyncSession) -> None:
    owner, workspace = await _setup(async_db, slug="acme-dismiss-used")
    service = WorkspaceActivationService(async_db, _config())
    issued = await service.issue_api_key(user=owner, workspace_id=workspace.id)

    key = await async_db.get(APIKey, issued.key_id)
    assert key is not None
    key.last_used_at = datetime.now(UTC)
    await async_db.commit()

    await service.dismiss(user=owner, workspace_id=workspace.id)

    await async_db.refresh(key)
    assert key.is_active is True, "somebody's working integration must survive a Skip"


async def test_dismiss_before_the_guide_ever_issued_a_key_still_records_it(async_db: AsyncSession) -> None:
    """Skip has to work on the presentation that never got as far as a key."""
    owner, workspace = await _setup(async_db, slug="acme-dismiss-early")
    service = WorkspaceActivationService(async_db, _config())

    await service.dismiss(user=owner, workspace_id=workspace.id)

    state = await async_db.get(WorkspaceActivationState, workspace.id)
    assert state is not None
    assert state.dismissed_at is not None
    assert state.api_key_id is None
