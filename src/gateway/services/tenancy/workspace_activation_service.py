"""The dashboard's first-request setup guide, per workspace.

A workspace that has never served a request is a dashboard full of empty
panels, and the one thing an operator needs from it is a working call. This
service is the state behind the guide that walks them there: whether to offer
it at all, the API key it hands out, what the workspace's traffic says about
the attempt, and the dismissal that retires it.

Ported from the platform's ``WorkspaceActivationService``
(`otari-ai` `backend/app/services/workspace_activation_service.py`), with three
deliberate departures, each of which is a consequence of this edition rather
than a simplification:

- **Activation is derived, not recorded.** The platform stores the attempt
  telemetry in columns because its usage pipeline is asynchronous and crosses
  services, so it has no local row to read. Here the usage row is written by
  this process into this database, so the first successful gateway request in
  the workspace *is* the evidence, queried through the index the migration adds
  for it. Nothing to backfill, and nothing that can disagree with the Activity
  page.
- **Nothing is per-viewer.** The platform keys its state on the user, because
  activation there is a person's first request on a hosted account. A
  standalone deployment's guide is about the workspace, so the state is one row
  per workspace and dismissing it says "this workspace is set up" for everyone
  who can manage it.
- **No profile survey, and no analytics.** The platform's success screen
  carries a questionnaire whose answers exist to be mapped onto CRM properties,
  which is permanently hosted-only (otari-ai#1749), and its presentation counter
  feeds a product-analytics funnel this dashboard deliberately does not have.
  Neither crosses.

The guide never invents a credential quietly: issuing a key is a workspace
management action, authorized exactly like renaming the workspace, and the key
it mints is an ordinary ``api_keys`` row that shows up on the Keys page under
its own name.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.auth.models import generate_api_key, hash_key, key_prefix
from gateway.core.config import GatewayConfig
from gateway.models.entities import APIKey, UsageLog, WorkspaceActivationState
from gateway.models.money import as_float
from gateway.models.tenancy import User, Workspace
from gateway.repositories.users_repository import get_or_create_attribution_user
from gateway.services.tenancy import authorization
from gateway.services.tenancy.errors import (
    WorkspaceActivationUnavailableError,
    WorkspaceAlreadyActivatedError,
)
from gateway.services.tenancy.organization_service import OrganizationService

# What the guide calls the key it mints, as the Keys page shows it. One name for
# every deployment, so an operator who finds it there can tell where it came
# from without a lookup.
ACTIVATION_KEY_NAME = "Setup guide"

# Only gateway-served requests count. Imported usage
# (``POST /v1/usage/external-events``) is somebody else's traffic recorded here
# for cost reporting, so a workspace whose only rows came from an import has
# still never called this gateway, and the guide would be lying to close.
_GATEWAY_SOURCE = "gateway"

# ``absorbed`` is excluded on purpose: it marks a failed attempt a routing
# policy recovered from, so the request it belongs to also wrote the
# ``success`` or ``error`` row that says what happened. Counting it would report
# a failure for a request that succeeded on its next candidate.
_ATTEMPT_STATUSES = ("success", "error")

# The one classification that is offered the guide. The column is the platform's
# and nothing else in this edition reads it; an operator who marks a workspace
# ``internal``, ``automated``, ``migrated`` or ``enterprise_assisted`` is saying
# its first request is not a milestone worth walking anyone through.
_ELIGIBLE_CLASSIFICATION = "eligible"

ActivationStatus = Literal["waiting", "failed", "activated"]
ActivationAttemptStatus = Literal["success", "failed"]
ActivationErrorCategory = Literal[
    "invalid_request",
    "configuration",
    "policy",
    "upstream",
    "timeout",
    "internal",
]

# The failure the guide reports, from the status that classifies the usage row.
# Grouped by what an operator would go and fix, which is why 402 (no pricing for
# the model) is configuration while 403 (a budget, a model allow-list, a rate
# limit) is policy: the two are different screens.
_ERROR_CATEGORY_BY_STATUS: dict[int, ActivationErrorCategory] = {
    400: "invalid_request",
    402: "configuration",
    403: "policy",
    404: "invalid_request",
    422: "invalid_request",
    429: "policy",
    502: "upstream",
    503: "upstream",
    504: "timeout",
}


def activation_error_category(status_code: int | None) -> ActivationErrorCategory:
    """Classify a failed request for the guide's failure copy.

    Never the provider's error prose, which is the whole point: the dashboard
    renders its own sentence per category, so a category is all that has to
    cross the wire and nothing upstream chooses what an operator reads.

    A status with no entry falls to ``invalid_request`` for the rest of the 4xx
    range (the caller sent something this gateway refused) and to ``internal``
    otherwise, including a row with no status at all, which is what a stream
    that ended without usage data leaves behind.
    """
    if status_code is None:
        return "internal"
    mapped = _ERROR_CATEGORY_BY_STATUS.get(status_code)
    if mapped is not None:
        return mapped
    if 400 <= status_code < 500:
        return "invalid_request"
    return "internal"


def _utc_iso(value: datetime) -> str:
    """Serialize a stored timestamp as unambiguous UTC ISO-8601.

    ``usage_logs.timestamp`` is timezone-aware, but SQLite does not persist the
    offset and hands it back naive, and a browser reads an offset-less timestamp
    as local time. The same four lines as ``api/routes/usage.py``'s ``_utc_iso``,
    duplicated rather than imported because a service may not import the API
    layer (`scripts/check_architecture.py`).
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


class ActivationAttemptPublic(BaseModel):
    """One gateway request in this workspace, as the guide reports it."""

    occurred_at: str = Field(description="When the request was recorded, UTC ISO-8601.")
    request_id: str = Field(description="The usage row's id, which the Activity page can be filtered by.")
    status: ActivationAttemptStatus
    provider: str | None = Field(default=None, description="Provider instance that served it, when one did.")
    model: str | None = Field(default=None, description="Model the request named.")
    error_category: ActivationErrorCategory | None = Field(
        default=None,
        description=(
            "What kind of failure this was, for a failed attempt only. The dashboard renders its "
            "own sentence per category; the provider's own error text is never returned here."
        ),
    )
    cost_usd: float | None = None
    latency_ms: int | None = None


class WorkspaceActivationPublic(BaseModel):
    """Where a workspace stands on its first successful request."""

    status: ActivationStatus = Field(
        description=(
            "'activated' once a gateway request in this workspace has succeeded, 'failed' when the "
            "last one failed and none has yet succeeded, 'waiting' when there has been none at all."
        )
    )
    activation_attempt: ActivationAttemptPublic | None = Field(
        default=None,
        description="The first request that succeeded, which is the guide's receipt. Null until one does.",
    )
    latest_attempt: ActivationAttemptPublic | None = Field(
        default=None,
        description="The most recent request, so a failure can be reported while the guide keeps waiting.",
    )
    experience_eligible: bool = Field(
        description=(
            "Whether the dashboard should offer the guide to this caller right now: the deployment "
            "has it enabled, the workspace is classified for it, nobody dismissed it, no request has "
            "succeeded yet, and the caller may manage the workspace."
        )
    )
    dismissed: bool = Field(description="Whether someone skipped the guide for this workspace.")


class ActivationApiKeyPublic(BaseModel):
    """The API key the guide hands out, with its plaintext.

    Returned once per call and never stored in plaintext, like every other key
    this gateway mints: a page reload issues a new one and rotates the same row,
    which is what makes the guide able to show a working key without keeping a
    readable secret anywhere.
    """

    key: str
    key_id: str
    key_prefix: str | None
    key_name: str | None


class WorkspaceActivationService:
    """State and key issuance for the first-request setup guide."""

    def __init__(self, db: AsyncSession, config: GatewayConfig):
        self.db = db
        self.config = config
        self.organizations = OrganizationService(db)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def get_status(self, *, user: User, workspace_id: uuid.UUID) -> WorkspaceActivationPublic:
        """Where the workspace stands, and whether the guide should be offered.

        Reachable by any member who can see the workspace; ``experience_eligible``
        is what says whether they are the one being offered the guide, so a
        viewer gets an honest "not for you" rather than a 403 the dashboard
        would have to interpret.
        """
        workspace = await authorization.resolve_visible_workspace(
            self.db,
            user=user,
            workspace_id=workspace_id,
            organizations=self.organizations,
        )
        state = await self.db.get(WorkspaceActivationState, workspace.id)
        dismissed = state is not None and state.dismissed_at is not None

        first_success = await self._first_successful_request(workspace.id)
        if first_success is not None:
            # Activated: the receipt is the request that did it, and the latest
            # attempt is not read at all. That keeps the steady state (every
            # dashboard load on an established workspace) at one index seek.
            return WorkspaceActivationPublic(
                status="activated",
                activation_attempt=_attempt_public(first_success),
                experience_eligible=False,
                dismissed=dismissed,
            )

        latest = await self._latest_request(workspace.id)
        return WorkspaceActivationPublic(
            status="failed" if latest is not None else "waiting",
            latest_attempt=_attempt_public(latest) if latest is not None else None,
            experience_eligible=await self._is_eligible(user=user, workspace=workspace, dismissed=dismissed),
            dismissed=dismissed,
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def issue_api_key(self, *, user: User, workspace_id: uuid.UUID) -> ActivationApiKeyPublic:
        """Mint the workspace's setup key, rotating the one the guide already issued.

        One row per workspace, rotated in place, so a guide reopened five times
        leaves one key on the Keys page rather than five. Rotation invalidates
        the previous plaintext, which is the same trade the platform's guide
        makes and the reason the modal says a reload issues a new key.

        Refuses once the workspace has activated or the guide was dismissed: a
        retired flow does not get to keep handing out credentials, and a stale
        browser tab is the caller most likely to try.
        """
        workspace = await self._resolve_manageable(user=user, workspace_id=workspace_id)
        state = await self._state_for_update(workspace.id)
        self._require_offerable(workspace=workspace, state=state)
        if await self._first_successful_request(workspace.id) is not None:
            raise WorkspaceAlreadyActivatedError

        plaintext = generate_api_key()
        record = await self._existing_key(state)
        if record is None:
            # Owned by the caller's own request-plane row, not the shared
            # ``default`` user that ``POST /v1/keys`` falls back to. Two reasons:
            # the dashboard's own key form requires an owner, so a key minted
            # from a dashboard flow should have a real one; and a key owned by an
            # identity's attribution row is what makes the request bill through
            # that member's scoped ceilings (`services/scoped_budget_service.py`
            # resolves the identity back out of ``users.user_id``), where one
            # owned by ``default`` would sit outside every per-member budget.
            # The row normally exists already: first-boot provisioning mints the
            # operator's, and adding a member mints theirs.
            owner = await get_or_create_attribution_user(
                self.db,
                user_id=str(user.id),
                alias=user.full_name or user.email,
            )
            record = APIKey(
                id=str(uuid.uuid4()),
                workspace_id=workspace.id,
                key_hash=hash_key(plaintext),
                key_prefix=key_prefix(plaintext),
                key_name=ACTIVATION_KEY_NAME,
                user_id=owner.user_id,
            )
            self.db.add(record)
        else:
            record.key_hash = hash_key(plaintext)
            record.key_prefix = key_prefix(plaintext)
            # A key the operator revoked from the Keys page and then asked the
            # guide for again is being asked for deliberately.
            record.is_active = True

        now = datetime.now(UTC)
        state = state or await self._create_state(workspace.id)
        if state.first_presented_at is None:
            state.first_presented_at = now
        state.last_presented_at = now
        state.api_key_id = record.id
        await self.db.commit()
        return ActivationApiKeyPublic(
            key=plaintext,
            key_id=record.id,
            key_prefix=record.key_prefix,
            key_name=record.key_name,
        )

    async def dismiss(self, *, user: User, workspace_id: uuid.UUID) -> None:
        """Retire the guide for this workspace. Permanent, and idempotent.

        The key the guide issued is deactivated on the way out **unless it has
        been used**: a credential nobody asked for and nobody called is a
        liability, while one that has already served a request is somebody's
        working integration and is left alone. Either way the row stays on the
        Keys page, so nothing disappears silently.
        """
        workspace = await self._resolve_manageable(user=user, workspace_id=workspace_id)
        state = await self._state_for_update(workspace.id)
        if state is None:
            state = await self._create_state(workspace.id)
        if state.dismissed_at is None:
            state.dismissed_at = datetime.now(UTC)
        record = await self._existing_key(state)
        if record is not None and record.last_used_at is None:
            record.is_active = False
        await self.db.commit()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _resolve_manageable(self, *, user: User, workspace_id: uuid.UUID) -> Workspace:
        """Resolve a workspace this caller may manage, or raise 404/403.

        Both writes here are management actions: one mints a credential, the
        other retires a workspace-wide offer. A member who may only see the
        workspace is refused, which is also why ``get_status`` reports
        eligibility rather than assuming the reader may act.
        """
        workspace = await authorization.resolve_visible_workspace(
            self.db,
            user=user,
            workspace_id=workspace_id,
            organizations=self.organizations,
        )
        await authorization.require_workspace_management_access(
            self.db,
            user=user,
            workspace=workspace,
            organizations=self.organizations,
        )
        return workspace

    async def _is_eligible(self, *, user: User, workspace: Workspace, dismissed: bool) -> bool:
        """Whether this caller is being offered the guide for this workspace.

        Callers reach this only on the not-yet-activated path, so activation is
        not re-checked here.
        """
        if not self.config.activation_guide:
            return False
        if workspace.activation_classification != _ELIGIBLE_CLASSIFICATION or dismissed:
            return False
        return await authorization.has_workspace_management_access(
            self.db,
            user=user,
            workspace=workspace,
            organizations=self.organizations,
        )

    def _require_offerable(self, *, workspace: Workspace, state: WorkspaceActivationState | None) -> None:
        """Refuse a write when the guide is not on offer for this workspace."""
        if not self.config.activation_guide:
            raise WorkspaceActivationUnavailableError("The first-request setup guide is disabled on this deployment")
        if workspace.activation_classification != _ELIGIBLE_CLASSIFICATION:
            raise WorkspaceActivationUnavailableError(
                f"This workspace is classified '{workspace.activation_classification}', "
                "which is not offered the first-request setup guide"
            )
        if state is not None and state.dismissed_at is not None:
            raise WorkspaceActivationUnavailableError("The first-request setup guide was dismissed for this workspace")

    async def _state_for_update(self, workspace_id: uuid.UUID) -> WorkspaceActivationState | None:
        """The workspace's row, locked for the rest of the transaction.

        ``FOR UPDATE`` so two tabs opening the guide at once serialize on the
        row rather than both rotating the key and leaving one of them holding a
        plaintext that no longer authenticates. A no-op on SQLite, which has no
        row locks and one writer anyway.
        """
        statement = (
            select(WorkspaceActivationState)
            .where(WorkspaceActivationState.workspace_id == workspace_id)
            .with_for_update()
        )
        return (await self.db.execute(statement)).scalar_one_or_none()

    async def _create_state(self, workspace_id: uuid.UUID) -> WorkspaceActivationState:
        """Insert the workspace's row, adopting a concurrent creator's instead of failing.

        Through a SAVEPOINT for the reason ``get_or_create_default_user`` uses
        one: losing the race must roll back this row alone, not the key rotation
        the caller has already staged.
        """
        state = WorkspaceActivationState(workspace_id=workspace_id)
        try:
            async with self.db.begin_nested():
                self.db.add(state)
        except IntegrityError:
            winner = await self._state_for_update(workspace_id)
            if winner is None:
                raise
            return winner
        return state

    async def _existing_key(self, state: WorkspaceActivationState | None) -> APIKey | None:
        """The key the guide issued for this workspace, if the row still exists.

        Null after someone deleted it from the Keys page, which is a legitimate
        thing to do: ``api_key_id`` is ``SET NULL``, so the next presentation
        mints a fresh row instead of resurrecting a deleted one.
        """
        if state is None or state.api_key_id is None:
            return None
        return await self.db.get(APIKey, state.api_key_id)

    async def _first_successful_request(self, workspace_id: uuid.UUID) -> UsageLog | None:
        """The oldest successful gateway request in the workspace, which is the activation."""
        statement = (
            select(UsageLog)
            .where(
                UsageLog.workspace_id == workspace_id,
                UsageLog.source == _GATEWAY_SOURCE,
                UsageLog.status == "success",
            )
            # Tie-broken on the id so two rows sharing a timestamp still name one
            # winner, rather than the receipt changing between two page loads.
            .order_by(UsageLog.timestamp.asc(), UsageLog.id.asc())
            .limit(1)
        )
        return (await self.db.execute(statement)).scalars().first()

    async def _latest_request(self, workspace_id: uuid.UUID) -> UsageLog | None:
        """The most recent gateway request in the workspace, successful or not."""
        statement = (
            select(UsageLog)
            .where(
                UsageLog.workspace_id == workspace_id,
                UsageLog.source == _GATEWAY_SOURCE,
                UsageLog.status.in_(_ATTEMPT_STATUSES),
            )
            .order_by(UsageLog.timestamp.desc(), UsageLog.id.desc())
            .limit(1)
        )
        return (await self.db.execute(statement)).scalars().first()


def _attempt_public(row: UsageLog) -> ActivationAttemptPublic:
    """Render one usage row as the attempt the guide reports."""
    succeeded = row.status == "success"
    return ActivationAttemptPublic(
        occurred_at=_utc_iso(row.timestamp),
        request_id=row.id,
        status="success" if succeeded else "failed",
        provider=row.provider,
        model=row.model,
        error_category=None if succeeded else activation_error_category(row.status_code),
        cost_usd=as_float(row.cost),
        latency_ms=row.latency_ms,
    )
