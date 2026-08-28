"""The caller's own organization's usage, for a tenant who does not operate the deployment.

``/v1/usage`` is deployment-wide and has been operator-only since #821, which
was right: it reads every tenant's rows and its ``workspace_id`` parameter is a
filter the client supplies, so nothing but the operator gate stands between a
signed-in member and another organization's traffic. On a deployment serving
mutually-untrusting tenants that left "show me my organization's usage", the
ordinary case, with no endpoint a member is allowed to call at all
(mozilla-ai/otari#837).

The answer is not a looser gate on that router. A mode that widened
``/v1/usage`` for a non-operator would inherit its client-supplied
``workspace_id`` and rebuild the escalation #821 closed. So the deployment-wide
routes keep the gate they have, unchanged, and this router is a second, narrower
reading of the same rows:

* **Scope is derived, never accepted.** It comes from the caller's own
  ``active_organization_id`` by way of ``resolve_visible_workspace_scope``,
  which refuses a pointer with no live membership behind it. No request here
  names an organization. Moving between organizations is
  ``POST /v1/organizations/me/switch``, which 404s on one the caller does not
  belong to.
* **How much of the organization** follows the rule the workspace list already
  uses: an owner, an admin or a superuser reads every workspace in it, and a
  member or viewer reads the ones they actively belong to. A member who belongs
  to no workspace gets an empty page, not a refusal: the surface is theirs and
  simply has nothing in it yet.
* **``workspace_id`` still narrows, and cannot widen.** It is put through
  ``resolve_workspace_in_organization``, the same resolver every other
  workspace-scoped read uses, so a workspace outside the caller's scope answers
  404 exactly as a workspace that does not exist does.

Reads only. Deleting rows and repricing them stay deployment-wide, as does
``/v1/usage/in-flight``: its registry entries carry no workspace at all
(see ``usage.list_in_flight``), so there is nothing there to scope yet.

Every aggregation is the one ``usage.py`` already runs. This module contributes
route declarations and a scope predicate, and nothing that could compute a
different answer to the same question.
"""

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import ColumnElement, false, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from gateway.api.deps import CurrentIdentity, get_db, verify_master_key
from gateway.api.routes.usage import (
    _API_KEY_MULTI_DESC,
    _COUNTS_DESC,
    _DIMENSIONS_DESC,
    _END_DESC,
    _ENDPOINT_DESC,
    _MAX_REQUEST_GROUPS,
    _MODEL_MULTI_DESC,
    _PRICED_DESC,
    _PROVIDER_DESC,
    _REQUEST_GROUP_DESC,
    _SOURCE_DESC,
    _SOURCE_LABEL_DESC,
    _START_DESC,
    _STATUS_CODE_DESC,
    _STATUS_DESC,
    _TOOL_DESC,
    _USER_MULTI_DESC,
    _WORKSPACE_DESC,
    Bucket,
    SeriesGroupBy,
    SummaryDimension,
    ToolFilter,
    UsageCount,
    UsageEntry,
    UsageGroupedSeries,
    UsageSummary,
    _grouped_series_response,
    _summary_context,
    _summary_response,
    _usage_filters,
)
from gateway.core.sql import MAX_FILTER_VALUES
from gateway.models.entities import APIKey, UsageLog, User
from gateway.models.tenancy import User as TenancyUser
from gateway.models.tenancy import Workspace
from gateway.services.tenancy import OrganizationService
from gateway.services.tenancy.authorization import (
    resolve_visible_workspace_scope,
    resolve_workspace_in_organization,
)

router = APIRouter(
    prefix="/v1/organizations/me/usage",
    tags=["organization-usage"],
    # Authentication only, like the rest of the ``/v1/organizations/me`` surface.
    # What the caller may read is decided per request by the scope below, which
    # is the pattern the tenant-scoped routers already follow and the reason the
    # deployment operator gate does not belong here.
    dependencies=[Depends(verify_master_key)],
)


async def _scope_condition(
    db: AsyncSession,
    *,
    user: TenancyUser,
    workspace_id: uuid.UUID | None,
) -> ColumnElement[bool]:
    """The WHERE clause that confines a read to what this caller may see.

    Resolved per request rather than cached on the session: a membership can be
    suspended or a role changed between two requests, and the cheaper answer is
    the one that goes stale in the unsafe direction.
    """
    organizations = OrganizationService(db)

    if workspace_id is not None:
        # Only the organization is resolved on this branch. The full scope would
        # also build the caller's workspace-id set, which is deliberately
        # unpaged, and the branch below discards it: the Activity page issues a
        # list, a count, a summary and a series per filter change, so resolving
        # it here would be four unbounded reads an interaction, for nothing.
        # ``get_active_organization_for_user`` is the same refusal the full scope
        # opens with, so a pointer with no live membership behind it still stops
        # here rather than reaching the resolver below.
        organization = await organizations.get_active_organization_for_user(user)
        # Narrowing only. The resolver raises ``WorkspaceNotFoundError`` (404) for
        # a workspace in another organization *and* for one in this organization
        # the caller is not a member of, which is what keeps the parameter from
        # being an existence oracle either way.
        #
        # The equality returned below duplicates the one ``_usage_filters`` builds
        # from the same parameter, and that is deliberate: the scope has to be
        # sufficient on its own, so a later change to how the filter is applied
        # cannot leave a request scoped by nothing.
        await resolve_workspace_in_organization(
            db,
            user=user,
            workspace_id=workspace_id,
            organization=organization,
            organizations=organizations,
        )
        return col(UsageLog.workspace_id) == workspace_id

    scope = await resolve_visible_workspace_scope(db, user=user, organizations=organizations)
    if scope.sees_every_workspace:
        return col(UsageLog.workspace_id).in_(
            select(col(Workspace.id)).where(col(Workspace.organization_id) == scope.organization.id)
        )
    if not scope.workspace_ids:
        # Belongs to no workspace yet. An empty result, and deliberately not a
        # 403: nothing was refused, there is simply nothing here.
        return false()
    return col(UsageLog.workspace_id).in_(scope.workspace_ids)


@router.get("")
async def list_organization_usage(
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    user_id: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_USER_MULTI_DESC)] = None,
    status: str | None = Query(default=None, description=_STATUS_DESC),
    status_code: int | None = Query(default=None, description=_STATUS_CODE_DESC),
    model: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_MODEL_MULTI_DESC)] = None,
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
    provider: str | None = Query(default=None, description=_PROVIDER_DESC),
    source: str | None = Query(default=None, description=_SOURCE_DESC),
    source_label: str | None = Query(default=None, description=_SOURCE_LABEL_DESC),
    api_key_id: Annotated[
        list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_API_KEY_MULTI_DESC)
    ] = None,
    priced: bool | None = Query(default=None, description=_PRICED_DESC),
    tool: ToolFilter | None = Query(default=None, description=_TOOL_DESC),
    counts_toward_budget: bool | None = Query(default=None, description=_COUNTS_DESC),
    request_group_id: Annotated[
        list[str] | None, Query(max_length=_MAX_REQUEST_GROUPS, description=_REQUEST_GROUP_DESC)
    ] = None,
    workspace_id: Annotated[uuid.UUID | None, Query(description=_WORKSPACE_DESC)] = None,
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[UsageEntry]:
    """List the caller's organization's usage logs, most recent first.

    The tenant-scoped counterpart of ``GET /v1/usage``: same filters, same bare
    JSON array, same separate ``/count`` for a paginator's total, confined to
    what the caller's membership lets them see. Scope is never a parameter here.
    """
    conditions = _usage_filters(
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        status=status,
        status_code=status_code,
        model=model,
        endpoint=endpoint,
        provider=provider,
        source=source,
        source_label=source_label,
        api_key_id=api_key_id,
        priced=priced,
        tool=tool,
        counts_toward_budget=counts_toward_budget,
        request_group_id=request_group_id,
        workspace_id=workspace_id,
        scope=await _scope_condition(db, user=identity, workspace_id=workspace_id),
    )
    stmt = (
        select(UsageLog, User.alias, APIKey.key_name)
        .outerjoin(User, User.user_id == UsageLog.user_id)
        .outerjoin(APIKey, APIKey.id == UsageLog.api_key_id)
        .where(*conditions)
        .order_by(UsageLog.timestamp.desc())
        .offset(skip)
        .limit(limit)
    )
    result = await db.execute(stmt)
    return [
        UsageEntry.from_model(log, user_alias=alias, api_key_name=key_name) for log, alias, key_name in result.all()
    ]


@router.get("/count")
async def count_organization_usage(
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    user_id: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_USER_MULTI_DESC)] = None,
    status: str | None = Query(default=None, description=_STATUS_DESC),
    status_code: int | None = Query(default=None, description=_STATUS_CODE_DESC),
    model: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_MODEL_MULTI_DESC)] = None,
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
    provider: str | None = Query(default=None, description=_PROVIDER_DESC),
    source: str | None = Query(default=None, description=_SOURCE_DESC),
    source_label: str | None = Query(default=None, description=_SOURCE_LABEL_DESC),
    api_key_id: Annotated[
        list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_API_KEY_MULTI_DESC)
    ] = None,
    priced: bool | None = Query(default=None, description=_PRICED_DESC),
    tool: ToolFilter | None = Query(default=None, description=_TOOL_DESC),
    counts_toward_budget: bool | None = Query(default=None, description=_COUNTS_DESC),
    request_group_id: Annotated[
        list[str] | None, Query(max_length=_MAX_REQUEST_GROUPS, description=_REQUEST_GROUP_DESC)
    ] = None,
    workspace_id: Annotated[uuid.UUID | None, Query(description=_WORKSPACE_DESC)] = None,
) -> UsageCount:
    """Total rows matching these filters, within the caller's scope.

    Serves the paginator's "N of M" beside the list above, and is scoped the
    same way, so the total can never describe more rows than the list will show.
    """
    conditions = _usage_filters(
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        status=status,
        status_code=status_code,
        model=model,
        endpoint=endpoint,
        provider=provider,
        source=source,
        source_label=source_label,
        api_key_id=api_key_id,
        priced=priced,
        tool=tool,
        counts_toward_budget=counts_toward_budget,
        request_group_id=request_group_id,
        workspace_id=workspace_id,
        scope=await _scope_condition(db, user=identity, workspace_id=workspace_id),
    )
    stmt: Any = select(func.count()).select_from(UsageLog).where(*conditions)
    return UsageCount(total=(await db.execute(stmt)).scalar_one())


@router.get("/summary")
async def organization_usage_summary(
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    user_id: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_USER_MULTI_DESC)] = None,
    status: str | None = Query(default=None, description=_STATUS_DESC),
    status_code: int | None = Query(default=None, description=_STATUS_CODE_DESC),
    model: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_MODEL_MULTI_DESC)] = None,
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
    provider: str | None = Query(default=None, description=_PROVIDER_DESC),
    source: str | None = Query(default=None, description=_SOURCE_DESC),
    source_label: str | None = Query(default=None, description=_SOURCE_LABEL_DESC),
    api_key_id: Annotated[
        list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_API_KEY_MULTI_DESC)
    ] = None,
    priced: bool | None = Query(default=None, description=_PRICED_DESC),
    tool: ToolFilter | None = Query(default=None, description=_TOOL_DESC),
    counts_toward_budget: bool | None = Query(default=None, description=_COUNTS_DESC),
    workspace_id: Annotated[uuid.UUID | None, Query(description=_WORKSPACE_DESC)] = None,
    bucket: Bucket = Query(default="day", description="Time-series granularity: 'hour' or 'day'"),
    dimensions: list[SummaryDimension] | None = Query(default=None, description=_DIMENSIONS_DESC),
) -> UsageSummary:
    """Aggregate spend, tokens and request volume for the caller's organization.

    The tenant-scoped counterpart of ``GET /v1/usage/summary``, running the same
    aggregation over a narrower row set: the same bounded window, the same
    breakdowns, the same ``dimensions`` selector for paying only for the passes a
    caller reads. The breakdown by user names the people inside the caller's own
    scope, which is the roster they can already read.
    """
    start, end, conditions, totals = await _summary_context(
        db,
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        status=status,
        status_code=status_code,
        model=model,
        endpoint=endpoint,
        provider=provider,
        source=source,
        source_label=source_label,
        api_key_id=api_key_id,
        priced=priced,
        tool=tool,
        counts_toward_budget=counts_toward_budget,
        workspace_id=workspace_id,
        scope=await _scope_condition(db, user=identity, workspace_id=workspace_id),
    )
    return await _summary_response(
        db,
        start=start,
        end=end,
        conditions=conditions,
        totals=totals,
        status=status,
        bucket=bucket,
        dimensions=dimensions,
    )


@router.get("/series")
async def organization_usage_series(
    identity: CurrentIdentity,
    db: Annotated[AsyncSession, Depends(get_db)],
    group_by: SeriesGroupBy = Query(description="Dimension to split the series by"),
    start_date: datetime | None = Query(default=None, description=_START_DESC),
    end_date: datetime | None = Query(default=None, description=_END_DESC),
    user_id: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_USER_MULTI_DESC)] = None,
    status: str | None = Query(default=None, description=_STATUS_DESC),
    status_code: int | None = Query(default=None, description=_STATUS_CODE_DESC),
    model: Annotated[list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_MODEL_MULTI_DESC)] = None,
    endpoint: str | None = Query(default=None, description=_ENDPOINT_DESC),
    provider: str | None = Query(default=None, description=_PROVIDER_DESC),
    source: str | None = Query(default=None, description=_SOURCE_DESC),
    source_label: str | None = Query(default=None, description=_SOURCE_LABEL_DESC),
    api_key_id: Annotated[
        list[str] | None, Query(max_length=MAX_FILTER_VALUES, description=_API_KEY_MULTI_DESC)
    ] = None,
    priced: bool | None = Query(default=None, description=_PRICED_DESC),
    tool: ToolFilter | None = Query(default=None, description=_TOOL_DESC),
    counts_toward_budget: bool | None = Query(default=None, description=_COUNTS_DESC),
    workspace_id: Annotated[uuid.UUID | None, Query(description=_WORKSPACE_DESC)] = None,
    bucket: Bucket = Query(default="day", description="Time-series granularity: 'hour' or 'day'"),
) -> UsageGroupedSeries:
    """Time series split by one dimension, for the caller's organization.

    The tenant-scoped counterpart of ``GET /v1/usage/series``, and kept in
    lockstep with the summary above for the reason that endpoint gives: the
    dashboard serializes one filter object for both, so a filter one of them
    ignored would make the stacked chart disagree with the tiles beside it.
    """
    start, end, conditions, totals = await _summary_context(
        db,
        start_date=start_date,
        end_date=end_date,
        user_id=user_id,
        status=status,
        status_code=status_code,
        model=model,
        endpoint=endpoint,
        provider=provider,
        source=source,
        source_label=source_label,
        api_key_id=api_key_id,
        priced=priced,
        tool=tool,
        counts_toward_budget=counts_toward_budget,
        workspace_id=workspace_id,
        scope=await _scope_condition(db, user=identity, workspace_id=workspace_id),
    )
    return await _grouped_series_response(
        db,
        start=start,
        end=end,
        conditions=conditions,
        totals=totals,
        status=status,
        bucket=bucket,
        group_by=group_by,
    )
