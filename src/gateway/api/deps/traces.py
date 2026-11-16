"""Who may read a workspace's span content, and the scope each trace route reads within."""

import uuid
from collections.abc import Collection
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.api.deps._base import (
    UnitOfWorkDep,
)
from gateway.core.config import (
    TRACE_CONTENT_HEADER,
    TRACE_CONTENT_KEEP,
    GatewayConfig,
)
from gateway.core.database import DATABASE_ERRORS, describe_database_error, get_db, release_session
from gateway.core.unit_of_work import UnitOfWork
from gateway.log_config import logger
from gateway.models.tenancy import MANAGEMENT_ROLES, Workspace
from gateway.repositories.tenancy import (
    OrganizationMemberRepository,
    UserRepository,
    WorkspaceMemberRepository,
    WorkspaceRepository,
)
from gateway.repositories.traces import TracesRepositories
from gateway.services.catalog import served_context_window
from gateway.services.traces import SpanContent, TraceService
from gateway.types.traces import (
    ContentCapture,
    SessionOwners,
    WorkspaceContent,
)


def build_trace_service(uow: UnitOfWork, content: SpanContent, owners: SessionOwners) -> TraceService:
    """Build the traces operations for a request or a worker job, sealing and opening content with ``content``."""
    return TraceService(uow, TracesRepositories.on(uow), content, owners, served_context_window)


async def is_workspace_member(
    db: AsyncSession, workspace_id: uuid.UUID, user_id: str | None, workspace: Workspace | None = None
) -> bool:
    """Whether ``user_id`` names a person who currently sees the workspace.

    An active owner or admin of its organization, or an active member of the
    workspace itself: the people whose own sessions' content they can read. A
    billing user no dashboard identity stands behind is none of them, and nor is
    a deactivated account, which can no longer sign in to read or manage it.
    """
    person = _person(user_id)
    if person is None:
        return False
    if workspace is None:
        workspace = await WorkspaceRepository(db).get(workspace_id)
    if workspace is None:
        return False
    account = await UserRepository(db).get(person)
    if account is None or not account.is_active:
        return False
    membership = await OrganizationMemberRepository(db).get_active_by_organization_and_user(
        workspace.organization_id, person
    )
    if membership is None:
        return False
    if membership.role in MANAGEMENT_ROLES:
        return True
    member = await WorkspaceMemberRepository(db).get_by_workspace_and_user(workspace_id, person)
    return member is not None and member.status == "active"


def _person(user_id: str | None) -> uuid.UUID | None:
    """The identity a trace's user id names, or None for one that names no person.

    The canonical spelling only: ownership compares the id as a string, so another
    spelling of the same id would keep content its owner could never open.
    """
    if user_id is None:
        return None
    try:
        person = uuid.UUID(user_id)
    except ValueError:
        return None
    return person if str(person) == user_id else None


async def workspace_members(
    db: AsyncSession, pairs: Collection[tuple[uuid.UUID, str | None]]
) -> set[tuple[uuid.UUID, str | None]]:
    """Which of these (workspace, user) pairs :func:`is_workspace_member` holds for, in four queries."""
    people = {pair: person for pair in pairs if (person := _person(pair[1])) is not None}
    if not people:
        return set()
    workspaces = {
        workspace.id: workspace.organization_id
        for workspace in await WorkspaceRepository(db).get_by_ids({workspace_id for workspace_id, _ in people})
    }
    active = await UserRepository(db).get_active_ids(set(people.values()))
    candidates = {pair: person for pair, person in people.items() if pair[0] in workspaces and person in active}
    roles = await OrganizationMemberRepository(db).get_active_roles(
        {workspaces[workspace_id] for workspace_id, _ in candidates}, set(candidates.values())
    )
    by_organization = {pair: roles.get((workspaces[pair[0]], person)) for pair, person in candidates.items()}
    explicit = await WorkspaceMemberRepository(db).get_active_pairs(
        {workspace_id for (workspace_id, _), role in by_organization.items() if role is not None},
        {candidates[pair] for pair, role in by_organization.items() if role is not None},
    )
    return {
        pair
        for pair, role in by_organization.items()
        if role is not None and (role in MANAGEMENT_ROLES or (pair[0], candidates[pair]) in explicit)
    }


async def workspaces_keeping_content_for(
    db: AsyncSession, pairs: Collection[tuple[uuid.UUID, str | None]]
) -> set[tuple[uuid.UUID, str | None]]:
    """Which of these (workspace, user) pairs may have content kept, in five queries.

    A current member of a workspace that still keeps what its sessions were about.
    The switch is read here rather than trusted from the export's admission, because
    turning it off and moving the workspace's content generation on are two commits:
    an export admitted between them carries a generation that is still current.
    """
    members = await workspace_members(db, pairs)
    if not members:
        return set()
    keeping = {
        workspace.id
        for workspace in await WorkspaceRepository(db).get_by_ids({workspace_id for workspace_id, _ in members})
        if workspace.trace_content_enabled
    }
    return {pair for pair in members if pair[0] in keeping}


class DatabaseSessionOwners:
    """:class:`SessionOwners` over this deployment's memberships, on the request's session."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def is_member(self, workspace_id: uuid.UUID, user_id: str | None) -> bool:
        return await is_workspace_member(self._db, workspace_id, user_id)

    async def content_kept_for(
        self, pairs: Collection[tuple[uuid.UUID, str | None]]
    ) -> set[tuple[uuid.UUID, str | None]]:
        return await workspaces_keeping_content_for(self._db, pairs)

    async def content_kept(self, workspace_id: uuid.UUID, user_id: str | None) -> WorkspaceContent:
        workspace = await WorkspaceRepository(self._db).get(workspace_id)
        if workspace is None or not workspace.trace_content_enabled:
            return WorkspaceContent()
        if not await is_workspace_member(self._db, workspace_id, user_id, workspace):
            return WorkspaceContent()
        return WorkspaceContent(keeps=True, tool_output=workspace.trace_content_tool_output)

    async def organization_of(self, workspace_id: uuid.UUID) -> uuid.UUID | None:
        workspace = await WorkspaceRepository(self._db).get(workspace_id)
        return workspace.organization_id if workspace is not None else None


class NoSessionOwners:
    """:class:`SessionOwners` for work that never opens a share: nobody is anybody's member."""

    async def is_member(self, workspace_id: uuid.UUID, user_id: str | None) -> bool:
        return False

    async def content_kept_for(
        self, pairs: Collection[tuple[uuid.UUID, str | None]]
    ) -> set[tuple[uuid.UUID, str | None]]:
        return set()

    async def content_kept(self, workspace_id: uuid.UUID, user_id: str | None) -> WorkspaceContent:
        return WorkspaceContent()

    async def organization_of(self, workspace_id: uuid.UUID) -> uuid.UUID | None:
        return None


async def workspace_keeps_trace_content(
    db: AsyncSession, config: GatewayConfig, workspace_id: uuid.UUID, user_id: str | None
) -> bool:
    """Whether what a span in this workspace was about is kept for this exporter.

    The deployment allows it, the workspace turned it on, and the key exports for a
    current member of the workspace: content nobody could ever read is not kept, so
    a key with no user, or one for a billing user no person stands behind, keeps none.

    Read fresh, not cached, so a workspace that turns it off keeps nothing from its next export on.
    Fails closed: where any of it cannot be read, nothing is kept. The transaction the reads
    open is ended at once, so no connection idles while the export's spans are handed on.
    """
    if not config.trace_content_enabled:
        return False
    try:
        keeps = (await DatabaseSessionOwners(db).content_kept(workspace_id, user_id)).keeps
        await release_session(db)
    except DATABASE_ERRORS as exc:
        logger.warning("Could not read whether a workspace keeps trace content (%s)", describe_database_error(exc))
        return False
    return keeps


TRACE_CONTENT_HEADER_INVALID_DETAIL = f"{TRACE_CONTENT_HEADER} must be {TRACE_CONTENT_KEEP!r}, or be left out"


def asks_to_keep_trace_content(request: Request) -> bool:
    """Whether the request asks for its content to be kept, refusing a value it cannot mean.

    A misspelled header is a caller mistake to report, as ``Otari-Code-Execution``'s
    is: otherwise ``Otari-Trace-Content: true`` would keep nothing and say nothing.
    """
    value = request.headers.get(TRACE_CONTENT_HEADER)
    if value is None or not value.strip():
        return False
    if value.strip().lower() != TRACE_CONTENT_KEEP:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=TRACE_CONTENT_HEADER_INVALID_DETAIL)
    return True


async def routed_trace_content(
    request: Request,
    uow: UnitOfWork,
    db: AsyncSession,
    config: GatewayConfig,
    workspace_id: uuid.UUID,
    user_id: str | None,
) -> ContentCapture | None:
    """What a routed request that asks for it (``Otari-Trace-Content: keep``) keeps of its content, or None.

    The deployment has to be able to seal content, and then the traces service
    applies the workspace's rule, on the request's own Unit of Work. Fails closed:
    a read that fails keeps no content, and the request is served without it.
    """
    if not config.trace_content_enabled:
        return None
    content: SpanContent | None = getattr(request.app.state, "trace_content", None)
    if content is None or not content.sealing:
        return None
    try:
        return await get_trace_service(uow, request, db).routed_content(
            workspace_id, user_id, tool_output=config.trace_content_tool_output
        )
    except DATABASE_ERRORS as exc:
        logger.warning("Could not read whether a routed request keeps its content (%s)", describe_database_error(exc))
        return None


def get_trace_service(
    uow: UnitOfWorkDep, request: Request, db: Annotated[AsyncSession, Depends(get_db)]
) -> TraceService:
    """Build the request's traces service on the request's Unit of Work, with the process's content keys."""
    content: SpanContent = getattr(request.app.state, "trace_content", None) or SpanContent.disabled()
    return build_trace_service(uow, content, DatabaseSessionOwners(db))


TraceServiceDep = Annotated[TraceService, Depends(get_trace_service)]
