"""Members asking a workspace's managers for a server of the MCP catalog, and the managers' answer.

Any member who can see a workspace may ask, a viewer included: asking changes nothing until a
manager answers. A manager answers by adding the server, which approves every pending request for
it however the server was added, or by declining. Mail is optional: where the deployment can send
it, the managers hear of a new request and the requester of the answer, and where it cannot the
requests are still listed for both. Either way the requester is shown the answer until they
acknowledge it.
"""

import uuid
from collections.abc import Awaitable, Callable, Collection, Sequence
from datetime import UTC, datetime, timedelta
from typing import Protocol

from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.tools_exceptions import (
    McpCatalogServerNotListedError,
    McpServerAlreadyAddedError,
    McpServerAlreadyRequestedError,
    McpServerRequestDecidedError,
    McpServerRequestLimitError,
    McpServerRequestNotFoundError,
    McpServerRequestPendingError,
)
from gateway.log_config import logger
from gateway.models.tenancy import User, Workspace
from gateway.models.tools import McpServerRequest, McpServerRequestStatus
from gateway.repositories.tools import (
    McpServerRequestConflict,
    McpServerRequestRepository,
    WorkspaceMcpServerRepository,
)
from gateway.schemas.mcp_catalog import McpCatalogServerPublic
from gateway.schemas.tools import (
    McpServerRequestCreate,
    McpServerRequestDecline,
    McpServerRequesterPublic,
    McpServerRequestGroupPublic,
    McpServerRequestGroupsPublic,
    McpServerRequestPublic,
    McpServerRequestsDeclined,
    McpServerRequestsPublic,
    McpServerRequestStatusValue,
)
from gateway.services.mail import Mailer, MailMessage, render_email

# How many requests one member may have waiting in a workspace, and how many they may make there in
# a day. The second also bounds the mail a member can cause: each request mails the managers once.
MAX_PENDING_REQUESTS_PER_MEMBER = 10
MAX_REQUESTS_PER_MEMBER_PER_DAY = 10
# The managers one request is mailed to, the workspace's own first.
MAX_REQUEST_MAIL_RECIPIENTS = 20
# What a manager's list reads. Past it the count still says how many wait.
MAX_PENDING_LISTED = 500
_MAX_OWN_LISTED = 50

SERVERS_PAGE = "/#/tools/mcp-servers"

# Runs a coroutine once the response is on its way, so slow mail never holds a request.
SendLater = Callable[[Callable[[], Awaitable[None]]], object]


class RequestWorkspaces(Protocol):
    """What this service needs of the tenancy domain."""

    async def resolve_visible_workspace(self, *, user: User, workspace_id: uuid.UUID) -> Workspace: ...

    async def require_workspace_management_access(self, *, user: User, workspace: Workspace) -> None: ...

    async def lock_workspace(self, workspace_id: uuid.UUID) -> None: ...

    async def workspace_managers(self, workspace: Workspace) -> list[User]: ...

    async def people(self, user_ids: Collection[uuid.UUID]) -> list[User]: ...


class CatalogEntries(Protocol):
    """What this service needs of the mcp-catalog domain: the entry a request names."""

    async def servers_by_names(self, names: Sequence[str]) -> dict[str, McpCatalogServerPublic]: ...


def _spellings(url: str) -> tuple[str, str]:
    """``url`` without and with a trailing slash, the two spellings of one endpoint."""
    bare = url.strip().rstrip("/")
    return bare, f"{bare}/"


def _person(user: User | None) -> str:
    if user is None:
        return "A member"
    return user.full_name or user.email or "A member"


def _own(row: McpServerRequest, entry: McpCatalogServerPublic | None) -> McpServerRequestPublic:
    status: McpServerRequestStatusValue = McpServerRequestStatus(row.status).value
    return McpServerRequestPublic(
        id=row.id,
        workspace_id=row.workspace_id,
        catalog_name=row.catalog_name,
        title=row.title,
        url=row.url,
        note=row.note,
        status=status,
        decision_reason=row.decision_reason,
        decided_at=row.decided_at,
        server_id=row.server_id,
        catalog_server=entry,
        created_at=row.created_at,
    )


def render_request_mail(
    *, requester: str, title: str, workspace: str, note: str | None, link: str, asset_root: str | None
) -> MailMessage:
    """What a workspace's managers are sent when a member asks for a server."""
    return render_email(
        "mcp_server_request",
        subject="{{REQUESTER}} asked for {{SERVER}} in {{WORKSPACE}}",
        preheader="{{INTRO}}",
        asset_root=asset_root,
        values={
            "REQUESTER": requester,
            "SERVER": title,
            "WORKSPACE": workspace,
            "HEADLINE": f"{requester} asked for {title}",
            "INTRO": f"They would like the MCP server {title} added to the workspace {workspace}.",
            "NOTE": note or "They left no note.",
            "SERVERS_LINK": link,
        },
    )


def render_decision_mail(
    *, approved: bool, title: str, workspace: str, reason: str | None, link: str, asset_root: str | None
) -> MailMessage:
    """What a requester is sent when their request is answered."""
    if approved:
        headline = f"{title} was added to {workspace}"
        intro = f"An admin added the MCP server {title} you asked for."
        detail = "Open MCP servers to use it. A Personal server asks you to connect your own account first."
    else:
        headline = f"Your request for {title} was declined"
        intro = f"An admin of {workspace} declined to add the MCP server {title}."
        detail = f"Their reason: {reason}" if reason else "They gave no reason."
    return render_email(
        "mcp_server_request_decided",
        subject="{{HEADLINE}}",
        preheader="{{INTRO}}",
        asset_root=asset_root,
        values={"HEADLINE": headline, "INTRO": intro, "DETAIL": detail, "SERVERS_LINK": link},
    )


class McpServerRequestService:
    """Takes members' requests for catalog servers, lists them, and records the managers' answers."""

    def __init__(
        self,
        uow: UnitOfWork,
        requests: McpServerRequestRepository,
        servers: WorkspaceMcpServerRepository,
        workspaces: RequestWorkspaces,
        catalog: CatalogEntries,
        mailer: Mailer,
        *,
        send_later: SendLater,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._uow = uow
        self._requests = requests
        self._servers = servers
        self._workspaces = workspaces
        self._catalog = catalog
        self._mailer = mailer
        self._send_later = send_later
        self._clock = clock

    async def create(
        self, *, user: User, workspace_id: uuid.UUID, request: McpServerRequestCreate
    ) -> McpServerRequestPublic:
        """Ask the workspace's managers for a catalog server, and mail them where mail can be sent.

        Raises:
            McpCatalogServerNotListedError: the catalog lists no server by that name.
            McpServerAlreadyAddedError: the workspace has a server at that URL.
            McpServerAlreadyRequestedError: the caller's request for it is already waiting.
            McpServerRequestLimitError: the caller has too many requests waiting, or made too many today.
        """
        workspace = await self._workspaces.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        entry = (await self._catalog.servers_by_names([request.catalog_name])).get(request.catalog_name)
        if entry is None:
            raise McpCatalogServerNotListedError(request.catalog_name)
        note = (request.note or "").strip() or None
        now = self._clock()
        async with self._uow:
            # Serializes with a manager adding the server, and makes the two counts below exact.
            await self._workspaces.lock_workspace(workspace.id)
            stored = {
                spelling
                for url in (await self._servers.urls_in_workspace(workspace.id)).values()
                for spelling in _spellings(url)
            }
            if stored.intersection(_spellings(entry.url)):
                raise McpServerAlreadyAddedError
            if await self._requests.has_pending(workspace.id, entry.name, user.id):
                raise McpServerAlreadyRequestedError
            if await self._requests.count_for_requester(workspace.id, user.id) >= MAX_PENDING_REQUESTS_PER_MEMBER:
                raise McpServerRequestLimitError(
                    f"You have {MAX_PENDING_REQUESTS_PER_MEMBER} requests waiting; withdraw one, or wait for an answer"
                )
            made_today = await self._requests.count_for_requester(workspace.id, user.id, since=now - timedelta(days=1))
            if made_today >= MAX_REQUESTS_PER_MEMBER_PER_DAY:
                raise McpServerRequestLimitError(
                    f"You asked for {MAX_REQUESTS_PER_MEMBER_PER_DAY} servers in the last day; try again tomorrow"
                )
            try:
                row = await self._requests.add(
                    workspace_id=workspace.id,
                    catalog_name=entry.name,
                    title=entry.title,
                    url=entry.url,
                    requester_id=user.id,
                    note=note,
                )
            except McpServerRequestConflict:
                # Where the workspace lock does not serialize (SQLite), the unique index still decides.
                raise McpServerAlreadyRequestedError from None
            # Asking again replaces an earlier decline: only the new request is shown from here on.
            await self._requests.acknowledge_declined(
                workspace.id, at=now, catalog_name=entry.name, requester_id=user.id
            )
            public = _own(row, entry)
        logger.info("MCP server requested workspace_id=%s request_id=%s", workspace.id, public.id)
        await self._mail_managers(workspace, requester=user, title=entry.title, note=note)
        return public

    async def mine(self, *, user: User, workspace_id: uuid.UUID) -> McpServerRequestsPublic:
        """The caller's own pending requests in a workspace, and the answers they have not acknowledged."""
        workspace = await self._workspaces.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        async with self._uow:
            rows = await self._requests.for_requester(workspace.id, user.id, limit=_MAX_OWN_LISTED)
        entries = await self._catalog.servers_by_names([row.catalog_name for row in rows])
        return McpServerRequestsPublic(data=[_own(row, entries.get(row.catalog_name)) for row in rows])

    async def withdraw(self, *, user: User, workspace_id: uuid.UUID, request_id: uuid.UUID) -> None:
        """Withdraw one of the caller's own pending requests.

        Raises:
            McpServerRequestNotFoundError: no such request in the workspace, or it is someone else's.
            McpServerRequestDecidedError: it was already answered or withdrawn.
        """
        workspace = await self._workspaces.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        async with self._uow:
            row = await self._requests.find(workspace.id, request_id)
            if row is None or row.requester_id != user.id:
                raise McpServerRequestNotFoundError(request_id)
            if row.status != McpServerRequestStatus.PENDING:
                raise McpServerRequestDecidedError
            row.status = McpServerRequestStatus.WITHDRAWN
            row.decided_at = self._clock()
            await self._requests.save()
        logger.info("MCP server request withdrawn workspace_id=%s request_id=%s", workspace.id, request_id)

    async def acknowledge(self, *, user: User, workspace_id: uuid.UUID, request_id: uuid.UUID) -> None:
        """Record that the caller has seen the answer to one of their own requests. Repeating it changes nothing.

        Raises:
            McpServerRequestNotFoundError: no such request in the workspace, or it is someone else's.
            McpServerRequestPendingError: it has no answer yet.
        """
        workspace = await self._workspaces.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        async with self._uow:
            row = await self._requests.find(workspace.id, request_id)
            if row is None or row.requester_id != user.id:
                raise McpServerRequestNotFoundError(request_id)
            if row.status == McpServerRequestStatus.PENDING:
                raise McpServerRequestPendingError
            if row.acknowledged_at is None:
                row.acknowledged_at = self._clock()
                await self._requests.save()

    async def pending(self, *, user: User, workspace_id: uuid.UUID) -> McpServerRequestGroupsPublic:
        """Every pending request of a workspace, grouped by the server asked for. Workspace managers only."""
        workspace = await self._workspaces.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        await self._workspaces.require_workspace_management_access(user=user, workspace=workspace)
        async with self._uow:
            rows = await self._requests.pending_in_workspace(workspace.id, limit=MAX_PENDING_LISTED)
            count = len(rows)
            if count == MAX_PENDING_LISTED:
                count = await self._requests.count_pending_in_workspace(workspace.id)
        people = {person.id: person for person in await self._workspaces.people({row.requester_id for row in rows})}
        entries = await self._catalog.servers_by_names(list({row.catalog_name for row in rows}))
        groups: dict[str, McpServerRequestGroupPublic] = {}
        for row in rows:
            group = groups.get(row.catalog_name)
            if group is None:
                group = groups[row.catalog_name] = McpServerRequestGroupPublic(
                    catalog_name=row.catalog_name,
                    title=row.title,
                    url=row.url,
                    catalog_server=entries.get(row.catalog_name),
                    requesters=[],
                    first_requested_at=row.created_at,
                )
            person = people.get(row.requester_id)
            group.requesters.append(
                McpServerRequesterPublic(
                    request_id=row.id,
                    user_id=row.requester_id,
                    full_name=person.full_name if person is not None else None,
                    email=person.email if person is not None else None,
                    note=row.note,
                    requested_at=row.created_at,
                )
            )
        return McpServerRequestGroupsPublic(data=list(groups.values()), count=count)

    async def decline(
        self, *, user: User, workspace_id: uuid.UUID, request: McpServerRequestDecline
    ) -> McpServerRequestsDeclined:
        """Decline every pending request for one catalog server, and tell each requester. Workspace managers only.

        Raises:
            McpServerRequestNotFoundError: nobody is waiting for that server in the workspace.
        """
        workspace = await self._workspaces.resolve_visible_workspace(user=user, workspace_id=workspace_id)
        await self._workspaces.require_workspace_management_access(user=user, workspace=workspace)
        reason = (request.reason or "").strip() or None
        async with self._uow:
            rows = await self._requests.pending_for_server(workspace.id, catalog_name=request.catalog_name)
            if not rows:
                raise McpServerRequestNotFoundError(request.catalog_name)
            answered = self._answer(rows, McpServerRequestStatus.DECLINED, decided_by=user.id, reason=reason)
            await self._requests.save()
        logger.info("MCP server requests declined workspace_id=%s count=%d", workspace.id, len(answered))
        await self._mail_requesters(workspace.name, answered, approved=False, reason=reason, decided_by=user.id)
        return McpServerRequestsDeclined(declined=len(answered))

    async def server_added(
        self, *, workspace_id: uuid.UUID, workspace_name: str, server_id: uuid.UUID, url: str, added_by: uuid.UUID
    ) -> int:
        """Approve every pending request for the server a workspace just stored, and tell each requester.

        Matched by URL, so it holds however the server was added: from a request, from the catalog,
        or by pasting its URL. Runs in the transaction that stores the server and commits it.
        Returns how many requests it approved.
        """
        async with self._uow:
            rows = await self._requests.pending_for_server(workspace_id, urls=_spellings(url))
            answered = self._answer(rows, McpServerRequestStatus.APPROVED, decided_by=added_by, server_id=server_id)
            if rows:
                await self._requests.save()
            # The server is here now, so an earlier decline of it no longer says anything true.
            await self._requests.acknowledge_declined(workspace_id, at=self._clock(), urls=_spellings(url))
        if answered:
            logger.info("MCP server requests approved workspace_id=%s count=%d", workspace_id, len(answered))
            await self._mail_requesters(workspace_name, answered, approved=True, reason=None, decided_by=added_by)
        return len(answered)

    def _answer(
        self,
        rows: Sequence[McpServerRequest],
        status: McpServerRequestStatus,
        *,
        decided_by: uuid.UUID,
        reason: str | None = None,
        server_id: uuid.UUID | None = None,
    ) -> list[tuple[uuid.UUID, str]]:
        """Record one answer on these rows. Returns each requester and the title they asked for."""
        now = self._clock()
        for row in rows:
            row.status = status
            row.decided_by = decided_by
            row.decided_at = now
            row.decision_reason = reason
            row.server_id = server_id
        return [(row.requester_id, row.title) for row in rows]

    def _send(self, deliveries: list[tuple[str, MailMessage]]) -> None:
        """Hand these messages to the mailer once the response is on its way. A failed one is the mailer's to log."""
        if not deliveries:
            return
        mailer = self._mailer

        async def send() -> None:
            for to, message in deliveries:
                await mailer.send(to=to, message=message)

        self._send_later(send)

    async def _mail_managers(self, workspace: Workspace, *, requester: User, title: str, note: str | None) -> None:
        if not self._mailer.can_send_links:
            return
        managers = await self._workspaces.workspace_managers(workspace)
        message = render_request_mail(
            requester=_person(requester),
            title=title,
            workspace=workspace.name,
            note=note,
            link=self._mailer.link(SERVERS_PAGE),
            asset_root=self._mailer.asset_root,
        )
        recipients = [manager.email for manager in managers if manager.email and manager.id != requester.id]
        self._send([(email, message) for email in recipients[:MAX_REQUEST_MAIL_RECIPIENTS]])

    async def _mail_requesters(
        self,
        workspace_name: str,
        answered: Sequence[tuple[uuid.UUID, str]],
        *,
        approved: bool,
        reason: str | None,
        decided_by: uuid.UUID,
    ) -> None:
        if not answered or not self._mailer.can_send_links:
            return
        people = {
            person.id: person for person in await self._workspaces.people({requester for requester, _ in answered})
        }
        deliveries: list[tuple[str, MailMessage]] = []
        for requester_id, title in answered:
            person = people.get(requester_id)
            # Whoever answered knows the answer.
            if person is None or not person.email or not person.is_active or requester_id == decided_by:
                continue
            message = render_decision_mail(
                approved=approved,
                title=title,
                workspace=workspace_name,
                reason=reason,
                link=self._mailer.link(SERVERS_PAGE),
                asset_root=self._mailer.asset_root,
            )
            deliveries.append((person.email, message))
        self._send(deliveries)


__all__ = [
    "MAX_PENDING_REQUESTS_PER_MEMBER",
    "MAX_REQUESTS_PER_MEMBER_PER_DAY",
    "MAX_REQUEST_MAIL_RECIPIENTS",
    "CatalogEntries",
    "McpServerRequestService",
    "RequestWorkspaces",
    "SendLater",
    "render_decision_mail",
    "render_request_mail",
]
