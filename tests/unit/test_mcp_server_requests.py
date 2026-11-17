"""Members' MCP server requests without a database: the mail they render, and who a decision is mailed to."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable, Collection, Sequence
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from sqlalchemy.exc import OperationalError

from gateway.core.unit_of_work import UnitOfWork
from gateway.models.tenancy import User
from gateway.models.tools import McpServerRequest, McpServerRequestStatus
from gateway.repositories.tools import McpServerRequestRepository, WorkspaceMcpServerRepository
from gateway.schemas.mcp_catalog import McpCatalogServerPublic
from gateway.services.mail import Mailer, MailMessage
from gateway.services.tools._mcp_server_requests import (
    McpServerRequestService,
    _spellings,
    render_decision_mail,
    render_request_mail,
)

LINK = "https://otari.example/#/tools/mcp-servers"


def test_the_request_mail_names_who_asked_for_what_and_carries_the_note() -> None:
    message = render_request_mail(
        requester="Ada Lovelace",
        title="Sentry",
        workspace="Research",
        note="Triage <prod> errors",
        link=LINK,
        asset_root=None,
    )

    assert message.subject == "Ada Lovelace asked for Sentry in Research"
    assert "Triage <prod> errors" in message.text
    assert LINK in message.text
    # The note is a member's free text, so the HTML body escapes it.
    assert "Triage &lt;prod&gt; errors" in message.html
    assert "<prod>" not in message.html


def test_a_request_with_no_note_says_so_and_a_name_cannot_break_the_subject() -> None:
    message = render_request_mail(
        requester="Ada\r\nBcc: someone@example.com",
        title="Sentry",
        workspace="Research",
        note=None,
        link=LINK,
        asset_root=None,
    )

    assert "\n" not in message.subject and "\r" not in message.subject
    assert "They left no note." in message.text


@pytest.mark.parametrize(
    ("approved", "reason", "subject", "said"),
    [
        (True, None, "Sentry was added to Research", "connect your own account"),
        (False, "We use Datadog", "Your request for Sentry was declined", "Their reason: We use Datadog"),
        (False, None, "Your request for Sentry was declined", "They gave no reason."),
    ],
)
def test_the_decision_mail_says_what_was_decided(approved: bool, reason: str | None, subject: str, said: str) -> None:
    message = render_decision_mail(
        approved=approved, title="Sentry", workspace="Research", reason=reason, link=LINK, asset_root=None
    )

    assert message.subject == subject
    assert said in message.text
    assert LINK in message.text


def test_a_url_is_matched_with_and_without_its_trailing_slash() -> None:
    assert _spellings("https://mcp.example.com/mcp/") == ("https://mcp.example.com/mcp", "https://mcp.example.com/mcp/")
    assert _spellings(" https://mcp.example.com ") == ("https://mcp.example.com", "https://mcp.example.com/")


class _Session:
    """What a Unit of Work needs of a session when nothing reads it."""

    def __init__(self) -> None:
        self.commits = 0

    async def commit(self) -> None:
        self.commits += 1

    async def rollback(self) -> None:
        pass


class _Requests:
    def __init__(self, rows: list[McpServerRequest]) -> None:
        self.rows = rows
        self.asked: list[dict[str, Any]] = []

    async def pending_for_server(
        self, workspace_id: uuid.UUID, *, catalog_name: str | None = None, urls: Collection[str] = ()
    ) -> list[McpServerRequest]:
        self.asked.append({"workspace_id": workspace_id, "urls": tuple(urls)})
        return [row for row in self.rows if row.url in urls]

    async def acknowledge_declined(self, workspace_id: uuid.UUID, **named: Any) -> None:
        pass

    async def save(self) -> None:
        pass


class _Workspaces:
    def __init__(self, people: list[User]) -> None:
        self._people = {person.id: person for person in people}

    async def people(self, user_ids: Collection[uuid.UUID]) -> list[User]:
        return [self._people[user_id] for user_id in user_ids if user_id in self._people]


class _Catalog:
    async def servers_by_names(self, names: Sequence[str]) -> dict[str, McpCatalogServerPublic]:
        return {}


class _Mail:
    can_send_links = True
    asset_root = None

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def link(self, path: str) -> str:
        return f"https://otari.example{path}"

    async def send(self, *, to: str, message: MailMessage) -> None:
        self.sent.append((to, message.subject))


def _person(name: str, *, email: str | None = "", is_active: bool = True) -> User:
    address = f"{name.lower()}@example.com" if email == "" else email
    return User(id=uuid.uuid4(), full_name=name, email=address, is_active=is_active)


def _request(requester: User, workspace_id: uuid.UUID, url: str = "https://mcp.sentry.dev/mcp") -> McpServerRequest:
    return McpServerRequest(
        id=uuid.uuid4(),
        workspace_id=workspace_id,
        catalog_name="io.sentry/mcp",
        title="Sentry",
        url=url,
        requester_id=requester.id,
        status=McpServerRequestStatus.PENDING,
        created_at=datetime.now(UTC),
    )


@pytest.mark.asyncio
async def test_adding_a_server_approves_its_requests_and_mails_only_those_who_can_be_told() -> None:
    workspace_id = uuid.uuid4()
    ada, gone, silent, admin = (
        _person("Ada"),
        _person("Gone", is_active=False),
        _person("Silent", email=None),
        _person("Admin"),
    )
    rows = [_request(person, workspace_id) for person in (ada, gone, silent, admin)]
    other = _request(ada, workspace_id, url="https://mcp.linear.app/mcp")
    requests = _Requests([*rows, other])
    mail = _Mail()
    deferred: list[Callable[[], Awaitable[None]]] = []
    session = _Session()
    service = McpServerRequestService(
        UnitOfWork(cast(Any, session)),
        cast(McpServerRequestRepository, requests),
        cast(WorkspaceMcpServerRepository, None),
        cast(Any, _Workspaces([ada, gone, silent, admin])),
        _Catalog(),
        cast(Mailer, mail),
        send_later=deferred.append,
    )
    server_id = uuid.uuid4()

    approved = await service.server_added(
        workspace_id=workspace_id,
        workspace_name="Research",
        server_id=server_id,
        url="https://mcp.sentry.dev/mcp/",
        added_by=admin.id,
    )

    assert approved == 4
    assert requests.asked[0]["urls"] == ("https://mcp.sentry.dev/mcp", "https://mcp.sentry.dev/mcp/")
    assert {(row.status, row.server_id, row.decided_by) for row in rows} == {
        (McpServerRequestStatus.APPROVED, server_id, admin.id)
    }
    assert other.status == McpServerRequestStatus.PENDING
    # The approval, then the read of whom to tell in a block of its own.
    assert session.commits == 2
    # Nothing goes out until the deferred work runs, and then only to an active account with an
    # address that is not the admin who added the server.
    assert mail.sent == []
    for work in deferred:
        await work()
    assert mail.sent == [("ada@example.com", "Sentry was added to Research")]


@pytest.mark.asyncio
async def test_a_failed_read_of_whom_to_tell_does_not_fail_the_approval_it_follows() -> None:
    """The approval has committed by then, so a retry would find nothing to approve."""

    class _Unreadable:
        async def people(self, user_ids: Collection[uuid.UUID]) -> list[User]:
            raise OperationalError("SELECT", {}, Exception("connection lost"))

    workspace_id = uuid.uuid4()
    ada, admin = _person("Ada"), _person("Admin")
    rows = [_request(ada, workspace_id)]
    mail = _Mail()
    deferred: list[Callable[[], Awaitable[None]]] = []
    session = _Session()
    service = McpServerRequestService(
        UnitOfWork(cast(Any, session)),
        cast(McpServerRequestRepository, _Requests(rows)),
        cast(WorkspaceMcpServerRepository, None),
        cast(Any, _Unreadable()),
        _Catalog(),
        cast(Mailer, mail),
        send_later=deferred.append,
    )

    approved = await service.server_added(
        workspace_id=workspace_id,
        workspace_name="Research",
        server_id=uuid.uuid4(),
        url="https://mcp.sentry.dev/mcp",
        added_by=admin.id,
    )

    assert approved == 1
    assert rows[0].status == McpServerRequestStatus.APPROVED
    assert session.commits == 1
    assert deferred == []


@pytest.mark.asyncio
async def test_adding_a_server_nobody_asked_for_sends_nothing() -> None:
    mail = _Mail()
    deferred: list[Callable[[], Awaitable[None]]] = []
    service = McpServerRequestService(
        UnitOfWork(cast(Any, _Session())),
        cast(McpServerRequestRepository, _Requests([])),
        cast(WorkspaceMcpServerRepository, None),
        cast(Any, _Workspaces([])),
        _Catalog(),
        cast(Mailer, mail),
        send_later=deferred.append,
    )

    approved = await service.server_added(
        workspace_id=uuid.uuid4(),
        workspace_name="Research",
        server_id=uuid.uuid4(),
        url="https://mcp.sentry.dev/mcp",
        added_by=uuid.uuid4(),
    )

    assert approved == 0
    assert deferred == []
