"""A stored MCP server's health: what the gateway last learned by talking to it.

Two things write it. A manager's check, which lists the server's tools on request, and
``McpServerContacts``, which records what a real request found when it listed a server's tools or
called one, so a server in use needs no check by hand.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from time import monotonic
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, field_validator

from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork, create_unit_of_work
from gateway.exceptions.tools_exceptions import McpConnectionUnavailableError, McpSignInRequiredError
from gateway.log_config import logger
from gateway.models.tools import McpCredentialMode
from gateway.repositories.tools import WorkspaceMcpServerRepository
from gateway.services.mcp_stateless import (
    CheckFailure,
    McpExecutionError,
    ServerCheck,
    classify_check_failure,
    reads_as_pattern,
)

# Bounds on the tool allow-list, which is arbitrary JSON stored on the row and
# read again on the completion path. The names a check keeps follow the same two.
MAX_ALLOWED_TOOLS = 200
MAX_TOOL_NAME_LENGTH = 256

CONTACT_WRITE_INTERVAL_S = 15 * 60
"""How often one server's health is rewritten from request traffic while its status stays the same."""

# Past this many servers the throttle forgets what it saw, which costs one extra write each.
_MAX_REMEMBERED_SERVERS = 10_000
# Writes in flight at once. A contact past this bound is not recorded; a later contact is.
_MAX_PENDING_WRITES = 64


class WorkspaceMcpServerHealth(BaseModel):
    """The last time the gateway talked to the server: whether it answered, and what it found.

    Written by a manager's check and by real requests that reach the server. Describes the server
    as the gateway saw it at ``checked_at``, not now: a server that answered then can be down
    since, and the request path checks nothing in advance on the strength of this.
    """

    MAX_ALLOWED_TOOLS: ClassVar[int] = MAX_ALLOWED_TOOLS
    MAX_TOOL_NAME_LENGTH: ClassVar[int] = MAX_TOOL_NAME_LENGTH

    status: Literal["healthy", "failing"]
    checked_at: str
    error: CheckFailure | None = Field(default=None, description="Why the check failed; null when healthy")
    http_status: int | None = Field(default=None, description="The HTTP status the server refused with, if any")
    tool_count: int | None = Field(
        default=None,
        description="Live tools the allow-list admits; null when the check failed, or when the server "
        "answered a request that did not list its tools and no listing is on record",
    )
    missing_allowed_tools: list[str] = Field(
        default_factory=list, description="Allow-listed names the server did not offer"
    )
    tools: list[str] = Field(
        default_factory=list,
        description=f"Every tool name the server offers, allowed or not, up to {MAX_ALLOWED_TOOLS}; empty on a failure",
    )
    read_only_hinted_tools: list[str] = Field(
        default_factory=list,
        description="Those of tools the server annotates readOnlyHint: untrusted, and what read_tools "
        "follows until a manager edits the labels",
    )
    tool_descriptions: dict[str, str] = Field(
        default_factory=dict,
        description="The first line of each offered tool's own description, where it has one: untrusted text "
        "the server wrote, for a person to read",
    )

    @field_validator("tool_descriptions")
    @classmethod
    def _described_literal_names(cls, descriptions: dict[str, str]) -> dict[str, str]:
        """Keep descriptions of literal names only, as ``tools`` does."""
        return {name: text for name, text in descriptions.items() if not reads_as_pattern(name)}

    @field_validator("tools", "read_only_hinted_tools")
    @classmethod
    def _literal_names(cls, names: list[str]) -> list[str]:
        """Drop a name a rule would read as a pattern, from a fresh check and from one stored before this screen."""
        return [name for name in names if not reads_as_pattern(name)]

    @staticmethod
    def from_check(allowed_tools: list[str] | None, check: ServerCheck) -> WorkspaceMcpServerHealth:
        """The health one check of the server found (``server_health``)."""
        return server_health(allowed_tools, check)


def server_health(allowed_tools: list[str] | None, check: ServerCheck) -> WorkspaceMcpServerHealth:
    """Read a check against an allow-list: ``None`` admits every tool and ``[]`` none, as the tool loop reads it."""
    offered = list(dict.fromkeys(name for name in check.tool_names if len(name) <= MAX_TOOL_NAME_LENGTH))
    failed = check.failure is not None
    allowed = None if allowed_tools is None else list(dict.fromkeys(allowed_tools))
    return WorkspaceMcpServerHealth(
        status="failing" if failed else "healthy",
        checked_at=datetime.now(UTC).isoformat(),
        error=check.failure,
        http_status=check.http_status,
        tool_count=None if failed else len(offered) if allowed is None else len(set(allowed) & set(offered)),
        missing_allowed_tools=[] if failed or allowed is None else [name for name in allowed if name not in offered],
        tools=offered[:MAX_ALLOWED_TOOLS],
        read_only_hinted_tools=[name for name in dict.fromkeys(check.read_only_hinted) if name in offered][
            :MAX_ALLOWED_TOOLS
        ],
        tool_descriptions={
            name: check.descriptions[name] for name in offered[:MAX_ALLOWED_TOOLS] if name in check.descriptions
        },
    )


@dataclass(frozen=True, slots=True)
class McpServerContact:
    """What one request learned about a stored server by talking to it. Carries nothing about who asked.

    ``listing`` is the server's whole tool catalog when the request read all of it, as a check
    does. ``failure`` is why the server could not be reached.
    """

    server_id: uuid.UUID
    url: str
    listing: ServerCheck | None = None
    failure: CheckFailure | None = None
    http_status: int | None = None


# One user's rejected token says nothing about the server: everyone else's may still work.
_ONE_USERS_REFUSAL = frozenset({CheckFailure.UNAUTHORIZED, CheckFailure.FORBIDDEN})


# Found without talking to the server, so they say nothing a request learned from it.
_DECIDED_BEFORE_CONNECTING = frozenset({CheckFailure.UNSAFE_URL, CheckFailure.CREDENTIALS_UNAVAILABLE})


def health_after_contact(
    stored: dict[str, Any] | None,
    *,
    allowed_tools: list[str] | None,
    per_user: bool,
    contact: McpServerContact,
    interval_s: float = CONTACT_WRITE_INTERVAL_S,
) -> dict[str, Any] | None:
    """The health to store after a request's contact with a server, or ``None`` to leave it as it is.

    A contact that says what the stored health already says, within ``interval_s`` of it, is not
    written again. A successful one that listed no tools keeps the tools on record.
    """
    previous = WorkspaceMcpServerHealth.model_validate(stored) if stored is not None else None
    recent = previous is not None and _is_recent(previous.checked_at, interval_s)
    if contact.failure is not None:
        if per_user and contact.failure in _ONE_USERS_REFUSAL:
            return None
        if recent and previous is not None and previous.error == contact.failure:
            return None
        failed = ServerCheck(tool_names=[], failure=contact.failure, http_status=contact.http_status)
        health = server_health(allowed_tools, failed)
        if previous is not None:
            # What '@read' stands for until a manager labels tools: one failed contact must not empty it.
            health = health.model_copy(update={"read_only_hinted_tools": previous.read_only_hinted_tools})
        return health.model_dump(mode="json")
    was_healthy = previous is not None and previous.status == "healthy"
    if contact.listing is not None:
        listed = server_health(allowed_tools, contact.listing)
        if (
            recent
            and previous is not None
            and was_healthy
            and previous.tools == listed.tools
            and previous.read_only_hinted_tools == listed.read_only_hinted_tools
        ):
            return None
        return listed.model_dump(mode="json")
    if previous is not None and was_healthy:
        if recent:
            return None
        return previous.model_copy(update={"checked_at": datetime.now(UTC).isoformat()}).model_dump(mode="json")
    return WorkspaceMcpServerHealth(
        status="healthy",
        checked_at=datetime.now(UTC).isoformat(),
        read_only_hinted_tools=previous.read_only_hinted_tools if previous is not None else [],
    ).model_dump(mode="json")


def _is_recent(checked_at: str, interval_s: float) -> bool:
    try:
        at = datetime.fromisoformat(checked_at)
    except ValueError:
        return False
    if at.tzinfo is None:
        at = at.replace(tzinfo=UTC)
    return datetime.now(UTC) - at < timedelta(seconds=interval_s)


class McpServerContacts:
    """Records on a stored server what real requests find when they talk to it.

    Neither method waits or raises: the write runs in a task of its own, on a Unit of Work of its
    own, and a failure there is logged. One server is written at most once per
    ``interval_s`` per process unless its status changes, and :func:`health_after_contact` holds
    the stored row to the same interval across processes.
    """

    def __init__(
        self,
        *,
        unit_of_work: Callable[[], AbstractAsyncContextManager[UnitOfWork]] = create_unit_of_work,
        interval_s: float = CONTACT_WRITE_INTERVAL_S,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._unit_of_work = unit_of_work
        self._interval_s = interval_s
        self._clock = clock
        # When each server was last handed to a write, with which status, and whether it carried a listing.
        self._seen: dict[uuid.UUID, tuple[float, bool, bool]] = {}
        # Per-user servers a write found, and when: one user's 401 or 403 there is never stored.
        self._per_user: dict[uuid.UUID, float] = {}
        # Held so a running write is not collected before it finishes.
        self._tasks: set[asyncio.Task[None]] = set()

    def reached(self, server_id: uuid.UUID | str | None, url: str, listing: ServerCheck | None = None) -> None:
        """A request got an answer from the server; ``listing`` is its whole tool catalog, when it read one."""
        parsed = _server_id(server_id)
        if parsed is not None:
            self._record(McpServerContact(server_id=parsed, url=url, listing=listing))

    def failed(self, server_id: uuid.UUID | str | None, url: str, failure: BaseException) -> None:
        """A request could not open a session on the server.

        A failure that is about the caller and not the server is ignored: a sign-in that is
        missing or could not be renewed, and a refusal the gateway itself decided.
        """
        parsed = _server_id(server_id)
        if parsed is None or isinstance(
            failure, McpSignInRequiredError | McpConnectionUnavailableError | McpExecutionError
        ):
            return
        cause, http_status = classify_check_failure(failure)
        self._record(McpServerContact(server_id=parsed, url=url, failure=cause, http_status=http_status))

    def observed(self, server_id: uuid.UUID, url: str, check: ServerCheck) -> None:
        """A request listed the server's whole catalog the way a check does, and this is what it found."""
        if check.failure is None:
            self.reached(server_id, url, check)
        elif check.failure not in _DECIDED_BEFORE_CONNECTING:
            self._record(
                McpServerContact(server_id=server_id, url=url, failure=check.failure, http_status=check.http_status)
            )

    def _record(self, contact: McpServerContact) -> None:
        ok = contact.failure is None
        listed = contact.listing is not None
        now = self._clock()
        if contact.failure in _ONE_USERS_REFUSAL:
            learned = self._per_user.get(contact.server_id)
            if learned is not None and now - learned < self._interval_s:
                return
        last = self._seen.get(contact.server_id)
        # A listing after a contact that listed nothing is let through once, so the catalog is stored.
        if last is not None and last[1] == ok and (last[2] or not listed) and now - last[0] < self._interval_s:
            return
        if len(self._tasks) >= _MAX_PENDING_WRITES:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if len(self._seen) >= _MAX_REMEMBERED_SERVERS:
            self._seen.clear()
            self._per_user.clear()
        handed = (now, ok, listed)
        self._seen[contact.server_id] = handed
        task = loop.create_task(self._write(contact, handed, last), name="mcp-server-contact")
        self._tasks.add(task)
        task.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (failure := task.exception()) is not None:
            logger.warning("MCP server contact not recorded error_class=%s", type(failure).__name__)

    async def _write(
        self,
        contact: McpServerContact,
        handed: tuple[float, bool, bool] | None = None,
        before: tuple[float, bool, bool] | None = None,
    ) -> None:
        try:
            async with self._unit_of_work() as uow, uow:
                server = await WorkspaceMcpServerRepository(uow).find(contact.server_id, lock=True)
                # A request that resolved the server before its URL was edited describes the old one.
                if server is None or server.url != contact.url:
                    return
                per_user = server.credential_mode == McpCredentialMode.PER_USER
                if per_user and contact.failure in _ONE_USERS_REFUSAL:
                    self._discarded(contact.server_id, handed, before)
                    return
                health = health_after_contact(
                    server.health,
                    allowed_tools=server.allowed_tools,
                    per_user=per_user,
                    contact=contact,
                    interval_s=self._interval_s,
                )
                if health is None:
                    return
                server.health = health
        except DATABASE_ERRORS as exc:
            self._seen.pop(contact.server_id, None)
            logger.warning(
                "MCP server contact not recorded server_id=%s %s", contact.server_id, describe_database_error(exc)
            )
            return
        logger.info("MCP server contact recorded server_id=%s outcome=%s", contact.server_id, contact.failure or "ok")

    def _discarded(
        self,
        server_id: uuid.UUID,
        handed: tuple[float, bool, bool] | None,
        before: tuple[float, bool, bool] | None,
    ) -> None:
        """Take back the throttle entry of a contact that was not stored, so a real failure after it is."""
        self._per_user[server_id] = self._clock()
        if handed is None or self._seen.get(server_id) != handed:
            return
        if before is None:
            del self._seen[server_id]
        else:
            self._seen[server_id] = before

    async def drain(self) -> None:
        """Wait for the writes in flight, for a shutdown or a test that reads what they stored."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
            # A finished task leaves the set in its done callback, which runs on the loop's next turn.
            await asyncio.sleep(0)


def _server_id(server_id: uuid.UUID | str | None) -> uuid.UUID | None:
    if server_id is None or isinstance(server_id, uuid.UUID):
        return server_id
    try:
        return uuid.UUID(server_id)
    except ValueError:
        return None


mcp_server_contacts = McpServerContacts()
"""The process's recorder, which the request paths that talk to a stored server report to."""
