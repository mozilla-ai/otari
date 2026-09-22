"""Leasing one code-execution sandbox to a caller across requests.

A container id (``otari_cntr_…``) names a sandbox this deployment leased on a
caller's behalf. Within one request the tool loop already shares it across
every code call; this module is what lets the *next* request pick it up, so
the files a run wrote, the packages it installed and the variables it set are
still there. It is the gateway's counterpart to Anthropic's ``container`` and
OpenAI's ``code_interpreter`` container: a client reads the id off one
response and sends it with the next.

What the registry decides:

* **Ownership.** A lease is bound to the user and workspace that took it, and
  a resume by anyone else is answered exactly as an unknown id, so an id never
  reveals that another tenant's sandbox exists.
* **Lifetime.** Two clocks. The idle clock restarts on every use and is what
  the provider is told to hold the sandbox for; the hard clock starts at the
  first lease and never restarts, so a client resuming forever still gives the
  sandbox back. Past either, the id is gone.
* **Provider.** A lease belongs to the provider that made it. A deployment
  that switched providers cannot resume the other's sandbox, and is told so
  the same way.
* **Exclusivity.** One request at a time holds a sandbox. Two sharing one
  workspace would interleave their code and each collect the other's files, so
  a resume while another request is running is refused with
  :class:`ContainerBusyError` rather than admitted.

Standalone only: it keeps its rows in the local database. Reclaiming the
sandbox itself is the provider's job, told the idle timeout at release; the
sweep (``container_sweeper``) only drops rows whose clocks ran out.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from gateway.core.unit_of_work import UnitOfWork
from gateway.repositories.code_execution import (
    SandboxContainerRow,
    claim_container_row,
    delete_container_rows,
    get_container_row,
    release_container_claim,
    upsert_container_row,
)

# The gateway's own container ids. OpenAI issues ``cntr_``-prefixed ids and
# Anthropic ``container_``-prefixed ones, so a reserved prefix is what lets an
# id the gateway minted be told from one that names a provider's container.
CONTAINER_ID_PREFIX = "otari_cntr_"
# How long a request's claim on a container lasts before another request may
# take it. The ceiling on a sandbox session's lease
# (``sandbox_backend._MAX_SESSION_TTL_S``), because a request cannot outlive the
# sandbox it is running code in, so nothing legitimate is displaced and a claim
# a crashed gateway never released expires with the lease it was protecting.
CONTAINER_CLAIM_TTL_S = 3600.0


def new_container_id() -> str:
    """A fresh container id, minted once per lease and kept across resumes."""
    return f"{CONTAINER_ID_PREFIX}{uuid.uuid4().hex}"


class ContainerNotFoundError(LookupError):
    """The container is unknown, expired, another tenant's, or another provider's.

    One error for all four on purpose: telling them apart would tell a caller
    which ids exist.
    """


class ContainerBusyError(RuntimeError):
    """The container is this caller's, and another of their requests is using it.

    Distinct from unknown because the answer is different: the id is good and
    retrying works, where an unknown id has to be dropped. Only reachable for a
    container the caller owns, so it reveals nothing about anyone else's.
    """


@dataclass(frozen=True)
class ContainerLease:
    """One held sandbox: the id a client resumes it by and what it maps to.

    ``provider_session_id`` is the provider's own handle, the one the port
    resumes. ``expires_at`` is the idle clock and ``hard_expires_at`` the hard
    one, both as reported to the client; the sandbox may in fact live a little
    longer, never shorter.
    """

    container_id: str
    provider: str
    provider_session_id: str
    expires_at: datetime
    hard_expires_at: datetime


class SandboxContainers(Protocol):
    """What the sandbox backend needs to hold a session past its request.

    Implemented by :class:`SandboxContainerRegistry`; a Protocol so the backend
    does not depend on the database-backed one and a test can hand it a stub.
    """

    def keep_alive_s(self, resumed: ContainerLease | None) -> float:
        """How long the provider should hold the session after this request."""
        ...

    def lease(self, container_id: str, provider_session_id: str, *, resumed: ContainerLease | None) -> ContainerLease:
        """The lease this request holds, with both clocks set from now."""
        ...

    async def record(self, lease: ContainerLease) -> None:
        """Persist the lease so the next request can resume it."""
        ...

    async def release(self, container_id: str) -> None:
        """Give back a claim taken at admission that no lease will be recorded against."""
        ...


class SandboxContainerRegistry:
    """The database-backed :class:`SandboxContainers`, scoped to one caller.

    Built per request by the route once the billed user and workspace are
    known. Writes go through the request's Unit of Work, one block each, for
    the reason the file bridge's do: the request session is released before
    the provider is dispatched, so nothing here holds a transaction open while
    code runs.
    """

    def __init__(
        self,
        *,
        uow: UnitOfWork,
        user_id: str,
        workspace_id: uuid.UUID,
        provider: str,
        idle_ttl_s: int,
        max_lifetime_s: int,
    ) -> None:
        self._uow = uow
        self._user_id = user_id
        self._workspace_id = workspace_id
        self._provider = provider
        self._idle_ttl_s = idle_ttl_s
        self._max_lifetime_s = max_lifetime_s

    @property
    def idle_ttl_s(self) -> int:
        return self._idle_ttl_s

    def keep_alive_s(self, resumed: ContainerLease | None) -> float:
        # Never past the hard clock, and never so short the provider reclaims
        # the sandbox before the row saying so is written.
        if resumed is None:
            return float(self._idle_ttl_s)
        remaining = (resumed.hard_expires_at - datetime.now(UTC)).total_seconds()
        return max(1.0, min(float(self._idle_ttl_s), remaining))

    def lease(self, container_id: str, provider_session_id: str, *, resumed: ContainerLease | None) -> ContainerLease:
        now = datetime.now(UTC)
        hard = resumed.hard_expires_at if resumed is not None else now + timedelta(seconds=self._max_lifetime_s)
        return ContainerLease(
            container_id=container_id,
            provider=self._provider,
            provider_session_id=provider_session_id,
            expires_at=min(now + timedelta(seconds=self._idle_ttl_s), hard),
            hard_expires_at=hard,
        )

    async def resolve(self, container_id: str) -> ContainerLease:
        """The lease behind ``container_id`` for this caller, claimed for this request.

        Raises :class:`ContainerNotFoundError` when the id is unknown, expired,
        another tenant's or another provider's, and :class:`ContainerBusyError`
        when it is this caller's own and another of their requests holds it.

        A row whose clocks ran out is dropped on the way, so the next resume
        does not read it again. Every write lands before either error is raised,
        which is why the block closes first.
        """
        if not container_id.startswith(CONTAINER_ID_PREFIX):
            raise ContainerNotFoundError(container_id)
        now = datetime.now(UTC)
        async with self._uow:
            row = await get_container_row(self._uow, container_id)
            expired = row is not None and (row.expires_at <= now or row.hard_expires_at <= now)
            if expired:
                await delete_container_rows(self._uow, [container_id])
            resumable = (
                row is not None
                and not expired
                and row.user_id == self._user_id
                and row.workspace_id == self._workspace_id
                and row.provider == self._provider
            )
            claimed = resumable and await claim_container_row(
                self._uow,
                container_id,
                now=now,
                until=now + timedelta(seconds=CONTAINER_CLAIM_TTL_S),
            )
        if row is None or not resumable:
            raise ContainerNotFoundError(container_id)
        if not claimed:
            raise ContainerBusyError(container_id)
        return _lease_from_row(row)

    async def record(self, lease: ContainerLease) -> None:
        now = datetime.now(UTC)
        async with self._uow:
            await upsert_container_row(
                self._uow,
                SandboxContainerRow(
                    id=lease.container_id,
                    user_id=self._user_id,
                    workspace_id=self._workspace_id,
                    provider=lease.provider,
                    provider_session_id=lease.provider_session_id,
                    created_at=now,
                    last_used_at=now,
                    expires_at=lease.expires_at,
                    hard_expires_at=lease.hard_expires_at,
                ),
            )

    async def forget(self, container_id: str) -> None:
        async with self._uow:
            await delete_container_rows(self._uow, [container_id])

    async def release(self, container_id: str) -> None:
        """Give back a claim this request took and will not record a lease against."""
        async with self._uow:
            await release_container_claim(self._uow, container_id)


def _lease_from_row(row: SandboxContainerRow) -> ContainerLease:
    return ContainerLease(
        container_id=row.id,
        provider=row.provider,
        provider_session_id=row.provider_session_id,
        expires_at=row.expires_at,
        hard_expires_at=row.hard_expires_at,
    )
