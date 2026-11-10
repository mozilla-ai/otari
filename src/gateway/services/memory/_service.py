"""The memory domain's one service: what agents search, propose, confirm, flag and forget, and the consolidator.

No model is called inside a Unit of Work block, so a slow provider never holds
a transaction open: each use case reads in one block, calls its model, and
writes in another.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol

from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.memory_exceptions import (
    MemoryEntryNotFoundError,
    MemoryModelUnavailableError,
    MemoryOffError,
    MemoryReadOnlyError,
    MemoryRefusedError,
    MemoryTooLongError,
    NotYoursToForgetError,
    WorkspaceScopeOffError,
)
from gateway.log_config import logger
from gateway.models.memory import MemoryKind, MemoryScope, MemoryStatus, MemoryTrust, MemoryWritePolicy
from gateway.models.tenancy import User
from gateway.repositories.memory import MemoryRepositories
from gateway.services.memory._keys import WorkspaceKeys
from gateway.services.memory._models import Decision, MemoryModelError, MemoryModels, ModelUse
from gateway.services.memory._scoring import INITIAL_CONFIDENCE, QUARANTINE_FLAGS, confidence, cosine, relevance
from gateway.services.memory._screening import screen
from gateway.services.memory._store import (
    EntryDraft,
    EntryStore,
    OpenedEntry,
    OpenedProposal,
    ProposalDraft,
    Provenance,
)

SEARCH_LIMIT_MAX = 10
HISTORY_LIMIT = 20
# Entries a proposal is compared with, and how alike one must be to count.
NEIGHBOURS = 5
COMPARE_FLOOR = 0.75
# Without a consolidation model, a proposal this alike to an entry adds nothing.
SAME_FLOOR = 0.95
# Passes a proposal may fail on a model before it waits for an admin instead.
MAX_ATTEMPTS = 5

_TRUST_RANK: dict[MemoryTrust, int] = {"external": 0, "agent": 1, "reviewed": 2}

FlagReason = Literal["wrong", "stale", "duplicate", "unsafe"]


class WorkspaceManagers(Protocol):
    """Who may change memory's switches: an organization's owners and admins, and a workspace's own.

    The builder in ``api/deps`` implements it with the tenancy checks every workspace policy uses.
    """

    async def require_manager(self, user: User, workspace_id: uuid.UUID) -> uuid.UUID:
        """The workspace's organization, unless ``user`` does not manage it; one they cannot see reads as missing."""
        ...

    async def require_organization_manager(self, user: User) -> uuid.UUID:
        """The caller's active organization, unless they are not one of its owners or admins."""
        ...


class MemoryMeter(Protocol):
    """Where memory's model calls are recorded as its workspace's usage. The builder in ``api/deps`` implements it."""

    async def record(self, workspace_id: uuid.UUID, api_key_id: str | None, use: ModelUse) -> None:
        """Stage one call in the open block."""
        ...


@dataclass(frozen=True)
class MemoryCaller:
    """Who is asking, as their credential names them: never as a tool argument says."""

    workspace_id: uuid.UUID
    member_id: uuid.UUID
    is_manager: bool
    can_write: bool = True
    api_key_id: str | None = None
    session_id: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class Recalled:
    """One search result and what it was ranked by."""

    entry: OpenedEntry
    relevance: float


@dataclass(frozen=True)
class EntryWithHistory:
    entry: OpenedEntry
    versions: list[OpenedEntry]


@dataclass(frozen=True)
class SignalReceipt:
    entry_id: uuid.UUID
    confidence: float
    status: MemoryStatus


@dataclass(frozen=True)
class Swept:
    """What one retention pass deleted from a workspace."""

    entries: int
    proposals: int
    reads: int
    keys: int


@dataclass(frozen=True)
class OrganizationMemory:
    organization_id: uuid.UUID
    enabled: bool


@dataclass(frozen=True)
class WorkspaceMemory:
    """A workspace's memory switches, its organization's, and how much it holds."""

    workspace_id: uuid.UUID
    organization_enabled: bool
    enabled: bool
    workspace_scope_enabled: bool
    write_policy: MemoryWritePolicy
    hide_external: bool
    retention_days: int | None
    entries: dict[str, int]
    proposals_waiting: int


@dataclass(frozen=True)
class SwitchChange:
    enabled: bool
    workspace_scope_enabled: bool
    write_policy: MemoryWritePolicy
    hide_external: bool
    retention_days: int | None


@dataclass(frozen=True)
class _Switches:
    """A workspace's memory switches, read once and held past the block that read them."""

    workspace_scope: bool
    hide_external: bool
    write_policy: MemoryWritePolicy


class MemoryService:
    """Store what agents propose and what the consolidator keeps, sealed, and open what a member may recall."""

    def __init__(
        self,
        uow: UnitOfWork,
        repos: MemoryRepositories,
        keys: WorkspaceKeys,
        models: MemoryModels,
        meter: MemoryMeter,
        config: GatewayConfig,
        managers: WorkspaceManagers | None,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._uow = uow
        self._repos = repos
        self._store = EntryStore(repos, keys)
        self._models = models
        self._meter = meter
        self._config = config
        self._managers = managers
        self._keys = keys
        self._now = now

    # -- What agents call ---------------------------------------------------

    async def search(self, caller: MemoryCaller, query: str, *, limit: int) -> list[Recalled]:
        """The entries the caller may recall that best match ``query``, best first.

        Raises:
            MemoryOffError: the workspace has not turned memory on.
            MemoryModelUnavailableError: the query could not be embedded.
        """
        async with self._uow:
            switches = await self._switches(caller.workspace_id)
        [vector] = await self._embed([query], caller.workspace_id, caller.api_key_id)
        model = self._models.embedding_model
        async with self._uow:
            entries = await self._store.recallable(
                caller.workspace_id,
                member_id=caller.member_id,
                include_workspace=switches.workspace_scope,
                include_external=not switches.hide_external,
            )
            ranked = sorted(
                (
                    Recalled(entry, relevance(cosine(vector, entry.embedding), entry.confidence))
                    for entry in entries
                    if entry.embedding is not None and entry.embedding_model == model
                ),
                key=lambda recalled: recalled.relevance,
                reverse=True,
            )[: max(1, min(limit, SEARCH_LIMIT_MAX))]
            ids = [recalled.entry.id for recalled in ranked]
            await self._record_read(caller, "memory_search", ids)
        return ranked

    async def get(self, caller: MemoryCaller, entry_id: uuid.UUID) -> EntryWithHistory:
        """One entry the caller may recall, with the versions it replaced.

        Raises:
            MemoryOffError: the workspace has not turned memory on.
            MemoryEntryNotFoundError: the caller may recall no entry by that id.
        """
        async with self._uow:
            switches = await self._switches(caller.workspace_id)
            entry = await self._visible_entry(caller, switches, entry_id)
            versions = await self._store.history(caller.workspace_id, entry, limit=HISTORY_LIMIT)
            await self._record_read(caller, "memory_get", [entry.id, *(version.id for version in versions)])
        return EntryWithHistory(entry, versions)

    async def propose(
        self, caller: MemoryCaller, *, text: str, kind: MemoryKind, scope: MemoryScope, external: bool
    ) -> uuid.UUID:
        """Store one proposal for the consolidator; its id. Nothing becomes memory until it is decided.

        Raises:
            MemoryReadOnlyError: the caller is a workspace viewer.
            MemoryTooLongError: the text is longer than memory keeps.
            MemoryRefusedError: the text carries a credential or personal data.
            MemoryOffError: the workspace has not turned memory on.
            WorkspaceScopeOffError: the proposal is for the shared memory, which the workspace has off.
            MemorySealingUnavailableError: no data key could be had, so nothing was stored.
        """
        _require_write(caller)
        text = text.strip()
        if len(text) > self._config.memory_entry_max_length:
            raise MemoryTooLongError(self._config.memory_entry_max_length)
        finding = screen(text)
        if finding is not None:
            raise MemoryRefusedError(finding)
        async with self._uow:
            switches = await self._switches(caller.workspace_id)
            if scope == "workspace" and not switches.workspace_scope:
                raise WorkspaceScopeOffError
            return await self._store.write_proposal(
                ProposalDraft(
                    workspace_id=caller.workspace_id,
                    scope=scope,
                    owner_user_id=caller.member_id if scope == "member" else None,
                    kind=kind,
                    trust="external" if external else "agent",
                    text=text,
                    provenance=Provenance(
                        author_user_id=caller.member_id,
                        api_key_id=caller.api_key_id,
                        session_id=caller.session_id,
                        model=caller.model,
                    ),
                )
            )

    async def confirm(self, caller: MemoryCaller, entry_id: uuid.UUID) -> SignalReceipt:
        """Say an entry held up in use. A member counts once per entry, whatever they said before.

        Raises:
            MemoryOffError: the workspace has not turned memory on.
            MemoryEntryNotFoundError: the caller may recall no entry by that id.
        """
        return await self._signal(caller, entry_id, "confirm", None)

    async def flag(self, caller: MemoryCaller, entry_id: uuid.UUID, reason: FlagReason) -> SignalReceipt:
        """Say an entry is wrong or stale. Flagged by enough members, it leaves recall for an admin to look at.

        Raises:
            MemoryOffError: the workspace has not turned memory on.
            MemoryEntryNotFoundError: the caller may recall no entry by that id.
        """
        return await self._signal(caller, entry_id, "flag", reason)

    async def forget(self, caller: MemoryCaller, entry_id: uuid.UUID) -> int:
        """Delete an entry and every version it replaced; how many versions went.

        Its author, its owner, or a workspace manager for a workspace entry, may.

        Raises:
            MemoryOffError: the workspace has not turned memory on.
            MemoryEntryNotFoundError: the caller may recall no entry by that id.
            NotYoursToForgetError: the caller may recall it but not remove it.
        """
        _require_write(caller)
        async with self._uow:
            switches = await self._switches(caller.workspace_id)
            entry = await self._visible_entry(caller, switches, entry_id)
            allowed = caller.member_id in (entry.author_user_id, entry.owner_user_id) or (
                caller.is_manager and entry.scope == "workspace"
            )
            if not allowed:
                raise NotYoursToForgetError
            versions = await self._store.history(caller.workspace_id, entry, limit=10_000)
            return await self._repos.entries.delete_versions(
                caller.workspace_id, [entry.id, *(version.id for version in versions)]
            )

    # -- What workspace managers call ----------------------------------------

    async def organization_memory(self, user: User) -> OrganizationMemory:
        """The caller's organization's switch. Its owners and admins only."""
        organization_id = await self._managers_or_fail().require_organization_manager(user)
        async with self._uow:
            return OrganizationMemory(organization_id, await self._repos.organizations.enabled(organization_id))

    async def change_organization_switch(self, user: User, *, enabled: bool) -> OrganizationMemory:
        """Turn memory on or off for every workspace of the caller's organization. Off deletes nothing.

        On lets each workspace's managers turn it on for their workspace; off stops recall in all of them at once.
        """
        organization_id = await self._managers_or_fail().require_organization_manager(user)
        async with self._uow:
            await self._repos.organizations.put(organization_id, enabled=enabled)
        logger.info("Memory was turned %s for organization %s", "on" if enabled else "off", organization_id)
        return OrganizationMemory(organization_id, enabled)

    async def workspace_memory(self, user: User, workspace_id: uuid.UUID) -> WorkspaceMemory:
        """The workspace's switches and what it holds. Reading them takes the same role as changing them."""
        organization_id = await self._managers_or_fail().require_manager(user, workspace_id)
        async with self._uow:
            return await self._workspace_memory(workspace_id, organization_id)

    async def change_switches(self, user: User, workspace_id: uuid.UUID, change: SwitchChange) -> WorkspaceMemory:
        """Set the workspace's switches. Turning memory off stops recall at once and deletes nothing.

        A workspace can be switched on while its organization has memory off; it runs once both are on.
        """
        organization_id = await self._managers_or_fail().require_manager(user, workspace_id)
        async with self._uow:
            await self._repos.settings.put(
                workspace_id,
                organization_id=organization_id,
                enabled=change.enabled,
                workspace_scope_enabled=change.workspace_scope_enabled,
                write_policy=change.write_policy,
                hide_external=change.hide_external,
                retention_days=change.retention_days,
            )
            return await self._workspace_memory(workspace_id, organization_id)

    async def erase(self, user: User, workspace_id: uuid.UUID) -> Swept:
        """Delete everything the workspace's memory holds, its data keys last.

        With the keys gone, no backup taken before opens what was deleted. The switches stay as they were.
        """
        await self._managers_or_fail().require_manager(user, workspace_id)
        async with self._uow:
            entries = await self._repos.entries.delete_all(workspace_id)
            proposals = await self._repos.proposals.delete_all(workspace_id)
            reads = await self._repos.reads.delete_all(workspace_id)
            keys = await self._repos.keys.delete_all(workspace_id)
        self._keys.forget(keys)
        logger.info("Memory of workspace %s was erased by a manager", workspace_id)
        return Swept(entries, proposals, reads, len(keys))

    def _managers_or_fail(self) -> WorkspaceManagers:
        if self._managers is None:
            msg = "this memory service was built without workspace management checks"
            raise RuntimeError(msg)
        return self._managers

    async def _workspace_memory(self, workspace_id: uuid.UUID, organization_id: uuid.UUID) -> WorkspaceMemory:
        settings = await self._repos.settings.of_workspace(workspace_id)
        return WorkspaceMemory(
            workspace_id=workspace_id,
            organization_enabled=await self._repos.organizations.enabled(organization_id),
            enabled=settings is not None and settings.enabled,
            workspace_scope_enabled=settings is not None and settings.workspace_scope_enabled,
            write_policy=settings.write_policy if settings is not None else "review",
            hide_external=settings is not None and settings.hide_external,
            retention_days=settings.retention_days if settings is not None else None,
            entries=await self._repos.entries.count_by_status(workspace_id),
            proposals_waiting=await self._repos.proposals.count_pending(workspace_id),
        )

    # -- Retention ----------------------------------------------------------

    async def workspaces_with_settings(self) -> list[uuid.UUID]:
        """Every workspace that has set its memory switches, on or off: what it stored is kept to its retention."""
        async with self._uow:
            return [settings.workspace_id for settings in await self._repos.settings.all()]

    async def sweep(self, workspace_id: uuid.UUID) -> Swept:
        """Delete what the workspace keeps past its retention, and the data keys that seal nothing any more.

        Entries and their versions go once nobody has used them for the workspace's retention days, retired
        entries at once, proposals once older than ``memory_proposal_max_age_hours`` and read records after
        ``memory_read_retention_days``.
        """
        now = self._now()
        async with self._uow:
            settings = await self._repos.settings.of_workspace(workspace_id)
            days = settings.retention_days if settings is not None and settings.retention_days else None
            entries = await self._repos.entries.delete_expired(
                workspace_id, used_before=now - timedelta(days=days or self._config.memory_retention_days)
            )
            proposals = await self._repos.proposals.delete_older(
                workspace_id, created_before=now - timedelta(hours=self._config.memory_proposal_max_age_hours)
            )
            reads = await self._repos.reads.delete_older(
                workspace_id, read_before=now - timedelta(days=self._config.memory_read_retention_days)
            )
            in_use = await self._repos.entries.key_ids_in_use(workspace_id)
            in_use |= await self._repos.proposals.key_ids_in_use(workspace_id)
            keys = await self._repos.keys.delete_unused(
                workspace_id,
                in_use=in_use,
                made_before=now - timedelta(days=self._config.memory_key_rotation_days),
            )
        return Swept(entries, proposals, reads, keys)

    # -- The consolidator ---------------------------------------------------

    async def workspaces_with_memory(self) -> list[uuid.UUID]:
        """Every workspace that has memory on, and whose organization has it on too."""
        async with self._uow:
            workspaces = await self._repos.settings.enabled_workspaces()
            organizations = await self._repos.organizations.enabled_among({org for _, org in workspaces})
        return [workspace_id for workspace_id, organization_id in workspaces if organization_id in organizations]

    async def consolidate(self, workspace_id: uuid.UUID, *, limit: int) -> int:
        """Decide up to ``limit`` of the workspace's pending proposals; how many were decided.

        A proposal whose model call fails is left for a later pass, and after
        ``MAX_ATTEMPTS`` waits for an admin. A proposal another replica decided
        first is skipped.
        """
        async with self._uow:
            switches = await self._switches_if_on(workspace_id)
            if switches is None:
                return 0
            proposals = await self._store.pending_proposals(workspace_id, limit=limit)
        decided = 0
        for proposal in proposals:
            try:
                decided += await self._consolidate_one(proposal, switches)
            except (MemoryModelUnavailableError, MemoryModelError):
                await self._count_failed_pass(proposal)
        return decided

    async def _consolidate_one(self, proposal: OpenedProposal, switches: _Switches) -> int:
        workspace_id = proposal.workspace_id
        if proposal.scope == "workspace" and switches.write_policy == "review" and proposal.trust != "reviewed":
            async with self._uow:
                await self._repos.proposals.hold_for_review(workspace_id, proposal.id, "policy")
            return 1
        [vector] = await self._embed([proposal.text], workspace_id, proposal.provenance.api_key_id)
        model = self._models.embedding_model or ""
        async with self._uow:
            candidates = await self._store.comparable(
                workspace_id, scope=proposal.scope, owner_user_id=proposal.owner_user_id, embedding_model=model
            )
        nearest = sorted(
            (
                (cosine(vector, entry.embedding), entry)
                for entry in candidates
                if entry.embedding is not None and cosine(vector, entry.embedding) >= COMPARE_FLOOR
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )[:NEIGHBOURS]
        decision = await self._decide(proposal, nearest)
        target = nearest[decision.target][1] if decision.target is not None else None
        if decision.action == "conflict" or (
            decision.action == "update"
            and target is not None
            and _TRUST_RANK[target.trust] > _TRUST_RANK[proposal.trust]
        ):
            async with self._uow:
                await self._repos.proposals.hold_for_review(
                    workspace_id, proposal.id, "conflict" if decision.action == "conflict" else "higher_trust"
                )
            return 1
        async with self._uow:
            # Removed first, in the block that applies it: a replica that finds it gone applies nothing.
            if not await self._repos.proposals.remove(workspace_id, proposal.id):
                return 0
            if decision.action == "no_change" and target is not None:
                await self._repos.entries.touch(workspace_id, [target.id], self._now())
                return 1
            if decision.action == "add":
                await self._make_room(workspace_id)
            await self._store.write_entry(
                EntryDraft(
                    workspace_id=workspace_id,
                    scope=proposal.scope,
                    owner_user_id=proposal.owner_user_id,
                    kind=proposal.kind,
                    trust=proposal.trust,
                    text=proposal.text,
                    embedding=vector,
                    embedding_model=model,
                    confidence=INITIAL_CONFIDENCE,
                    provenance=proposal.provenance,
                    version=target.version + 1 if target is not None else 1,
                    supersedes_id=target.id if target is not None else None,
                )
            )
            if target is not None:
                await self._repos.entries.set_status(workspace_id, target.id, "superseded")
        return 1

    async def _decide(self, proposal: OpenedProposal, nearest: list[tuple[float, OpenedEntry]]) -> Decision:
        if not nearest:
            return Decision("add")
        if not self._models.can_decide:
            return Decision("no_change", 0) if nearest[0][0] >= SAME_FLOOR else Decision("add")
        try:
            decision, use = await self._models.decide(proposal.text, [entry.text for _, entry in nearest])
        except MemoryModelError as exc:
            await self._meter_use(proposal.workspace_id, proposal.provenance.api_key_id, exc.use)
            raise
        await self._meter_use(proposal.workspace_id, proposal.provenance.api_key_id, use)
        return decision

    async def _make_room(self, workspace_id: uuid.UUID) -> None:
        """Retire the least used entry while the workspace is at its cap."""
        if await self._repos.entries.count_active(workspace_id) < self._config.memory_max_entries_per_workspace:
            return
        oldest = await self._repos.entries.least_used(workspace_id)
        if oldest is not None:
            await self._repos.entries.set_status(workspace_id, oldest.id, "retired")
            logger.info("Memory of workspace %s is at its cap; retired its least used entry", workspace_id)

    async def _count_failed_pass(self, proposal: OpenedProposal) -> None:
        async with self._uow:
            attempts = await self._repos.proposals.count_attempt(proposal.workspace_id, proposal.id)
            if attempts >= MAX_ATTEMPTS:
                await self._repos.proposals.hold_for_review(proposal.workspace_id, proposal.id, "model")

    # -- Shared steps -------------------------------------------------------

    async def _switches(self, workspace_id: uuid.UUID) -> _Switches:
        switches = await self._switches_if_on(workspace_id)
        if switches is None:
            raise MemoryOffError
        return switches

    async def _switches_if_on(self, workspace_id: uuid.UUID) -> _Switches | None:
        """The workspace's switches while it and its organization both have memory on; None otherwise."""
        settings = await self._repos.settings.of_workspace(workspace_id)
        if settings is None or not settings.enabled:
            return None
        if not await self._repos.organizations.enabled(settings.organization_id):
            return None
        return _Switches(settings.workspace_scope_enabled, settings.hide_external, settings.write_policy)

    async def _visible_entry(self, caller: MemoryCaller, switches: _Switches, entry_id: uuid.UUID) -> OpenedEntry:
        entry = await self._store.entry(caller.workspace_id, entry_id)
        if entry is None or not _visible(caller, switches, entry):
            raise MemoryEntryNotFoundError
        return entry

    async def _signal(
        self, caller: MemoryCaller, entry_id: uuid.UUID, signal: Literal["confirm", "flag"], reason: str | None
    ) -> SignalReceipt:
        _require_write(caller)
        async with self._uow:
            switches = await self._switches(caller.workspace_id)
            entry = await self._visible_entry(caller, switches, entry_id)
            await self._repos.signals.put(caller.workspace_id, entry.id, caller.member_id, signal=signal, reason=reason)
            confirmations, flags = await self._repos.signals.counts(caller.workspace_id, entry.id)
            status: MemoryStatus = "quarantined" if flags >= QUARANTINE_FLAGS else "active"
            score = confidence(confirmations, flags)
            await self._repos.entries.set_signals(
                caller.workspace_id, entry.id, confirmations=confirmations, flags=flags, confidence=score, status=status
            )
        return SignalReceipt(entry.id, score, status)

    async def _record_read(self, caller: MemoryCaller, tool: str, entry_ids: list[uuid.UUID]) -> None:
        await self._repos.entries.touch(caller.workspace_id, entry_ids, self._now())
        await self._repos.reads.record(
            caller.workspace_id,
            reader_user_id=caller.member_id,
            api_key_id=caller.api_key_id,
            tool=tool,
            entry_ids=entry_ids,
        )

    async def _embed(self, texts: list[str], workspace_id: uuid.UUID, api_key_id: str | None) -> list[list[float]]:
        try:
            vectors, use = await self._models.embed(texts)
        except MemoryModelError as exc:
            await self._meter_use(workspace_id, api_key_id, exc.use)
            raise MemoryModelUnavailableError from None
        await self._meter_use(workspace_id, api_key_id, use)
        return vectors

    async def _meter_use(self, workspace_id: uuid.UUID, api_key_id: str | None, use: ModelUse | None) -> None:
        """Record one model call as the workspace's usage, in a step of its own that cannot fail the caller's."""
        if use is None:
            return
        try:
            async with self._uow.savepoint():
                await self._meter.record(workspace_id, api_key_id, use)
        except DATABASE_ERRORS as exc:
            logger.warning("A memory model call could not be recorded as usage: %s", describe_database_error(exc))


def _require_write(caller: MemoryCaller) -> None:
    if not caller.can_write:
        raise MemoryReadOnlyError


def _visible(caller: MemoryCaller, switches: _Switches, entry: OpenedEntry) -> bool:
    """Whether the caller may recall the entry: their own, or a shared one while the workspace shares."""
    if entry.scope == "member":
        return entry.owner_user_id == caller.member_id
    if not switches.workspace_scope:
        return False
    return not (switches.hide_external and entry.trust == "external")
