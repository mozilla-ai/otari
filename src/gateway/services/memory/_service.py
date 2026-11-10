"""The memory domain's one service: what agents search, propose, confirm, flag and forget, and the consolidator.

No model is called inside a Unit of Work block, so a slow provider never holds
a transaction open: each use case reads in one block, calls its model, and
writes in another.
"""

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from gateway.core.config import GatewayConfig
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
from gateway.repositories.memory import MemoryRepositories
from gateway.services.memory._keys import WorkspaceKeys
from gateway.services.memory._models import Decision, MemoryModelError, MemoryModels
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
        config: GatewayConfig,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._uow = uow
        self._repos = repos
        self._store = EntryStore(repos, keys)
        self._models = models
        self._config = config
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
        [vector] = await self._embed([query])
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

    # -- The consolidator ---------------------------------------------------

    async def workspaces_with_memory(self) -> list[uuid.UUID]:
        """Every workspace that has memory on."""
        async with self._uow:
            return list(await self._repos.settings.enabled_workspaces())

    async def consolidate(self, workspace_id: uuid.UUID, *, limit: int) -> int:
        """Decide up to ``limit`` of the workspace's pending proposals; how many were decided.

        A proposal whose model call fails is left for a later pass, and after
        ``MAX_ATTEMPTS`` waits for an admin. A proposal another replica decided
        first is skipped.
        """
        async with self._uow:
            settings = await self._repos.settings.of_workspace(workspace_id)
            if settings is None or not settings.enabled:
                return 0
            switches = _Switches(settings.workspace_scope_enabled, settings.hide_external, settings.write_policy)
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
        [vector] = await self._embed([proposal.text])
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
        return await self._models.decide(proposal.text, [entry.text for _, entry in nearest])

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
        settings = await self._repos.settings.of_workspace(workspace_id)
        if settings is None or not settings.enabled:
            raise MemoryOffError
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

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            return await self._models.embed(texts)
        except MemoryModelError:
            raise MemoryModelUnavailableError from None


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
