"""The memory domain's one service: what agents search, propose, confirm, flag and forget, and the consolidator.

No model is called inside a Unit of Work block, so a slow provider never holds
a transaction open: each use case reads in one block, calls its model, and
writes in another.
"""

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol

from gateway.core.config import GatewayConfig
from gateway.core.database import DATABASE_ERRORS, describe_database_error
from gateway.core.unit_of_work import UnitOfWork
from gateway.exceptions.memory_exceptions import (
    MemoryApprovalRefusedError,
    MemoryEntryNotFoundError,
    MemoryModelUnavailableError,
    MemoryOffError,
    MemoryProposalNotFoundError,
    MemoryReadOnlyError,
    MemoryRefusedError,
    MemoryTooLongError,
    NotYoursToForgetError,
    WorkspaceScopeOffError,
)
from gateway.log_config import logger
from gateway.models.memory import (
    MemoryKind,
    MemoryReviewReason,
    MemoryScope,
    MemoryStatus,
    MemoryTrust,
    MemoryWorkspaceSettings,
    MemoryWritePolicy,
)
from gateway.models.tenancy import User
from gateway.repositories.memory import MemoryRepositories
from gateway.services.memory._experiment import (
    ArmSummary,
    Difference,
    ExperimentArm,
    SessionUsage,
    assign_arm,
    compare,
    summarize,
)
from gateway.services.memory._guardrails import GuardrailsUnavailableError, MandatedGuardrails, MemoryGuardrails
from gateway.services.memory._keys import WorkspaceKeys
from gateway.services.memory._models import (
    EMBEDDING_ENDPOINT,
    RECALL_ENDPOINT,
    Decision,
    DecisionAction,
    MemoryModelError,
    MemoryModels,
    ModelUse,
)
from gateway.services.memory._recall_vectors import RecallVectors
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
# How alike a proposal must be to an entry for the model's update or no_change of it to stand.
REPLACE_FLOOR = 0.85
# Passes a proposal may fail on a model, or on a mandated guardrail that could not run, before it waits for an
# admin instead.
MAX_ATTEMPTS = 5
# Automatic recall: how alike an entry must be to a request's newest turn, and how many are handed over. How alike
# is the embedding model's own scale: 0.65 suits nomic-embed-text, and another model may want another floor.
DEFAULT_RECALL_MIN_SIMILARITY = 0.65
DEFAULT_RECALL_MAX_ENTRIES = 3
# How far below the most alike entry another may be and still be handed over: one alike only in a name the
# question mentions falls well below what the question is about.
RECALL_MARGIN = 0.1
# Where a workspace checks relevance: the score an entry must reach by default, on qwen3-reranker's scale, and how
# many of the most alike entries the rerank model scores, whatever their similarity, since it is the check.
DEFAULT_RECALL_MIN_RELEVANCE = 0.8
RERANK_CANDIDATES = 10
# The read records automatic recall and the search tool leave, which the experiment counts.
RECALL_TOOL = "auto_recall"
SEARCH_TOOL = "memory_search"

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
class MemoryCallUsage:
    """What memory's own model calls of one kind used: embedding, recall or consolidation."""

    calls: int
    input_tokens: int
    output_tokens: int


class UsageLedger(Protocol):
    """What a workspace's requests used, read for the savings experiment. The builder in ``api/deps`` implements it."""

    async def sessions(self, workspace_id: uuid.UUID, since: datetime, until: datetime) -> dict[str, SessionUsage]:
        """Each session's usage of the workspace's requests in the window, by its trace id; side calls left out."""
        ...

    async def memory_calls(
        self, workspace_id: uuid.UUID, since: datetime, until: datetime
    ) -> dict[str, MemoryCallUsage]:
        """Memory's own model calls in the window, by their endpoint."""
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
class MemoryOffer:
    """What a request through the gateway may get: memory's tools when it asks, or whether it asks or not.

    ``auto_recall`` hands it what memory holds on its newest turn, unasked. ``arm`` is the experiment's arm its
    session is in; the control arm gets none of it.
    """

    on: bool
    every_request: bool
    auto_recall: bool = False
    arm: ExperimentArm | None = None


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
    offer_to_every_request: bool
    hide_external: bool
    retention_days: int | None
    auto_recall: bool
    recall_min_similarity: float
    recall_max_entries: int
    recall_rerank: bool
    recall_min_relevance: float
    # The running experiment's holdout, zero while none runs; the last one's window is kept once it stops.
    experiment_holdout_percent: int
    experiment_started_at: datetime | None
    experiment_ended_at: datetime | None
    entries: dict[str, int]
    # Proposals the consolidator has yet to decide, those an admin approved included; and the shared ones it held
    # for an admin, which the review queue lists.
    proposals_waiting: int
    proposals_held: int


@dataclass(frozen=True)
class HeldProposal:
    """A shared proposal waiting for an admin, opened, with when it is deleted undecided and the entry it would change.

    ``entry`` is the entry the consolidator named: the one it contradicts, or the reviewed one it would replace.
    None where it named none, or that entry is gone.
    """

    proposal: OpenedProposal
    expires_at: datetime
    entry: OpenedEntry | None


@dataclass(frozen=True)
class SwitchChange:
    enabled: bool
    workspace_scope_enabled: bool
    write_policy: MemoryWritePolicy
    offer_to_every_request: bool
    hide_external: bool
    retention_days: int | None
    auto_recall: bool = False
    recall_min_similarity: float = DEFAULT_RECALL_MIN_SIMILARITY
    recall_max_entries: int = DEFAULT_RECALL_MAX_ENTRIES
    # Checked only where the deployment names a rerank model; kept either way, so naming one later honours it.
    recall_rerank: bool = False
    recall_min_relevance: float = DEFAULT_RECALL_MIN_RELEVANCE
    experiment_holdout_percent: int = 0


@dataclass(frozen=True)
class ExperimentReport:
    """The savings experiment of one workspace: each arm's sessions, and what memory changed, net of its own calls."""

    holdout_percent: int
    started_at: datetime | None
    ended_at: datetime | None
    control: ArmSummary
    memory: ArmSummary
    # Tokens and cost per session, the memory arm's mean less the control arm's; None until each arm has enough.
    tokens: Difference | None
    cost: Difference | None
    # Memory's own model calls since the experiment began, by kind, which a memory session costs on top.
    memory_calls: dict[str, MemoryCallUsage]
    memory_call_tokens_per_session: float
    # How often automatic recall found something, and searches agents ran, since the experiment began.
    recalls: int
    recalls_with_results: int
    searches: int


@dataclass(frozen=True)
class _Switches:
    """A workspace's memory switches, read once and held past the block that read them."""

    workspace_scope: bool
    hide_external: bool
    write_policy: MemoryWritePolicy
    organization_id: uuid.UUID


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
        ledger: UsageLedger | None = None,
        recall_vectors: RecallVectors | None = None,
        guardrails: MemoryGuardrails | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._uow = uow
        self._repos = repos
        self._store = EntryStore(repos, keys)
        self._models = models
        self._meter = meter
        self._config = config
        self._managers = managers
        self._ledger = ledger
        # The embeddings automatic recall opened, shared by every request of the process; None opens them each turn.
        self._recall_vectors = recall_vectors
        # The guardrails the organization mandates, run over each proposal; None runs none.
        self._guardrails = guardrails
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

    async def offer(self, caller: MemoryCaller, session_label: str | None = None) -> MemoryOffer:
        """What memory the caller's request gets, by its workspace's switches and the experiment arm of its session.

        ``session_label`` is the trace id of the request's session, None where it names none.
        """
        async with self._uow:
            settings = await self._repos.settings.of_workspace(caller.workspace_id)
            if await self._switches_if_on(caller.workspace_id) is None or settings is None:
                return MemoryOffer(on=False, every_request=False)
            running = settings.experiment_started_at if settings.experiment_ended_at is None else None
            arm = assign_arm(caller.workspace_id, running, session_label, settings.experiment_holdout_percent)
            return MemoryOffer(
                on=True,
                every_request=settings.offer_to_every_request,
                auto_recall=settings.auto_recall,
                arm=arm,
            )

    async def recall(self, caller: MemoryCaller, query: str, *, context: str | None = None) -> list[Recalled]:
        """What the caller may recall that is alike enough to ``query``, best first, for a request's newest turn.

        ``context`` is the user's turns before it. An entry is as alike as the more alike of the turn alone and the
        turn read after ``context``: a follow-up finds what its conversation is about, and a question that changes
        the subject is not held to the old one.

        Where the workspace checks relevance, the deployment's rerank model scores the entries most alike against
        the turn, with ``context`` after it as an aside, and those reaching the workspace's score are handed over.
        The turn comes first: led by the conversation before it, a reranker favours entries about that
        conversation even once the subject changed. Otherwise, or when the check fails, entries clearing the
        similarity floor are, less those well below the most alike.

        Empty where the workspace does not recall automatically. Only what is handed over is recorded as read.

        Raises:
            MemoryOffError: the workspace has not turned memory on.
            MemoryModelUnavailableError: the query could not be embedded.
        """
        async with self._uow:
            switches = await self._switches(caller.workspace_id)
            settings = await self._repos.settings.of_workspace(caller.workspace_id)
            if settings is None or not settings.auto_recall:
                return []
            floor, most = settings.recall_min_similarity, settings.recall_max_entries
            min_relevance = settings.recall_min_relevance if settings.recall_rerank else None
        texts = [query] if not context else [query, f"{context}\n\n{query}"]
        vectors = await self._embed(texts, caller.workspace_id, caller.api_key_id, endpoint=RECALL_ENDPOINT)
        model = self._models.embedding_model
        if model is None:
            return []
        async with self._uow:
            candidates = await self._store.recallable_vectors(
                caller.workspace_id,
                member_id=caller.member_id,
                include_workspace=switches.workspace_scope,
                include_external=not switches.hide_external,
                embedding_model=model,
                held=self._recall_vectors,
            )
        alike = sorted(
            ((max(cosine(vector, candidate.embedding) for vector in vectors), candidate) for candidate in candidates),
            key=lambda scored: scored[0],
            reverse=True,
        )
        # Each kept entry's score, id and confidence; texts are opened only for what the reranker reads and what
        # is handed over.
        kept: list[tuple[float, uuid.UUID, float]] | None = None
        opened: dict[uuid.UUID, OpenedEntry] = {}
        if min_relevance is not None and self._models.can_rerank:
            asked = query if not context else f"{query}\n(Said earlier in the conversation: {context})"
            checked = [candidate.id for _, candidate in alike[:RERANK_CANDIDATES]]
            async with self._uow:
                opened = await self._store.opened(caller.workspace_id, checked)
            relevant = await self._relevant(
                caller, asked, [opened[entry_id] for entry_id in checked if entry_id in opened]
            )
            if relevant is not None:
                kept = [(score, entry.id, entry.confidence) for score, entry in relevant if score >= min_relevance]
        if kept is None:
            best = alike[0][0] if alike else floor
            kept = [
                (similarity, candidate.id, candidate.confidence)
                for similarity, candidate in alike
                if similarity >= max(floor, best - RECALL_MARGIN)
            ]
        chosen = sorted(kept, key=lambda scored: relevance(scored[0], scored[2]), reverse=True)[:most]
        if not chosen:
            return []
        async with self._uow:
            unopened = [entry_id for _, entry_id, _ in chosen if entry_id not in opened]
            if unopened:
                opened.update(await self._store.opened(caller.workspace_id, unopened))
            ranked = [
                Recalled(opened[entry_id], relevance(score, confidence))
                for score, entry_id, confidence in chosen
                if entry_id in opened
            ]
            if ranked:
                await self._record_read(caller, RECALL_TOOL, [recalled.entry.id for recalled in ranked])
        return ranked

    async def _relevant(
        self, caller: MemoryCaller, query: str, candidates: list[OpenedEntry]
    ) -> list[tuple[float, OpenedEntry]] | None:
        """Each candidate with how relevant the rerank model scores it to ``query``; None when it could not."""
        if not candidates:
            return []
        try:
            scores, use = await self._models.rerank(query, [entry.text for entry in candidates])
        except MemoryModelError as exc:
            await self._meter_use(caller.workspace_id, caller.api_key_id, exc.use)
            logger.warning("Automatic recall could not check relevance; it falls back to similarity")
            return None
        await self._meter_use(caller.workspace_id, caller.api_key_id, use)
        return list(zip(scores, candidates, strict=True))

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
            current = await self._repos.settings.of_workspace(workspace_id)
            await self._repos.settings.put(
                workspace_id,
                organization_id=organization_id,
                enabled=change.enabled,
                workspace_scope_enabled=change.workspace_scope_enabled,
                write_policy=change.write_policy,
                offer_to_every_request=change.offer_to_every_request,
                hide_external=change.hide_external,
                retention_days=change.retention_days,
                auto_recall=change.auto_recall,
                recall_min_similarity=change.recall_min_similarity,
                recall_max_entries=change.recall_max_entries,
                rerank=(change.recall_rerank, change.recall_min_relevance),
                experiment=self._experiment_after(current, change.experiment_holdout_percent),
            )
            return await self._workspace_memory(workspace_id, organization_id)

    async def experiment_report(self, user: User, workspace_id: uuid.UUID) -> ExperimentReport:
        """The workspace's savings experiment: the running one, or the last one once it stopped.

        Only sessions and calls inside its window are counted. Reading it takes a manager.
        """
        await self._managers_or_fail().require_manager(user, workspace_id)
        if self._ledger is None:
            msg = "this memory service was built without a usage ledger"
            raise RuntimeError(msg)
        async with self._uow:
            settings = await self._repos.settings.of_workspace(workspace_id)
        if settings is None or settings.experiment_started_at is None:
            empty = summarize([])
            return ExperimentReport(0, None, None, empty, empty, None, None, {}, 0.0, 0, 0, 0)
        holdout, started_at, ended_at = (
            settings.experiment_holdout_percent,
            settings.experiment_started_at,
            settings.experiment_ended_at,
        )
        until = ended_at or self._now()
        sessions = await self._ledger.sessions(workspace_id, started_at, until)
        calls = await self._ledger.memory_calls(workspace_id, started_at, until)
        arms: dict[ExperimentArm, list[SessionUsage]] = {"control": [], "memory": []}
        for label, usage in sessions.items():
            arm = assign_arm(workspace_id, started_at, label, holdout)
            if arm is not None:
                arms[arm].append(usage)
        control, with_memory = arms["control"], arms["memory"]
        call_tokens = sum(use.input_tokens + use.output_tokens for use in calls.values())
        per_session = call_tokens / len(with_memory) if with_memory else 0.0
        async with self._uow:
            reads = await self._repos.reads.count_by_tool(workspace_id, since=started_at, until=until)
        recall_calls = calls.get(RECALL_ENDPOINT)
        return ExperimentReport(
            holdout_percent=holdout,
            started_at=started_at,
            ended_at=ended_at,
            control=summarize(control),
            memory=summarize(with_memory),
            # A memory session is charged its share of memory's own calls, so the difference is net of them.
            tokens=compare([s.tokens for s in control], [s.tokens + per_session for s in with_memory]),
            cost=(
                compare([s.cost or 0.0 for s in control], [s.cost or 0.0 for s in with_memory])
                if all(s.cost is not None for s in (*control, *with_memory))
                else None
            ),
            memory_calls=calls,
            memory_call_tokens_per_session=per_session,
            recalls=recall_calls.calls if recall_calls is not None else 0,
            recalls_with_results=reads.get(RECALL_TOOL, 0),
            searches=reads.get(SEARCH_TOOL, 0),
        )

    def _experiment_after(
        self, current: MemoryWorkspaceSettings | None, holdout_percent: int
    ) -> tuple[int, datetime | None, datetime | None]:
        """The experiment's holdout, start and end once a change sets ``holdout_percent``.

        Zero stops a running experiment and keeps its window, so its results still read. The holdout it runs
        with keeps it running; another one starts it afresh, dealing every session again.
        """
        if current is None or current.experiment_started_at is None:
            return (holdout_percent, self._now(), None) if holdout_percent > 0 else (0, None, None)
        kept = (current.experiment_holdout_percent, current.experiment_started_at, current.experiment_ended_at)
        running = current.experiment_ended_at is None
        if holdout_percent <= 0:
            return (*kept[:2], self._now()) if running else kept
        if running and current.experiment_holdout_percent == holdout_percent:
            return kept
        return holdout_percent, self._now(), None

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
        if self._recall_vectors is not None:
            self._recall_vectors.forget(workspace_id)
        logger.info("Memory of workspace %s was erased by a manager", workspace_id)
        return Swept(entries, proposals, reads, len(keys))

    # -- The review queue ---------------------------------------------------

    async def review_queue(self, user: User, workspace_id: uuid.UUID) -> list[HeldProposal]:
        """The workspace's shared proposals waiting for an admin, oldest first, with the entries they would change.

        A member's own proposals are never listed: no admin reads a member's memory, so none is held for one.
        """
        await self._managers_or_fail().require_manager(user, workspace_id)
        async with self._uow:
            held = await self._store.held_proposals(workspace_id)
            entries = await self._store.opened(
                workspace_id, [proposal.review_entry_id for proposal in held if proposal.review_entry_id is not None]
            )
        max_age = timedelta(hours=self._config.memory_proposal_max_age_hours)
        return [
            HeldProposal(
                proposal=proposal,
                expires_at=proposal.created_at + max_age,
                entry=entries.get(proposal.review_entry_id) if proposal.review_entry_id is not None else None,
            )
            for proposal in held
        ]

    async def approve(self, user: User, workspace_id: uuid.UUID, proposal_id: uuid.UUID) -> None:
        """Vouch for a held shared proposal: it goes back to the consolidator as ``reviewed``.

        Neither the review policy nor a mandated guardrail holds it again, and it outranks what agents wrote. One
        held over an entry (a conflict, or a change to a reviewed entry) replaces that entry as its next version.
        What memory never keeps is still refused, whoever vouches for it.

        Raises:
            MemoryProposalNotFoundError: the workspace has no shared proposal waiting for review by that id.
            MemoryApprovalRefusedError: the text carries a credential or personal data.
        """
        await self._managers_or_fail().require_manager(user, workspace_id)
        async with self._uow:
            proposal = await self._store.held_proposal(workspace_id, proposal_id)
            if proposal is None:
                raise MemoryProposalNotFoundError
            finding = screen(proposal.text)
            if finding is not None:
                raise MemoryApprovalRefusedError(finding)
            if not await self._repos.proposals.approve(workspace_id, proposal_id):
                raise MemoryProposalNotFoundError
        logger.info(
            "Memory proposal %s of workspace %s held for %s was approved by %s",
            proposal_id,
            workspace_id,
            proposal.review_reason,
            user.id,
        )

    async def discard(self, user: User, workspace_id: uuid.UUID, proposal_id: uuid.UUID) -> None:
        """Delete a held shared proposal. Nothing of it is kept.

        Raises:
            MemoryProposalNotFoundError: the workspace has no shared proposal waiting for review by that id.
        """
        await self._managers_or_fail().require_manager(user, workspace_id)
        async with self._uow:
            if not await self._repos.proposals.remove_held(workspace_id, proposal_id, scope="workspace"):
                raise MemoryProposalNotFoundError
        logger.info("Memory proposal %s of workspace %s was discarded by %s", proposal_id, workspace_id, user.id)

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
            offer_to_every_request=settings is not None and settings.offer_to_every_request,
            hide_external=settings is not None and settings.hide_external,
            retention_days=settings.retention_days if settings is not None else None,
            auto_recall=settings is not None and settings.auto_recall,
            recall_min_similarity=(
                settings.recall_min_similarity if settings is not None else DEFAULT_RECALL_MIN_SIMILARITY
            ),
            recall_max_entries=settings.recall_max_entries if settings is not None else DEFAULT_RECALL_MAX_ENTRIES,
            recall_rerank=settings is not None and settings.recall_rerank,
            recall_min_relevance=(
                settings.recall_min_relevance if settings is not None else DEFAULT_RECALL_MIN_RELEVANCE
            ),
            experiment_holdout_percent=(
                settings.experiment_holdout_percent
                if settings is not None and settings.experiment_started_at and settings.experiment_ended_at is None
                else 0
            ),
            experiment_started_at=settings.experiment_started_at if settings is not None else None,
            experiment_ended_at=settings.experiment_ended_at if settings is not None else None,
            entries=await self._repos.entries.count_by_status(workspace_id),
            proposals_waiting=await self._repos.proposals.count_pending(workspace_id),
            proposals_held=await self._repos.proposals.count_held(workspace_id, scope="workspace"),
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

        The guardrails the organization mandates for the workspace are read once, and run over each proposal
        before anything else is. A proposal whose model call fails, or that a mandate failing closed could not
        check, is left for a later pass, and after ``MAX_ATTEMPTS`` waits for an admin; a member's own is
        discarded instead. A proposal another replica decided first is skipped.
        """
        async with self._uow:
            switches = await self._switches_if_on(workspace_id)
            if switches is None:
                return 0
            proposals = await self._store.pending_proposals(workspace_id, limit=limit)
        if not proposals:
            return 0
        try:
            mandated = (
                await self._guardrails.mandated(switches.organization_id, workspace_id)
                if self._guardrails is not None
                else None
            )
        except GuardrailsUnavailableError:
            # Nothing is decided unchecked: each proposal waits for a later pass, as on a failed model call. One an
            # admin approved is not checked, so it goes on.
            for proposal in proposals:
                if proposal.trust != "reviewed":
                    await self._count_failed_pass(proposal, "guardrail_unavailable")
            proposals = [proposal for proposal in proposals if proposal.trust == "reviewed"]
            mandated = None
        decided = 0
        for proposal in proposals:
            try:
                decided += await self._consolidate_one(proposal, switches, mandated)
            except (MemoryModelUnavailableError, MemoryModelError):
                await self._count_failed_pass(proposal, "model")
            except GuardrailsUnavailableError:
                logger.warning(
                    "Memory proposal %s of workspace %s could not be checked by its mandated guardrails; "
                    "it waits for a later pass",
                    proposal.id,
                    proposal.workspace_id,
                )
                await self._count_failed_pass(proposal, "guardrail_unavailable")
        return decided

    async def _consolidate_one(
        self, proposal: OpenedProposal, switches: _Switches, mandated: MandatedGuardrails | None
    ) -> int:
        """Decide one proposal: keep it, apply it over an entry, or leave it for an admin.

        One an admin approved (``reviewed``) is held neither by the review policy nor by a mandate again; held over
        an entry the admin saw, it replaces that entry without the model being asked, while it is still active.
        A member's own proposal never waits for an admin, who may not read it: see :meth:`_hold`.
        """
        workspace_id = proposal.workspace_id
        approved = proposal.trust == "reviewed"
        if mandated is not None and not approved and not await self._passes_guardrails(proposal, mandated):
            async with self._uow:
                await self._hold(proposal, "guardrail")
            return 1
        if proposal.scope == "workspace" and switches.write_policy == "review" and not approved:
            async with self._uow:
                await self._hold(proposal, "policy")
            return 1
        [vector] = await self._embed([proposal.text], workspace_id, proposal.provenance.api_key_id)
        model = self._models.embedding_model or ""
        async with self._uow:
            candidates = await self._store.comparable(
                workspace_id, scope=proposal.scope, owner_user_id=proposal.owner_user_id, embedding_model=model
            )
        approved_over = (
            next((entry for entry in candidates if entry.id == proposal.review_entry_id), None) if approved else None
        )
        if approved_over is not None:
            action: DecisionAction = "update"
            target: OpenedEntry | None = approved_over
        else:
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
            action = decision.action
            target = nearest[decision.target][1] if decision.target is not None else None
            outranked = (
                action == "update" and target is not None and _TRUST_RANK[target.trust] > _TRUST_RANK[proposal.trust]
            )
            if action == "conflict" or outranked:
                if proposal.scope == "member":
                    # A member's own memory has no admin to settle it, and the member is its authority: their newer
                    # word replaces what it contradicts, where the model named an entry alike enough to replace.
                    # One resting on less trusted ground than the entry it would change is kept beside it instead.
                    alike = nearest[decision.target][0] if decision.target is not None else 0.0
                    replaces = action == "conflict" and target is not None and alike >= REPLACE_FLOOR
                    action, target = ("update", target) if replaces else ("add", None)
                elif approved and action == "conflict" and target is None:
                    # An admin vouched for it, and the model named nothing it would replace: it is kept beside.
                    action = "add"
                else:
                    async with self._uow:
                        await self._hold(
                            proposal,
                            "conflict" if action == "conflict" else "higher_trust",
                            entry_id=target.id if target is not None else None,
                        )
                    return 1
        async with self._uow:
            # Removed first, in the block that applies it: a replica that finds it gone applies nothing.
            if not await self._repos.proposals.remove(workspace_id, proposal.id):
                return 0
            if action == "no_change" and target is not None:
                await self._repos.entries.touch(workspace_id, [target.id], self._now())
                return 1
            if action == "add":
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

    async def _passes_guardrails(self, proposal: OpenedProposal, mandated: MandatedGuardrails) -> bool:
        """Whether no ``block`` mandate flags the proposal.

        Run before it is embedded, so a refused text reaches no model. What flagged it is logged by profile,
        never with the text.
        """
        outcome = await mandated.check(proposal.text)
        if outcome.unchecked:
            logger.info(
                "Memory proposal %s of workspace %s went on unchecked by guardrails %s, which gave no verdict",
                proposal.id,
                proposal.workspace_id,
                ", ".join(outcome.unchecked),
            )
        if outcome.monitored:
            logger.info(
                "Memory proposal %s of workspace %s was flagged by monitored guardrails %s",
                proposal.id,
                proposal.workspace_id,
                ", ".join(outcome.monitored),
            )
        if outcome.blocked:
            logger.info(
                "Memory proposal %s of workspace %s was refused by guardrails %s",
                proposal.id,
                proposal.workspace_id,
                ", ".join(outcome.blocked),
            )
        return not outcome.blocked

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
        guarded = guarded_decision(decision, [similarity for similarity, _ in nearest])
        if guarded != decision:
            logger.info(
                "Memory proposal %s of workspace %s is added: the model's %s named an entry too unlike it",
                proposal.id,
                proposal.workspace_id,
                decision.action,
            )
        return guarded

    async def _make_room(self, workspace_id: uuid.UUID) -> None:
        """Retire the least used entry while the workspace is at its cap."""
        if await self._repos.entries.count_active(workspace_id) < self._config.memory_max_entries_per_workspace:
            return
        oldest = await self._repos.entries.least_used(workspace_id)
        if oldest is not None:
            await self._repos.entries.set_status(workspace_id, oldest.id, "retired")
            logger.info("Memory of workspace %s is at its cap; retired its least used entry", workspace_id)

    async def _count_failed_pass(self, proposal: OpenedProposal, reason: MemoryReviewReason) -> None:
        """Count a pass that could not decide the proposal; at ``MAX_ATTEMPTS`` it waits for an admin for ``reason``."""
        async with self._uow:
            attempts = await self._repos.proposals.count_attempt(proposal.workspace_id, proposal.id)
            if attempts >= MAX_ATTEMPTS:
                await self._hold(proposal, reason)

    async def _hold(
        self, proposal: OpenedProposal, reason: MemoryReviewReason, *, entry_id: uuid.UUID | None = None
    ) -> None:
        """Leave a shared proposal for an admin, in the open block; a member's own is discarded instead.

        No admin reads a member's memory, so a member's proposal held for one would only wait to be swept: one a
        mandate refused, or that a model or mandate could not decide pass after pass, is deleted, and logged by id.
        """
        if proposal.scope == "member":
            await self._repos.proposals.remove(proposal.workspace_id, proposal.id)
            logger.info(
                "Memory proposal %s of workspace %s is a member's own, which no admin reviews; it was discarded (%s)",
                proposal.id,
                proposal.workspace_id,
                reason,
            )
            return
        await self._repos.proposals.hold_for_review(proposal.workspace_id, proposal.id, reason, entry_id=entry_id)

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
        return _Switches(
            settings.workspace_scope_enabled, settings.hide_external, settings.write_policy, settings.organization_id
        )

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

    async def _embed(
        self, texts: list[str], workspace_id: uuid.UUID, api_key_id: str | None, *, endpoint: str = EMBEDDING_ENDPOINT
    ) -> list[list[float]]:
        try:
            vectors, use = await self._models.embed(texts, endpoint=endpoint)
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


def guarded_decision(decision: Decision, alike: Sequence[float]) -> Decision:
    """``decision``, unless it replaces an entry with the proposal, or drops the proposal as one, too unlike it.

    ``alike`` is how alike the proposal is to each entry the model was shown, in order. A model, a small one
    most, takes two notes that share a name or a place for one; adding the proposal beside it costs a little
    room, while replacing or dropping what should have stayed apart loses it from recall.
    """
    if decision.action in ("update", "no_change") and decision.target is not None:
        if alike[decision.target] < REPLACE_FLOOR:
            return Decision("add")
    return decision


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
