"""The arguments of memory's MCP tools, and the shapes their results take.

Every argument is validated here before a service is called, with unknown
fields refused, so a tool call can only carry what the tool reads. No argument
names a workspace or a member: those come from the caller's key.
"""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gateway.models.memory import (
    MemoryKind,
    MemoryReviewReason,
    MemoryScope,
    MemoryStatus,
    MemoryTrust,
    MemoryWritePolicy,
)

QUERY_MAX_LENGTH = 2_000
# The schema's own bound; the service holds a proposal to ``memory_entry_max_length``.
PROPOSAL_MAX_LENGTH = 16_000


class _Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchArguments(_Arguments):
    query: str = Field(min_length=1, max_length=QUERY_MAX_LENGTH, description="What to look for, in plain words.")
    limit: int = Field(default=5, ge=1, le=10, description="Most results to return.")


class EntryArguments(_Arguments):
    id: uuid.UUID = Field(description="The entry's id, as a search returned it.")


class ProposeArguments(_Arguments):
    text: str = Field(
        min_length=1,
        max_length=PROPOSAL_MAX_LENGTH,
        description="One fact, decision or procedure, stated so it is useful on its own to another agent later.",
    )
    kind: MemoryKind = Field(description="fact, decision or procedure.")
    scope: MemoryScope = Field(
        default="workspace",
        description="workspace shares it with every member's agents; member keeps it for this member's agents only.",
    )
    from_external_source: bool = Field(
        default=False,
        description="True when it rests on a web page, a fetched file or a third-party tool rather than your own work.",
    )


class FlagArguments(EntryArguments):
    reason: Literal["wrong", "stale", "duplicate", "unsafe"] = Field(description="Why the entry should not be trusted.")


class RecalledEntry(BaseModel):
    """One entry as a tool returns it: what it says, and where it came from."""

    id: uuid.UUID
    text: str
    kind: MemoryKind
    scope: MemoryScope
    trust: MemoryTrust
    confidence: float
    version: int
    created_at: datetime
    relevance: float | None = None


class EntryWithVersions(BaseModel):
    entry: RecalledEntry
    versions: list[RecalledEntry]


class ProposalAccepted(BaseModel):
    proposal_id: uuid.UUID
    state: Literal["pending"] = "pending"
    note: str = "Proposed. It becomes memory once the consolidator decides on it."


class SignalRecorded(BaseModel):
    id: uuid.UUID
    confidence: float
    status: MemoryStatus


class Forgotten(BaseModel):
    id: uuid.UUID
    versions_deleted: int


class WorkspaceMemoryPublic(BaseModel):
    """A workspace's memory switches, what the deployment allows, and how much the workspace holds."""

    workspace_id: uuid.UUID
    deployment_enabled: bool = Field(description="Whether this deployment offers memory at all.")
    organization_enabled: bool = Field(description="Whether the workspace's organization has memory on.")
    enabled: bool = Field(description="Memory on for this workspace: each member's own entries.")
    workspace_scope_enabled: bool = Field(description="The memory every member's agents share.")
    write_policy: MemoryWritePolicy = Field(
        description="auto keeps what the consolidator decides; review holds every shared proposal for an admin."
    )
    offer_to_every_request: bool = Field(
        description="Offer memory's tools to every request a member sends through the gateway, not only to one "
        "that asks for them."
    )
    hide_external: bool = Field(description="Keep entries resting on outside content out of recall.")
    retention_days: int | None = Field(description="Days an unused entry is kept; null keeps the deployment's.")
    deployment_retention_days: int
    auto_recall: bool = Field(
        description="Search memory for each request's newest user turn and add what is alike enough to it, "
        "with no tool call."
    )
    recall_min_similarity: float = Field(description="How alike an entry must be to the turn to be added, 0.3-0.99.")
    recall_max_entries: int = Field(description="The most entries automatic recall adds to one turn.")
    recall_rerank: bool = Field(
        description="Have the deployment's rerank model score the entries most alike to the turn, and add only "
        "those relevant enough; the similarity floor then no longer applies."
    )
    recall_min_relevance: float = Field(description="The score an entry must reach where relevance is checked.")
    rerank_available: bool = Field(
        description="Whether the deployment names a rerank model, without which relevance is not checked."
    )
    experiment_holdout_percent: int = Field(
        description="The share of sessions the running savings experiment serves without memory; 0 while none runs."
    )
    experiment_started_at: datetime | None = Field(description="When the last savings experiment began.")
    experiment_ended_at: datetime | None = Field(description="When it stopped; null while it runs.")
    entries: dict[str, int] = Field(description="Entry versions the workspace holds, by status.")
    proposals_waiting: int = Field(
        description="Proposals the consolidator has not decided yet, those an admin approved included."
    )
    proposals_held: int = Field(description="Shared proposals waiting for an admin's review.")


class WorkspaceMemoryUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    workspace_scope_enabled: bool = False
    write_policy: MemoryWritePolicy = "review"
    offer_to_every_request: bool = False
    hide_external: bool = False
    retention_days: int | None = Field(default=None, ge=1, le=3650)
    auto_recall: bool = False
    recall_min_similarity: float = Field(default=0.65, ge=0.3, le=0.99)
    recall_max_entries: int = Field(default=3, ge=1, le=5)
    recall_rerank: bool = False
    recall_min_relevance: float = Field(default=0.8, ge=0.05, le=0.99)
    experiment_holdout_percent: int = Field(
        default=0,
        ge=0,
        le=50,
        description="Serve this share of sessions without memory to measure what it saves. A new share starts "
        "the experiment afresh; 0 stops it and keeps its results.",
    )


class ArmSummaryPublic(BaseModel):
    """One arm of the savings experiment, per session."""

    sessions: int
    requests_per_session: float
    tokens_per_session: float = Field(description="Input and output tokens, cached input counted once.")
    median_tokens_per_session: float
    input_tokens_per_session: float
    output_tokens_per_session: float
    cache_read_tokens_per_session: float
    cost_per_session: float | None = Field(description="USD; null where a session's model has no price.")


class DifferencePublic(BaseModel):
    """The memory arm's mean less the control arm's, with a 95% interval. Negative is a saving."""

    absolute: float
    relative: float | None = Field(description="As a share of the control arm's mean.")
    low: float
    high: float
    conclusive: bool = Field(description="Whether the interval leaves out no difference at all.")


class MemoryCallsPublic(BaseModel):
    calls: int
    input_tokens: int
    output_tokens: int


class MemoryExperimentPublic(BaseModel):
    """What memory saves a workspace: sessions served without it against sessions served with it.

    Counts only requests that name their session, made through the gateway while the experiment ran.
    """

    workspace_id: uuid.UUID
    holdout_percent: int
    started_at: datetime | None
    ended_at: datetime | None
    min_sessions: int = Field(description="Sessions each arm needs before the differences are estimated.")
    control: ArmSummaryPublic
    memory: ArmSummaryPublic
    tokens: DifferencePublic | None = Field(
        description="Tokens per session, net of memory's own model calls; null until each arm has min_sessions."
    )
    cost: DifferencePublic | None = Field(description="USD per session; null without enough sessions or prices.")
    memory_calls: dict[str, MemoryCallsPublic] = Field(description="Memory's own model calls, by endpoint.")
    memory_call_tokens_per_session: float
    recalls: int = Field(description="Turns automatic recall searched for.")
    recalls_with_results: int = Field(description="Turns it added something to.")
    searches: int = Field(description="Searches agents ran with memory's tool.")


class ReviewedEntryPublic(BaseModel):
    """The entry a held proposal would change, as it reads now."""

    id: uuid.UUID
    text: str
    version: int
    trust: MemoryTrust


class HeldProposalPublic(BaseModel):
    """A shared proposal waiting for an admin, what held it, and where it came from."""

    id: uuid.UUID
    text: str
    kind: MemoryKind
    trust: MemoryTrust
    reason: MemoryReviewReason | None = Field(
        description="policy: the workspace reviews every shared proposal; conflict: it contradicts an entry; "
        "higher_trust: an agent's change to an entry an admin reviewed; guardrail: a block mandate flagged it; "
        "guardrail_unavailable and model: a mandated guardrail, or the consolidation model, could not run."
    )
    author_user_id: uuid.UUID | None = Field(description="The member whose agent proposed it.")
    session_id: str | None
    model: str | None = Field(description="The model the proposing agent ran on.")
    created_at: datetime
    expires_at: datetime = Field(description="When it is deleted if nobody decides on it.")
    entry: ReviewedEntryPublic | None = Field(
        description="The entry it would change: the one it contradicts, or the reviewed one it would replace. "
        "Approving it replaces this entry with it as a new version."
    )


class MemoryReviewQueuePublic(BaseModel):
    """A workspace's shared proposals waiting for an admin, oldest first. A member's own are never listed."""

    data: list[HeldProposalPublic]
    count: int


class WorkspaceMemoryErased(BaseModel):
    workspace_id: uuid.UUID
    entries: int
    proposals: int
    reads: int
    keys: int


class OrganizationMemoryPublic(BaseModel):
    """An organization's memory switch. Memory runs in a workspace while this, and the workspace's, are on."""

    organization_id: uuid.UUID
    deployment_enabled: bool = Field(description="Whether this deployment offers memory at all.")
    enabled: bool


class OrganizationMemoryUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
