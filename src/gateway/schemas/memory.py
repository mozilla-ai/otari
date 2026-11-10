"""The arguments of memory's MCP tools, and the shapes their results take.

Every argument is validated here before a service is called, with unknown
fields refused, so a tool call can only carry what the tool reads. No argument
names a workspace or a member: those come from the caller's key.
"""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gateway.models.memory import MemoryKind, MemoryScope, MemoryStatus, MemoryTrust, MemoryWritePolicy

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
    entries: dict[str, int] = Field(description="Entry versions the workspace holds, by status.")
    proposals_waiting: int = Field(description="Proposals the consolidator has not decided yet.")


class WorkspaceMemoryUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    workspace_scope_enabled: bool = False
    write_policy: MemoryWritePolicy = "review"
    offer_to_every_request: bool = False
    hide_external: bool = False
    retention_days: int | None = Field(default=None, ge=1, le=3650)


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
