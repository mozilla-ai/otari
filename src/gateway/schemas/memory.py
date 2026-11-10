"""The arguments of memory's MCP tools, and the shapes their results take.

Every argument is validated here before a service is called, with unknown
fields refused, so a tool call can only carry what the tool reads. No argument
names a workspace or a member: those come from the caller's key.
"""

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from gateway.models.memory import MemoryKind, MemoryScope, MemoryStatus, MemoryTrust

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
