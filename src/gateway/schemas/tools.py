"""Request and response shapes for an organization's web search keys and their workspace overrides."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlmodel import Field, SQLModel

from gateway.models.tools import OrgWebSearchKey


class OrgWebSearchKeyCreateRequest(SQLModel):
    """What a caller sends to add a key. The service keeps only its ciphertext and ``last4``."""

    provider: str = Field(max_length=255, description="The search provider the key is for: tavily or brave.")
    name: str = Field(max_length=255, description="How the organization tells this key apart from its others.")
    api_key: str = Field(min_length=1)


class OrgWebSearchKeyUpdateRequest(SQLModel):
    """A partial update: only what is set is applied."""

    name: str | None = Field(default=None, max_length=255)
    api_key: str | None = Field(default=None, min_length=1)


class OrgWebSearchKeyPublic(SQLModel):
    """One key as the API shows it: never the key itself, only ``last4``."""

    id: uuid.UUID
    organization_id: uuid.UUID
    provider: str
    name: str
    last4: str | None = None
    is_org_default: bool
    usable: bool = Field(
        description=(
            "False when this deployment cannot decrypt the stored key, so no search uses it. "
            "It is still listed, because replacing or deleting it is what fixes it."
        )
    )
    archived_at: datetime | None = None
    created_at: datetime
    updated_at: datetime | None = None

    @classmethod
    def from_row(cls, key: OrgWebSearchKey, *, usable: bool) -> OrgWebSearchKeyPublic:
        """Read one row for the API; ``usable`` comes from the service, which owns decryption."""
        return cls(
            id=key.id,
            organization_id=key.organization_id,
            provider=key.provider,
            name=key.name,
            last4=key.last4,
            is_org_default=key.is_org_default,
            usable=usable,
            archived_at=key.archived_at,
            created_at=key.created_at,
            updated_at=key.updated_at,
        )


class OrgWebSearchKeysPublic(SQLModel):
    data: list[OrgWebSearchKeyPublic]
    count: int


class WorkspaceWebSearchKeyOverrideRequest(SQLModel):
    """Tri-state: an omitted flag keeps its value.

    Pinning a key re-enables it and unpins any other key of the workspace, and turning a
    key off unpins it. Sending both flags true is refused. Both false deletes the override.
    """

    is_default: bool | None = None
    disabled: bool | None = None


class WorkspaceWebSearchKeyPublic(SQLModel):
    """One of the organization's keys, as one workspace sees it."""

    org_web_search_key_id: uuid.UUID
    workspace_id: uuid.UUID
    provider: str
    name: str
    last4: str | None = None
    is_default: bool = Field(description="The workspace pinned this key as its own.")
    disabled: bool = Field(description="The workspace turned this key off.")
    is_effective: bool = Field(description="This is the key the workspace's searches use.")
    usable: bool


class WorkspaceWebSearchKeysPublic(SQLModel):
    data: list[WorkspaceWebSearchKeyPublic]
