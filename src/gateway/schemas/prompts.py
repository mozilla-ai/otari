"""Request and response schemas for the prompts domain."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PromptCreate(BaseModel):
    """A prompt to store."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Unique name callers use to reference this prompt.")
    content: str = Field(description="The prompt text.")


class PromptUpdate(BaseModel):
    """Fields to change on a stored prompt. An omitted field keeps its value."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, description="New name for the prompt.")
    content: str | None = Field(default=None, description="New prompt text.")


class PromptResponse(BaseModel):
    """A stored prompt."""

    id: uuid.UUID
    name: str
    content: str
    created_at: datetime
    updated_at: datetime
