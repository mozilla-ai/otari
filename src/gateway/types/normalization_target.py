"""What a completion request's preamble resolved that its attachments are normalized against."""

import uuid
from dataclasses import dataclass

from any_llm import LLMProvider

from gateway.models.tools import CodeExecutor


@dataclass(frozen=True, slots=True)
class NormalizationTarget:
    """The resolved facts a request's attachments are normalized against.

    Two workspaces, because they answer different questions.
    ``file_workspace_id`` is the caller's own, so a key confined to one
    workspace reaches only its files, and it is ``None`` for the master key,
    which sees every workspace.
    ``credential_workspace_id`` is the one the request dispatches under, so
    anything done at a provider is done with the credential the request will use.
    """

    user_id: str
    provider: LLMProvider | None
    model: str
    instance: str | None
    file_workspace_id: uuid.UUID | None
    credential_workspace_id: uuid.UUID | None
    workspace_executor: CodeExecutor | None
