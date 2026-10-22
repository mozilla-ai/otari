"""What a completion request's preamble resolved that its attachments are normalized against."""

import uuid
from dataclasses import dataclass

from any_llm import LLMProvider

from gateway.models.tools import CodeExecutor


@dataclass(frozen=True, slots=True)
class NormalizationTarget:
    """The resolved facts a request's attachments are normalized against.

    ``file_workspace_id`` is the caller's own, so a key confined to one
    workspace reaches only its files, and it is ``None`` for the master key,
    which sees every workspace.
    ``candidate_providers`` is the provider of every candidate the request may
    dispatch to, head first, which is what the code-execution executor decides against.
    ``upstream_user`` is the end-user identifier the request's provider calls carry,
    which a side-call the normalizer makes carries too.
    """

    user_id: str
    provider: LLMProvider | None
    model: str
    instance: str | None
    file_workspace_id: uuid.UUID | None
    workspace_executor: CodeExecutor | None
    candidate_providers: tuple[LLMProvider | None, ...]
    upstream_user: str | None
