"""What a completion request's preamble resolved that its attachments are normalized against."""

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from any_llm import LLMProvider

from gateway.models.tools import CodeExecutor
from gateway.types.provider_account import ResolvedCredential


@dataclass(frozen=True, slots=True)
class NormalizationTarget:
    """The resolved facts a request's attachments are normalized against.

    Two workspaces, because they answer different questions.
    ``file_workspace_id`` is the caller's own, so a key confined to one
    workspace reaches only its files, and it is ``None`` for the master key,
    which sees every workspace.
    ``credential_workspace_id`` is the one the request dispatches under, so
    anything done at a provider is done with the credential the request will use.
    ``candidate_providers`` is the provider of every candidate the request may
    dispatch to, head first, which is what the code-execution executor decides against.
    ``upstream_user`` is the end-user identifier the request's provider calls carry,
    which a side-call the normalizer makes carries too.
    ``hosted_credential`` resolves the deployment's hosted provider for the head
    candidate, the rung dispatch falls back to when no stored key serves it, so
    an attachment moved to the provider ahead of the call uses the same key.
    """

    user_id: str
    provider: LLMProvider | None
    model: str
    instance: str | None
    file_workspace_id: uuid.UUID | None
    credential_workspace_id: uuid.UUID | None
    workspace_executor: CodeExecutor | None
    candidate_providers: tuple[LLMProvider | None, ...]
    upstream_user: str | None
    hosted_credential: Callable[[], Awaitable[ResolvedCredential | None]] | None = None
