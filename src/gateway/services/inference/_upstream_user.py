"""The end-user identifier a provider call carries, and the field each provider reads it from.

A provider that enforces its usage policy per end user needs an identifier per
end user, or a block meant for one person lands on the whole upstream account.
The identifier is a digest, so neither the caller's user tag (which may be
personal data) nor an Otari id reaches the provider.
"""

import hashlib
import hmac
import json
from typing import Any

from any_llm import LLMProvider

_DIGEST_VERSION = "otari-upstream-user-v1"

# Chat APIs that document ``user`` as an end-user id for abuse monitoring. any-llm
# sends DeepSeek's as DeepSeek's own ``user_id``. A provider not named in this module
# gets nothing, because an API that does not take ``user`` can fail the call.
_USER_FIELD = frozenset(
    {
        LLMProvider.OPENAI,
        LLMProvider.AZUREOPENAI,
        LLMProvider.CEREBRAS,
        LLMProvider.DEEPSEEK,
        LLMProvider.FIREWORKS,
        LLMProvider.GROQ,
        LLMProvider.OPENROUTER,
        LLMProvider.XAI,
    }
)
# Anthropic's API reads ``metadata.user_id`` and rejects ``user``.
_METADATA_USER_ID = frozenset({LLMProvider.ANTHROPIC, LLMProvider.AZUREANTHROPIC, LLMProvider.VERTEXAIANTHROPIC})
# Bedrock's abuse detection reads no end-user id, but ``requestMetadata`` reaches the
# operator's invocation logs, which is where an AWS abuse report's prompt ids are
# traced back to a user.
_BEDROCK_METADATA_KEY = "otari_user"


def upstream_user_identifier(workspace_id: str, subject: str, key: str | None) -> str:
    """The identifier for ``subject`` (a user tag or user id) within ``workspace_id``.

    Keyed with ``key`` when one is set, so the same user is not linkable across
    deployments; an unkeyed digest otherwise.
    """
    payload = json.dumps([_DIGEST_VERSION, workspace_id, subject], ensure_ascii=False, separators=(",", ":")).encode()
    if key is None:
        return hashlib.sha256(payload).hexdigest()
    # codeql[py/weak-sensitive-data-hashing]
    return hmac.new(key.encode(), payload, hashlib.sha256).hexdigest()


def chat_user_fields(provider: LLMProvider | str | None, identifier: str) -> dict[str, Any]:
    """The chat-completion keyword arguments that carry ``identifier`` to ``provider``."""
    try:
        resolved = LLMProvider(provider) if provider is not None else None
    except ValueError:
        return {}
    if resolved in _USER_FIELD:
        return {"user": identifier}
    if resolved in _METADATA_USER_ID:
        return {"metadata": {"user_id": identifier}}
    if resolved == LLMProvider.BEDROCK:
        return {"requestMetadata": {_BEDROCK_METADATA_KEY: identifier}}
    return {}
