"""The end-user identifier a provider call carries, and the field each provider reads it from.

A provider that enforces its usage policy per end user needs an identifier per
end user, or a block meant for one person lands on the whole upstream account.
The identifier is a digest, so neither the caller's user tag (which may be
personal data) nor an Otari id reaches the provider.
"""

import hashlib
import hmac
import json
from collections.abc import Callable
from typing import Any

from any_llm import LLMProvider

_DIGEST_VERSION = "otari-upstream-user-v1"

UPSTREAM_USER_FIELD = "_otari_upstream_user"
"""Carries the end-user identifier from the route to each attempt's provider call.

Never a caller field: each format adapter turns it into the field its provider
reads just before the call.
"""

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


class UnidentifiedEndUserError(Exception):
    """There is no workspace, or neither a user tag nor a user, to derive the identifier from."""


def end_user_identifier(
    *,
    enabled: bool,
    key: str | None,
    workspace_id: str | None,
    tag: str | None,
    user_id: str | None,
    session: bool = False,
) -> str | None:
    """The identifier a request's provider calls carry, or ``None`` while the feature is off.

    The caller's own user tag wins over the resolved user, except on a dashboard
    ``session``: it names no end user, so its tag would let a member pick another's
    identifier. While the feature is on, a request with nothing to derive it from
    raises :class:`UnidentifiedEndUserError` rather than going upstream unscoped,
    where a block would land on the whole account.
    """
    if not enabled:
        return None
    subject = user_id if session else tag or user_id
    if not workspace_id or not subject:
        raise UnidentifiedEndUserError
    return upstream_user_identifier(workspace_id, subject, key)


def bind_upstream_user(request_fields: dict[str, Any], identifier: str | None) -> dict[str, Any]:
    """Replace the caller's ``user`` with ``identifier``, in place."""
    request_fields.pop("user", None)
    request_fields.pop(UPSTREAM_USER_FIELD, None)
    if identifier is not None:
        request_fields[UPSTREAM_USER_FIELD] = identifier
    return request_fields


def with_upstream_user(
    kwargs: dict[str, Any], fields_for: Callable[[dict[str, Any], str], dict[str, Any]]
) -> dict[str, Any]:
    """A copy of ``kwargs`` with the carried identifier in the fields ``fields_for`` names for this call."""
    if UPSTREAM_USER_FIELD not in kwargs:
        return kwargs
    out = {key: value for key, value in kwargs.items() if key != UPSTREAM_USER_FIELD}
    out.update(fields_for(out, kwargs[UPSTREAM_USER_FIELD]))
    return out


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


def messages_user_fields(provider: LLMProvider | str | None, identifier: str, *, native: bool) -> dict[str, Any]:
    """The Messages keyword arguments that carry ``identifier`` to ``provider``.

    ``native`` says the call's model is served by a Messages API of its own.
    any-llm serves any other as a chat completion, where it drops ``metadata``
    and forwards extra keyword arguments to that call, so the provider's chat
    field is used instead.
    """
    if native:
        return {"metadata": {"user_id": identifier}}
    return chat_user_fields(provider, identifier)
