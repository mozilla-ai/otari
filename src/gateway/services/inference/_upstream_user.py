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
        LLMProvider.CEREBRAS,
        LLMProvider.DEEPSEEK,
        LLMProvider.FIREWORKS,
        LLMProvider.GROQ,
        LLMProvider.OPENROUTER,
        LLMProvider.XAI,
    }
)
# Chat APIs that document ``safety_identifier`` and deprecate ``user``.
_SAFETY_IDENTIFIER_FIELD = frozenset({LLMProvider.OPENAI, LLMProvider.AZUREOPENAI, LLMProvider.META})
# Anthropic's API reads ``metadata.user_id`` and rejects ``user``.
_METADATA_USER_ID = frozenset({LLMProvider.ANTHROPIC, LLMProvider.AZUREANTHROPIC, LLMProvider.VERTEXAIANTHROPIC})
# Bedrock's abuse detection reads no end-user id, but ``requestMetadata`` reaches the
# operator's invocation logs, which is where an AWS abuse report's prompt ids are
# traced back to a user.
_BEDROCK_METADATA_KEY = "otari_user"
# The Gemini API reads an aggregator's end-user id from the ``safety_identifier``
# label, whose value is at most 63 characters: one short of the digest. google-genai
# refuses a ``labels`` config field outside Vertex AI, so it goes in the request body.
# Vertex AI documents the same ``labels`` for billing only, so it is left out.
_GEMINI_LABEL_KEY = "safety_identifier"
_GEMINI_LABEL_VALUE_MAX = 63


def upstream_user_identifier(workspace_id: str, user_id: str | None, key: str | None, *, tag: str | None = None) -> str:
    """The identifier for ``user_id`` within ``workspace_id``, or for a ``tag`` of that user's own.

    A tag only subdivides its caller's identifier: the user is always digested, so
    no tag gives one user another's identifier, and a digest with a tag never
    equals one without. Keyed with ``key`` when one is set, so the same user is
    not linkable across deployments; an unkeyed digest otherwise.
    """
    parts: list[str | None] = [_DIGEST_VERSION, workspace_id, user_id]
    if tag is not None:
        parts.append(tag)
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    if key is None:
        return hashlib.sha256(payload).hexdigest()
    # codeql[py/weak-sensitive-data-hashing]
    return hmac.new(key.encode(), payload, hashlib.sha256).hexdigest()


class UnidentifiedEndUserError(Exception):
    """There is no workspace, or neither a user tag nor a user, to derive the identifier from."""


def end_user_identifier(
    *,
    key: str | None,
    workspace_id: str | None,
    tag: str | None,
    user_id: str | None,
    session: bool = False,
) -> str:
    """The identifier a request's provider calls carry.

    The caller's own user tag tells apart the end users behind the resolved user,
    except on a dashboard ``session``, which names no end user; a tag equal to the
    resolved user is no tag. A request with nothing to derive it from raises
    :class:`UnidentifiedEndUserError` rather than going upstream unscoped, where a
    block would land on the whole account.
    """
    if session or tag == user_id:
        tag = None
    if not workspace_id or not (tag or user_id):
        raise UnidentifiedEndUserError
    return upstream_user_identifier(workspace_id, user_id, key, tag=tag or None)


def bind_upstream_user(request_fields: dict[str, Any], identifier: str | None) -> dict[str, Any]:
    """Replace the caller's ``user`` and ``safety_identifier`` with ``identifier``, in place."""
    request_fields.pop("user", None)
    request_fields.pop("safety_identifier", None)
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
    if resolved in _SAFETY_IDENTIFIER_FIELD:
        return {"safety_identifier": identifier}
    if resolved in _METADATA_USER_ID:
        return {"metadata": {"user_id": identifier}}
    if resolved == LLMProvider.BEDROCK:
        return {"requestMetadata": {_BEDROCK_METADATA_KEY: identifier}}
    if resolved == LLMProvider.GEMINI:
        label = {_GEMINI_LABEL_KEY: identifier[:_GEMINI_LABEL_VALUE_MAX]}
        return {"http_options": {"extra_body": {"labels": label}}}
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
