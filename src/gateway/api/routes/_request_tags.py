"""Request tags: caller-supplied attribution read from a request's ``metadata`` and ``Otari-Tags``.

A tag is a short string pair (``purpose=chat``, ``country=DE``) that lands on
every usage row the request writes, so spend can be sliced by it later. Tags
come from the request's ``metadata`` object, OpenAI's own field for this, and
also from LiteLLM's nested ``metadata.spend_logs_metadata``, so a caller moving
off a LiteLLM proxy keeps its attribution without changing its requests. The
``Otari-Tags`` header carries the same pairs for a client that can add a header
but not a body field, such as Claude Code; body tags win over it on a key.

The bounds are OpenAI's for ``metadata``: 16 pairs, keys up to 64 characters,
values up to 512.
"""

from __future__ import annotations

import re
from typing import Annotated, Any
from urllib.parse import unquote

from fastapi import Depends, Header, Request
from fastapi.exceptions import RequestValidationError
from pydantic import AfterValidator, Field

from gateway.core.config import TAGS_HEADER

MAX_REQUEST_TAGS = 16
MAX_TAG_KEY_LENGTH = 64
MAX_TAG_VALUE_LENGTH = 512

# LiteLLM's key for spend-log attribution. The gateway consumes it, so it never
# reaches a provider, which would reject a nested object in ``metadata``.
LITELLM_SPEND_LOGS_KEY = "spend_logs_metadata"

# A ``%`` that does not start a two-digit hex escape, which ``unquote`` would
# keep verbatim rather than refuse.
_BAD_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")

REQUEST_METADATA_DESC = (
    "Tags for cost attribution, recorded on the request's usage rows and filterable in the usage API: "
    f"up to {MAX_REQUEST_TAGS} string pairs, keys up to {MAX_TAG_KEY_LENGTH} characters and values up to "
    f"{MAX_TAG_VALUE_LENGTH}. A null value is ignored. LiteLLM's nested `{LITELLM_SPEND_LOGS_KEY}` object "
    "is also read, and wins over a flat key of the same name; it is never forwarded to the provider."
)


def _tag_pairs(source: dict[str, Any], where: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for key, value in source.items():
        if value is None:
            continue
        if not isinstance(value, str):
            msg = f"{where}.{key}: a tag value must be a string"
            raise ValueError(msg)
        if not key or len(key) > MAX_TAG_KEY_LENGTH:
            msg = f"{where}: a tag key must be 1 to {MAX_TAG_KEY_LENGTH} characters"
            raise ValueError(msg)
        if len(value) > MAX_TAG_VALUE_LENGTH:
            msg = f"{where}.{key}: a tag value must be at most {MAX_TAG_VALUE_LENGTH} characters"
            raise ValueError(msg)
        pairs[key] = value
    return pairs


# Anthropic's own ``metadata`` key on the Messages API, which names the billed
# user rather than a tag (see ``messages.py``).
ANTHROPIC_USER_KEY = "user_id"


def request_tags(metadata: dict[str, Any] | None, *, ignore: frozenset[str] = frozenset()) -> dict[str, str] | None:
    """Return the tags ``metadata`` carries, or None when it carries none.

    Keys in ``ignore`` belong to the wire format rather than the caller's tags,
    and are neither read nor checked. Raises ``ValueError`` when ``metadata``
    holds anything but string tags (and the one nested LiteLLM object), or more
    than the bounds allow.
    """
    if not metadata:
        return None
    flat = {k: v for k, v in metadata.items() if k != LITELLM_SPEND_LOGS_KEY and k not in ignore}
    tags = _tag_pairs(flat, "metadata")
    nested = metadata.get(LITELLM_SPEND_LOGS_KEY)
    if isinstance(nested, dict):
        tags.update(_tag_pairs(nested, f"metadata.{LITELLM_SPEND_LOGS_KEY}"))
    elif nested is not None:
        msg = f"metadata.{LITELLM_SPEND_LOGS_KEY}: must be an object"
        raise ValueError(msg)
    if len(tags) > MAX_REQUEST_TAGS:
        msg = f"metadata: at most {MAX_REQUEST_TAGS} tags"
        raise ValueError(msg)
    return tags or None


def _validated(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    request_tags(metadata)
    return metadata


def _validated_anthropic(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    request_tags(metadata, ignore=frozenset({ANTHROPIC_USER_KEY}))
    return metadata


# The ``metadata`` field of the chat and responses request schemas. It stays the
# object the caller sent, so a route that forwards it upstream forwards what it
# was given; ``request_tags`` reads the tags out.
RequestMetadata = Annotated[
    dict[str, Any] | None,
    AfterValidator(_validated),
    Field(description=REQUEST_METADATA_DESC),
]
# The same for the Messages API, where ``user_id`` is Anthropic's and not a tag.
AnthropicRequestMetadata = Annotated[
    dict[str, Any] | None,
    AfterValidator(_validated_anthropic),
    Field(description=f"{REQUEST_METADATA_DESC} `{ANTHROPIC_USER_KEY}` names the billed user and is not a tag."),
]


def forwarded_metadata(fields: dict[str, Any]) -> dict[str, Any]:
    """Drop the gateway-consumed LiteLLM key from ``fields["metadata"]`` in place.

    For the Responses API, whose provider stores ``metadata`` itself. The field
    is removed when nothing else was in it.
    """
    metadata = fields.get("metadata")
    if isinstance(metadata, dict) and LITELLM_SPEND_LOGS_KEY in metadata:
        rest = {k: v for k, v in metadata.items() if k != LITELLM_SPEND_LOGS_KEY}
        if rest:
            fields["metadata"] = rest
        else:
            fields.pop("metadata")
    return fields


TAGS_HEADER_DESC = (
    "Tags for cost attribution, as comma-separated `key=value` pairs (`agent=bug-fix,run=1234`), "
    "for a client that cannot set the body's `metadata`. Keys and values are percent-decoded, so either "
    "may carry an encoded `,` or `=`, and a non-ASCII character must be encoded. Whitespace around a pair "
    "is ignored, and a repeated header is read as one list. The bounds are the body's: "
    f"up to {MAX_REQUEST_TAGS} pairs, keys up to {MAX_TAG_KEY_LENGTH} characters and values up to "
    f"{MAX_TAG_VALUE_LENGTH}, counted after merging with the body's tags, which win on the same key. "
    "Never forwarded to the provider."
)


def header_tags(value: str | None) -> dict[str, str] | None:
    """Return the tags an ``Otari-Tags`` header value carries, or None when it carries none.

    Raises ``ValueError`` on a non-ASCII character, a pair without ``=``, a
    malformed percent escape, a key given twice, a value that does not
    percent-decode to UTF-8, or anything past the bounds.
    """
    if not value or not value.strip():
        return None
    if not value.isascii():
        # The server reads header bytes as Latin-1, so raw UTF-8 would be
        # recorded garbled; only the percent-encoded form is unambiguous.
        msg = f"{TAGS_HEADER}: percent-encode non-ASCII characters"
        raise ValueError(msg)
    pairs: dict[str, Any] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        raw_key, sep, raw_value = item.partition("=")
        if not sep:
            msg = f"{TAGS_HEADER}: each tag must be key=value"
            raise ValueError(msg)
        if _BAD_ESCAPE.search(item):
            msg = f"{TAGS_HEADER}: a % must start a two-digit hex escape"
            raise ValueError(msg)
        try:
            key = unquote(raw_key.strip(), errors="strict")
            tag_value = unquote(raw_value.strip(), errors="strict")
        except UnicodeDecodeError as exc:
            msg = f"{TAGS_HEADER}: a tag must percent-decode to UTF-8"
            raise ValueError(msg) from exc
        if key in pairs:
            msg = f"{TAGS_HEADER}.{key}: a tag key may appear once"
            raise ValueError(msg)
        pairs[key] = tag_value
    tags = _tag_pairs(pairs, TAGS_HEADER)
    if len(tags) > MAX_REQUEST_TAGS:
        msg = f"{TAGS_HEADER}: at most {MAX_REQUEST_TAGS} tags"
        raise ValueError(msg)
    return tags or None


def merge_tags(*layers: dict[str, str] | None) -> dict[str, str] | None:
    """Merge tag layers, a later layer winning on a key, within the bounds.

    Raises ``ValueError`` when the merged tags exceed ``MAX_REQUEST_TAGS``.
    """
    merged: dict[str, str] = {}
    for layer in layers:
        if layer:
            merged.update(layer)
    if len(merged) > MAX_REQUEST_TAGS:
        msg = f"{TAGS_HEADER} and metadata: at most {MAX_REQUEST_TAGS} tags together"
        raise ValueError(msg)
    return merged or None


def resolved_tags(
    header: str | None, metadata: dict[str, Any] | None, *, ignore: frozenset[str] = frozenset()
) -> dict[str, str] | None:
    """The tags to record for a request: the ``Otari-Tags`` header, overridden by the body's.

    Both sources were validated on their own at intake, so the one refusal left
    is their union exceeding the bound, which is answered like any other
    invalid request.
    """
    try:
        return merge_tags(header_tags(header), request_tags(metadata, ignore=ignore))
    except ValueError as exc:
        raise _header_error(exc) from exc


def _header_error(exc: ValueError) -> RequestValidationError:
    return RequestValidationError([{"type": "value_error", "loc": ("header", TAGS_HEADER), "msg": str(exc)}])


def _tags_header(
    raw_request: Request,
    _declared: Annotated[str | None, Header(alias=TAGS_HEADER, description=TAGS_HEADER_DESC)] = None,
) -> str | None:
    # Every ``Otari-Tags`` line the request sent, joined as one list: a header
    # that repeats means the same as its values comma-joined (RFC 9110, 5.3).
    # ``_declared`` is read only so the header appears in the OpenAPI spec.
    value = ",".join(raw_request.headers.getlist(TAGS_HEADER)) or None
    try:
        header_tags(value)
    except ValueError as exc:
        raise _header_error(exc) from exc
    return value


# The ``Otari-Tags`` request header, declared on each route that records tags.
RequestTagsHeader = Annotated[str | None, Depends(_tags_header)]
