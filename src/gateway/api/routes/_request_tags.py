"""Request tags: caller-supplied attribution read from a request's ``metadata``.

A tag is a short string pair (``purpose=chat``, ``country=DE``) that lands on
every usage row the request writes, so spend can be sliced by it later. Tags
come from the request's ``metadata`` object, OpenAI's own field for this, and
also from LiteLLM's nested ``metadata.spend_logs_metadata``, so a caller moving
off a LiteLLM proxy keeps its attribution without changing its requests.

The bounds are OpenAI's for ``metadata``: 16 pairs, keys up to 64 characters,
values up to 512.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import AfterValidator, Field

MAX_REQUEST_TAGS = 16
MAX_TAG_KEY_LENGTH = 64
MAX_TAG_VALUE_LENGTH = 512

# LiteLLM's key for spend-log attribution. The gateway consumes it, so it never
# reaches a provider, which would reject a nested object in ``metadata``.
LITELLM_SPEND_LOGS_KEY = "spend_logs_metadata"

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

    For the routes whose provider takes ``metadata`` itself (Anthropic's
    ``metadata.user_id``, the Responses API's stored metadata). The field is
    removed when nothing else was in it.
    """
    metadata = fields.get("metadata")
    if isinstance(metadata, dict) and LITELLM_SPEND_LOGS_KEY in metadata:
        rest = {k: v for k, v in metadata.items() if k != LITELLM_SPEND_LOGS_KEY}
        if rest:
            fields["metadata"] = rest
        else:
            fields.pop("metadata")
    return fields
